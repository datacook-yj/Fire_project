"""
Feature ablation training for 14-channel terrain + accessibility U-Net.

This script keeps the original 14-channel NPY file, but selects only requested
channels in code. It is intended for validation-based ablation experiments.

Required files in --data_dir:
- model.py
- X_train_terrain14_accessibility_2021_2024.npy
- Y_train_2021_2024.npy
- sample_index_2021_2024.csv

Example:
python -u model_ablation_resume.py --data_dir . --run_name ablate_drop_temp_vwind \
  --drop_features temperature v_wind --epochs 30 --batch_size 32 --num_workers 4
"""

from pathlib import Path
import argparse
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import UNet_6Ch, CombinedLoss

FEATURE_NAMES_14 = [
    "fire_mask_t",
    "temperature",
    "humidity",
    "u_wind",
    "v_wind",
    "dem_elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "tpi_5x5",
    "relative_elevation",
    "roughness_5x5",
    "wind_slope_alignment",
    "nearest_fire_station_distance",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "ablation"


def parse_feature_list(items):
    """Allow both '--drop a b' and '--drop a,b' styles."""
    out = []
    for item in items or []:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def build_keep_indices(drop_features, keep_features):
    drop_features = parse_feature_list(drop_features)
    keep_features = parse_feature_list(keep_features)

    unknown = sorted(set(drop_features + keep_features) - set(FEATURE_NAMES_14))
    if unknown:
        raise ValueError(f"Unknown feature name(s): {unknown}\nValid names: {FEATURE_NAMES_14}")

    if drop_features and keep_features:
        raise ValueError("Use either --drop_features or --keep_features, not both.")

    if keep_features:
        keep_idx = [FEATURE_NAMES_14.index(name) for name in keep_features]
    else:
        keep_idx = [i for i, name in enumerate(FEATURE_NAMES_14) if name not in set(drop_features)]

    selected_names = [FEATURE_NAMES_14[i] for i in keep_idx]
    dropped_names = [name for name in FEATURE_NAMES_14 if name not in selected_names]

    if not keep_idx:
        raise ValueError("No input features selected.")

    return keep_idx, selected_names, dropped_names


class WildfireSelectedDataset(Dataset):
    def __init__(self, x_path, y_path, indices, keep_idx):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.keep_idx = np.asarray(keep_idx, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.array(self.x[idx, self.keep_idx, :, :], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_i = int(np.nanargmax(f1))
    return float(thresholds[best_i]), float(f1[best_i])


def threshold_metrics(y_true, y_score, threshold):
    pred = y_score >= threshold
    true = y_true == 1
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_pred": int(pred.sum()),
        "n_true": int(true.sum()),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    y_true_parts = []
    y_score_parts = []
    use_amp = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        yy = y.detach().cpu().numpy().astype(np.uint8)
        y_score_parts.append(prob.reshape(-1))
        y_true_parts.append(yy.reshape(-1))
        loss_sum += float(loss.item())
        n_batches += 1

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    auc_pr = float(average_precision_score(y_true, y_score))
    threshold, best_f1 = best_f1_threshold(y_true, y_score)
    tm = threshold_metrics(y_true, y_score, threshold)
    return {
        "val_loss": loss_sum / max(n_batches, 1),
        "auc_pr": auc_pr,
        "threshold": threshold,
        "best_f1": best_f1,
        **tm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--drop_features", nargs="*", default=[])
    parser.add_argument("--keep_features", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    run_name = safe_name(args.run_name)

    x_path = data_dir / "X_train_terrain14_accessibility_2021_2024.npy"
    y_path = data_dir / "Y_train_2021_2024.npy"
    idx_path = data_dir / "sample_index_2021_2024.csv"
    save_path = data_dir / f"unet_{run_name}_best.pth"
    history_path = data_dir / f"{run_name}_train_history.csv"
    last_path = data_dir / f"{run_name}_last_checkpoint.pth"

    for p in [data_dir / "model.py", x_path, y_path, idx_path]:
        print(f"{p.name:55s} exists={p.exists()}")
        if not p.exists():
            raise FileNotFoundError(p)

    keep_idx, selected_names, dropped_names = build_keep_indices(args.drop_features, args.keep_features)
    in_channels = len(keep_idx)

    print("\nAblation run:", run_name)
    print("input channels:", in_channels)
    print("keep_idx:", keep_idx)
    print("selected features:", selected_names)
    print("dropped features:", dropped_names)
    print("save_path:", save_path)

    x_mem = np.load(x_path, mmap_mode="r")
    y_mem = np.load(y_path, mmap_mode="r")
    sample_df = pd.read_csv(idx_path)
    print("X shape:", x_mem.shape)
    print("Y shape:", y_mem.shape)
    print(sample_df["split"].value_counts())
    assert x_mem.shape[1] == 14, f"Expected original 14-channel X, got {x_mem.shape}"
    assert y_mem.shape[1:] == (1, 64, 64)
    assert len(sample_df) == x_mem.shape[0] == y_mem.shape[0]

    train_idx = sample_df.index[sample_df["split"] == "train"].to_numpy()
    val_idx = sample_df.index[sample_df["split"] == "val"].to_numpy()
    print("train samples:", len(train_idx), "Y fire pixels:", int(y_mem[train_idx].sum()))
    print("val samples:", len(val_idx), "Y fire pixels:", int(y_mem[val_idx].sum()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = WildfireSelectedDataset(x_path, y_path, train_idx, keep_idx)
    val_ds = WildfireSelectedDataset(x_path, y_path, val_idx, keep_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = UNet_6Ch(in_channels=in_channels, base_ch=args.base_ch).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    print("parameter count:", f"{sum(p.numel() for p in model.parameters()):,}")

    use_amp = device.type == "cuda"
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []
    start_epoch = 1

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume used but last checkpoint does not exist: {last_path}")
        print("Loading resume checkpoint:", last_path)
        ckpt = torch.load(last_path, map_location=device)
        old_selected = ckpt.get("feature_names", None)
        if old_selected is not None and list(old_selected) != list(selected_names):
            raise ValueError("Selected features in checkpoint do not match current selected features.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_auc = float(ckpt.get("best_auc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        history = ckpt.get("history", [])
        print(f"Resume from epoch {start_epoch}; best epoch={best_epoch}, best AUC-PR={best_auc:.6f}")
        if start_epoch > args.epochs:
            print(f"Already completed through epoch {start_epoch - 1}. Increase --epochs to continue.")
            return

    print("\nFeature ablation U-Net training start")
    print(f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>9} | {'AUC-PR':>8} | {'F1':>8} | {'IoU':>8} | {'thr':>8}")
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)
        val_result = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"], **val_result}
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"{epoch:5d} | {train_loss:10.4f} | {val_result['val_loss']:9.4f} | "
            f"{val_result['auc_pr']:8.4f} | {val_result['f1']:8.4f} | "
            f"{val_result['iou']:8.4f} | {val_result['threshold']:8.4f}"
        )

        if val_result["auc_pr"] > best_auc:
            best_auc = val_result["auc_pr"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "base_ch": args.base_ch,
                    "epoch": epoch,
                    "val_auc_pr": val_result["auc_pr"],
                    "val_threshold": val_result["threshold"],
                    "val_f1": val_result["f1"],
                    "val_iou": val_result["iou"],
                    "feature_names": selected_names,
                    "feature_names_14": FEATURE_NAMES_14,
                    "keep_idx": keep_idx,
                    "dropped_features": dropped_names,
                    "run_name": run_name,
                },
                save_path,
            )
        else:
            bad_epochs += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "in_channels": in_channels,
                "base_ch": args.base_ch,
                "feature_names": selected_names,
                "feature_names_14": FEATURE_NAMES_14,
                "keep_idx": keep_idx,
                "dropped_features": dropped_names,
                "run_name": run_name,
                "last_val_result": val_result,
            },
            last_path,
        )

        if bad_epochs >= args.patience:
            print(f"\nEarly stopping: no AUC-PR improvement for {args.patience} epochs")
            break

    print("\nTraining complete")
    print("run_name:", run_name)
    print("best epoch:", best_epoch)
    print("best val AUC-PR:", best_auc)
    print("best model:", save_path)
    print("history:", history_path)
    print("last checkpoint:", last_path)

    ckpt = torch.load(save_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    first_conv = None
    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 4:
            first_conv = (k, tuple(v.shape))
            break
    print("first conv:", first_conv)
    print("in_channels:", ckpt.get("in_channels"))
    print("selected features:", ckpt.get("feature_names"))
    print("dropped features:", ckpt.get("dropped_features"))
    print("val threshold:", ckpt.get("val_threshold"))


if __name__ == "__main__":
    main()
