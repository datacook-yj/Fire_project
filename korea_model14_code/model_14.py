
"""
14채널 terrain + accessibility U-Net 학습 스크립트

필요 파일:
- X_train_terrain14_accessibility_2021_2024.npy
- Y_train_2021_2024.npy
- sample_index_2021_2024.csv
- model.py

출력 파일:
- unet_terrain14_accessibility_best.pth
- terrain14_accessibility_train_history.csv
"""

from pathlib import Path
import argparse
import random
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


FEATURE_NAMES = [
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


class WildfireNpyDataset(Dataset):
    def __init__(self, x_path, y_path, indices):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    x_path = data_dir / "X_train_terrain14_accessibility_2021_2024.npy"
    y_path = data_dir / "Y_train_2021_2024.npy"
    idx_path = data_dir / "sample_index_2021_2024.csv"

    save_path = data_dir / "unet_terrain14_accessibility_best.pth"
    history_path = data_dir / "terrain14_accessibility_train_history.csv"

    print("data_dir:", data_dir)
    print("save_path:", save_path)

    for p in [x_path, y_path, idx_path]:
        print(p.name, p.exists())
        if not p.exists():
            raise FileNotFoundError(p)

    x_mem = np.load(x_path, mmap_mode="r")
    y_mem = np.load(y_path, mmap_mode="r")
    sample_df = pd.read_csv(idx_path)

    print("X shape:", x_mem.shape)
    print("Y shape:", y_mem.shape)
    print(sample_df["split"].value_counts())

    assert x_mem.shape[1] == 14, f"14채널 데이터가 아닙니다: {x_mem.shape}"
    assert y_mem.shape[1:] == (1, 64, 64)
    assert len(sample_df) == x_mem.shape[0] == y_mem.shape[0]

    train_idx = sample_df.index[sample_df["split"] == "train"].to_numpy()
    val_idx = sample_df.index[sample_df["split"] == "val"].to_numpy()

    print("train samples:", len(train_idx), "Y fire pixels:", int(y_mem[train_idx].sum()))
    print("val samples:", len(val_idx), "Y fire pixels:", int(y_mem[val_idx].sum()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = WildfireNpyDataset(x_path, y_path, train_idx)
    val_ds = WildfireNpyDataset(x_path, y_path, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet_6Ch(in_channels=14, base_ch=args.base_ch).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    print("parameter count:", f"{sum(p.numel() for p in model.parameters()):,}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    print()
    print("14채널 U-Net 학습 시작")
    print(f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>9} | {'AUC-PR':>8} | {'F1':>8} | {'IoU':>8} | {'thr':>8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **val_result,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"{epoch:5d} | "
            f"{train_loss:10.4f} | "
            f"{val_result['val_loss']:9.4f} | "
            f"{val_result['auc_pr']:8.4f} | "
            f"{val_result['f1']:8.4f} | "
            f"{val_result['iou']:8.4f} | "
            f"{val_result['threshold']:8.4f}"
        )

        if val_result["auc_pr"] > best_auc:
            best_auc = val_result["auc_pr"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": 14,
                    "base_ch": args.base_ch,
                    "epoch": epoch,
                    "val_auc_pr": val_result["auc_pr"],
                    "val_threshold": val_result["threshold"],
                    "val_f1": val_result["f1"],
                    "val_iou": val_result["iou"],
                    "feature_names": FEATURE_NAMES,
                },
                save_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"\nEarly stopping: {args.patience} epoch 동안 AUC-PR 개선 없음")
            break

    print("\n학습 완료")
    print("best epoch:", best_epoch)
    print("best val AUC-PR:", best_auc)
    print("saved model:", save_path)
    print("history:", history_path)

    ckpt = torch.load(save_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    first_conv = None
    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 4:
            first_conv = (k, tuple(v.shape))
            break

    print("\n체크포인트 확인")
    print("first conv:", first_conv)
    print("in_channels:", ckpt.get("in_channels"))
    print("val_threshold:", ckpt.get("val_threshold"))


if __name__ == "__main__":
    main()
