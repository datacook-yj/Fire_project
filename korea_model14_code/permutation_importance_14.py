"""
Permutation importance for 14-channel terrain + accessibility U-Net.

Required files in --data_dir:
- model.py
- X_train_terrain14_accessibility_2021_2024.npy
- Y_train_2021_2024.npy
- sample_index_2021_2024.csv
- unet_terrain14_accessibility_best.pth

Outputs:
- permutation_importance_14.csv
- permutation_importance_14_aucpr.png
"""

from pathlib import Path
import argparse
import sys
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


class ValPermutationDataset(Dataset):
    def __init__(self, x_path, y_path, indices, perm_channel=None, perm_indices=None):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.perm_channel = perm_channel
        self.perm_indices = None if perm_indices is None else np.asarray(perm_indices, dtype=np.int64)

        if self.perm_channel is not None:
            if self.perm_indices is None:
                raise ValueError("perm_indices is required when perm_channel is set")
            if len(self.perm_indices) != len(self.indices):
                raise ValueError("perm_indices length must match indices length")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)

        if self.perm_channel is not None:
            perm_idx = int(self.perm_indices[i])
            x[self.perm_channel] = np.array(self.x[perm_idx, self.perm_channel], dtype=np.float32, copy=True)

        return torch.from_numpy(x), torch.from_numpy(y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    y_true_parts = []
    y_score_parts = []

    use_amp = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x)
        else:
            logits = model(x)

        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        yy = y.numpy().astype(np.uint8)

        y_score_parts.append(prob.reshape(-1))
        y_true_parts.append(yy.reshape(-1))

    y_true = np.concatenate(y_true_parts).astype(np.uint8)
    y_score = np.concatenate(y_score_parts).astype(np.float32)
    return y_true, y_score


def evaluate_scores(y_true, y_score, threshold):
    auc_pr = float(average_precision_score(y_true, y_score))
    tm = threshold_metrics(y_true, y_score, threshold)
    return {"auc_pr": auc_pr, **tm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default="unet_terrain14_accessibility_best.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_val_samples", type=int, default=0, help="0 means use all validation samples")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    x_path = data_dir / "X_train_terrain14_accessibility_2021_2024.npy"
    y_path = data_dir / "Y_train_2021_2024.npy"
    idx_path = data_dir / "sample_index_2021_2024.csv"
    ckpt_path = data_dir / args.checkpoint

    for p in [x_path, y_path, idx_path, ckpt_path, data_dir / "model.py"]:
        print(f"{p.name:55s} exists={p.exists()}")
        if not p.exists():
            raise FileNotFoundError(p)

    sys.path.insert(0, str(data_dir))
    from model import UNet_6Ch

    x_mem = np.load(x_path, mmap_mode="r")
    y_mem = np.load(y_path, mmap_mode="r")
    sample_df = pd.read_csv(idx_path)

    print("X shape:", x_mem.shape)
    print("Y shape:", y_mem.shape)
    print(sample_df["split"].value_counts())

    assert x_mem.shape[1] == 14, f"Expected 14 channels, got {x_mem.shape}"
    assert y_mem.shape[1:] == (1, 64, 64)
    assert len(sample_df) == x_mem.shape[0] == y_mem.shape[0]

    val_idx = sample_df.index[sample_df["split"] == "val"].to_numpy()
    if args.max_val_samples and args.max_val_samples > 0:
        val_idx = val_idx[:args.max_val_samples]
        print(f"Using first {len(val_idx)} validation samples for a quick run")

    print("val samples:", len(val_idx))
    print("val fire pixels:", int(y_mem[val_idx].sum()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    base_ch = int(ckpt.get("base_ch", 32)) if isinstance(ckpt, dict) else 32
    in_channels = int(ckpt.get("in_channels", 14)) if isinstance(ckpt, dict) else 14
    feature_names = ckpt.get("feature_names", FEATURE_NAMES) if isinstance(ckpt, dict) else FEATURE_NAMES

    if in_channels != 14:
        raise ValueError(f"Checkpoint in_channels is {in_channels}, expected 14")

    model = UNet_6Ch(in_channels=14, base_ch=base_ch).to(device)
    model.load_state_dict(state)
    model.eval()

    first_conv = None
    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 4:
            first_conv = (k, tuple(v.shape))
            break
    print("first conv:", first_conv)

    ckpt_threshold = ckpt.get("val_threshold", None) if isinstance(ckpt, dict) else None

    # Original validation performance.
    original_ds = ValPermutationDataset(x_path, y_path, val_idx)
    original_loader = DataLoader(
        original_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\n[0] Original validation prediction")
    y_true, y_score = predict_scores(model, original_loader, device)

    if ckpt_threshold is None:
        threshold = best_f1_threshold(y_true, y_score)
        print("threshold from current validation predictions:", threshold)
    else:
        threshold = float(ckpt_threshold)
        print("threshold from checkpoint:", threshold)

    base_metrics = evaluate_scores(y_true, y_score, threshold)
    print("Original AUC-PR:", f"{base_metrics['auc_pr']:.6f}")
    print("Original F1:", f"{base_metrics['f1']:.6f}")
    print("Original IoU:", f"{base_metrics['iou']:.6f}")

    rng = np.random.default_rng(args.seed)
    rows = []

    for ch, name in enumerate(feature_names):
        perm_indices = rng.permutation(val_idx)
        perm_ds = ValPermutationDataset(
            x_path,
            y_path,
            val_idx,
            perm_channel=ch,
            perm_indices=perm_indices,
        )
        perm_loader = DataLoader(
            perm_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        print(f"\n[{ch:02d}] Permuting {name}")
        yt, ys = predict_scores(model, perm_loader, device)
        met = evaluate_scores(yt, ys, threshold)

        row = {
            "channel": ch,
            "feature": name,
            "baseline_auc_pr": base_metrics["auc_pr"],
            "permuted_auc_pr": met["auc_pr"],
            "delta_auc_pr": base_metrics["auc_pr"] - met["auc_pr"],
            "baseline_f1": base_metrics["f1"],
            "permuted_f1": met["f1"],
            "delta_f1": base_metrics["f1"] - met["f1"],
            "baseline_iou": base_metrics["iou"],
            "permuted_iou": met["iou"],
            "delta_iou": base_metrics["iou"] - met["iou"],
            "threshold": threshold,
            "baseline_precision": base_metrics["precision"],
            "permuted_precision": met["precision"],
            "baseline_recall": base_metrics["recall"],
            "permuted_recall": met["recall"],
        }
        rows.append(row)

        print(
            f"  AUC-PR {met['auc_pr']:.6f} | delta {row['delta_auc_pr']:.6f} | "
            f"F1 {met['f1']:.6f} | IoU {met['iou']:.6f}"
        )

    out_df = pd.DataFrame(rows).sort_values("delta_auc_pr", ascending=False)
    csv_path = data_dir / "permutation_importance_14.csv"
    png_path = data_dir / "permutation_importance_14_aucpr.png"

    out_df.to_csv(csv_path, index=False)

    plot_df = out_df.sort_values("delta_auc_pr", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["delta_auc_pr"])
    plt.xlabel("AUC-PR decrease after permutation")
    plt.ylabel("Feature")
    plt.title("Permutation Importance on Validation Set")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:", csv_path)
    print("Saved:", png_path)
    print("\nTop 5 important features:")
    print(out_df[["channel", "feature", "delta_auc_pr", "delta_f1", "delta_iou"]].head(5).to_string(index=False))
    print("\nLowest 5 features:")
    print(out_df[["channel", "feature", "delta_auc_pr", "delta_f1", "delta_iou"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
