from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("/workspace/data")

required = [
    DATA_DIR / "X_train_terrain14_accessibility_2021_2024.npy",
    DATA_DIR / "Y_train_2021_2024.npy",
    DATA_DIR / "sample_index_2021_2024.csv",
]

print("DATA_DIR =", DATA_DIR)

for path in required:
    print(path.name, "OK" if path.exists() else "MISSING")
    if not path.exists():
        raise FileNotFoundError(path)

x = np.load(required[0], mmap_mode="r")
y = np.load(required[1], mmap_mode="r")
idx = pd.read_csv(required[2])

print("X shape:", x.shape)
print("Y shape:", y.shape)
print("sample_index rows:", len(idx))
print("split counts:")
print(idx["split"].value_counts())

assert x.ndim == 4, x.shape
assert y.ndim == 4, y.shape
assert x.shape[1] == 14, x.shape
assert x.shape[2:] == (64, 64), x.shape
assert y.shape[1:] == (1, 64, 64), y.shape
assert len(idx) == x.shape[0] == y.shape[0]
assert set(["train", "val"]).issubset(set(idx["split"].astype(str))), idx["split"].value_counts()

print("data check OK")
