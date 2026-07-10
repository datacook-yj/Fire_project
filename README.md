# Korea Wildfire Spread Prediction U-Net

한국 산불확산 예측을 위한 U-Net 기반 연구 코드입니다.  
이 연구는 Day T의 산불 화점, 기상, 지형, 접근성 정보를 입력으로 사용하여 Day T+1의 산불 화점 확률지도를 64×64 격자로 예측하는 것을 목표로 합니다.

---

## 1. Research Overview

- 문제 설정: Day T 입력 → Day T+1 산불 화점 마스크 예측
- 입력 격자: 64×64
- 출력 격자: 1×64×64 산불 화점 확률지도
- 주요 모델: U-Net
- 주요 비교 흐름: 6채널 baseline → 13채널 terrain-aware 모델 → 14채널 terrain + accessibility 모델
- 평가 기준: validation AUC-PR, F1, IoU, threshold
- 원칙: feature 선택과 threshold 선택은 validation set에서만 수행하고, test set은 마지막 최종 평가에만 사용합니다.

---

## 2. Dataset

이 연구에서 사용하는 데이터는 산불 화점, 기상, 지형, 접근성 정보로 구성됩니다.

| 구분 | 자료 | 모델에서 쓰는 형태 |
|---|---|---|
| 산불 화점 | NASA FIRMS VIIRS | 오늘 화점 마스크 `fire_mask_t`, 내일 정답 `Y` |
| 기상 | 기상청 ASOS 일자료 | 기온, 습도, 동서 바람, 남북 바람 |
| 지형 | SRTM DEM GeoTIFF | 고도, 경사도, 사면방향, TPI, 상대고도, 거칠기 |
| 접근성 | 소방서/119안전센터 위치 | 가장 가까운 소방시설까지 거리 |

### 14-channel input features

| Channel | Feature | Description |
|---:|---|---|
| 0 | `fire_mask_t` | 오늘 산불 화점 마스크 |
| 1 | `temperature` | 기온 |
| 2 | `humidity` | 습도 |
| 3 | `u_wind` | 동서 방향 바람 성분 |
| 4 | `v_wind` | 남북 방향 바람 성분 |
| 5 | `dem_elevation` | DEM 고도 |
| 6 | `slope_deg` | 경사도 |
| 7 | `aspect_sin` | 사면 방향 sin 값 |
| 8 | `aspect_cos` | 사면 방향 cos 값 |
| 9 | `tpi_5x5` | 지형 위치 지수 |
| 10 | `relative_elevation` | 주변 대비 상대 고도 |
| 11 | `roughness_5x5` | 지형 거칠기 |
| 12 | `wind_slope_alignment` | 바람-사면 정렬도 |
| 13 | `nearest_fire_station_distance` | 가장 가까운 소방시설까지 거리 |

### Expected data files

```text
data/
  X_train_terrain14_accessibility_2021_2024.npy
  Y_train_2021_2024.npy
  sample_index_2021_2024.csv
```

예상 shape는 다음과 같습니다.

```text
X: (13140, 14, 64, 64)
Y: (13140, 1, 64, 64)
```

`sample_index_2021_2024.csv`에는 각 sample의 지역, 날짜, split 정보가 포함됩니다.

| Split | Samples | Y fire pixels |
|---|---:|---:|
| train | 9,855 | 7,364 |
| val | 1,638 | 1,070 |
| test | 1,647 | 1,056 |

---

## 3. Repository Files

| File | Description |
|---|---|
| `all_data.py` / `all_merge.py` | ASOS, DEM, FIRMS 데이터를 이용해 6채널 X/Y를 생성하는 전처리 코드입니다. |
| `baseline_us_fixed.py` | Huot et al. 논문 형식의 미국 TFRecord 데이터셋에 대한 Persistence, Logistic Regression, Random Forest baseline 재현 코드입니다. 한국 14채널 U-Net의 메인 학습 코드는 아닙니다. |
| `model.py` | U-Net 구조, SEBlock, Focal Loss, Dice Loss, CombinedLoss가 정의된 모델 파일입니다. |
| `model_14.py` | 14채널 terrain + accessibility U-Net의 기본 학습 스크립트입니다. 이 연구의 메인 학습 파일입니다. |
| `model_14_resume.py` | 14채널 모델을 중간 checkpoint부터 이어 학습하기 위한 스크립트입니다. 서버 중단 상황에 대비한 보조 파일입니다. |
| `permutation_importance_14.py` | 14채널 전체 모델의 validation set 기준 permutation importance를 계산합니다. |
| `model_ablation_resume.py` | 특정 feature를 제외한 채널 조합으로 U-Net을 다시 학습하는 ablation 실험 코드입니다. |
| `permutation_importance_selected.py` | ablation 모델 또는 선택된 feature 조합 모델에 대해 permutation importance를 다시 계산합니다. |
| `vis.py` | 학습된 모델의 예측 결과를 시각화하거나 샘플 단위 평가를 확인하는 보조 코드입니다. |

---

## 4. Local Environment Setup

Python 가상환경을 만든 뒤 필요한 패키지를 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU를 사용할 경우, 로컬 CUDA 환경에 맞는 PyTorch가 설치되어 있어야 합니다.  
CUDA 설정이 되어 있지 않으면 CPU로도 실행은 가능하지만 학습 속도가 느릴 수 있습니다.

---

## 5. Local Data Check

데이터 파일이 `./data` 폴더에 있는지 먼저 확인합니다.

```bash
ls -lh data/X_train_terrain14_accessibility_2021_2024.npy
ls -lh data/Y_train_2021_2024.npy
ls -lh data/sample_index_2021_2024.csv
```

간단한 shape 확인 예시는 다음과 같습니다.

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
X = np.load(DATA_DIR / 'X_train_terrain14_accessibility_2021_2024.npy', mmap_mode='r')
Y = np.load(DATA_DIR / 'Y_train_2021_2024.npy', mmap_mode='r')
idx = pd.read_csv(DATA_DIR / 'sample_index_2021_2024.csv')

print('X shape:', X.shape)
print('Y shape:', Y.shape)
print(idx['split'].value_counts())
PY
```

---

## 6. Train 14-channel U-Net

14채널 U-Net을 학습합니다.

```bash
python -u model_14.py \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32
```

주요 출력 파일은 `./data` 폴더에 저장됩니다.

```text
data/unet_terrain14_accessibility_best.pth
data/terrain14_accessibility_train_history.csv
```

`model_14.py`는 validation AUC-PR이 가장 좋은 epoch의 checkpoint를 저장합니다.

---

## 7. Current 14-channel Result

현재 14채널 terrain + accessibility U-Net의 validation 기준 결과는 다음과 같습니다.

| Metric | Value |
|---|---:|
| best epoch | 2 |
| validation AUC-PR | 0.4149 |
| validation F1 | 0.5202 |
| validation IoU | 0.3515 |
| validation threshold | 0.1439 |

---

## 8. Permutation Importance

학습된 14채널 모델에 대해 validation set 기준 permutation importance를 계산합니다.

```bash
python -u permutation_importance_14.py \
  --data_dir ./data \
  --batch_size 64 \
  --num_workers 4
```

Permutation importance는 특정 feature 채널을 섞었을 때 validation AUC-PR이 얼마나 감소하는지로 계산합니다.  
이 값은 feature 삭제를 확정하기 위한 기준이 아니라, ablation 후보를 찾기 위한 해석용 지표입니다.

현재 중요도가 크게 나온 feature는 다음과 같습니다.

| Feature | Meaning | AUC-PR drop |
|---|---|---:|
| `aspect_cos` | 사면 방향 cos | 0.1337 |
| `aspect_sin` | 사면 방향 sin | 0.0982 |
| `fire_mask_t` | 오늘 화점 마스크 | 0.0954 |
| `slope_deg` | 경사도 | 0.0119 |
| `nearest_fire_station_distance` | 최근접 소방시설 거리 | 0.0109 |
| `roughness_5x5` | 지형 거칠기 | 0.0083 |

---

## 9. Ablation Study

Ablation은 원본 `.npy` 파일을 삭제하거나 수정하지 않고, 코드에서 특정 입력 채널만 제외한 뒤 새 모델을 다시 학습하는 방식으로 진행합니다.

예시: temperature 제거

```bash
python -u model_ablation_resume.py \
  --data_dir ./data \
  --run_name ablate_1_drop_temp \
  --drop_features temperature \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 4
```

Ablation 후보는 다음과 같습니다.

| Step | Dropped feature(s) | Description |
|---:|---|---|
| 1 | `temperature` | 기온 단독 제거 |
| 2 | `dem_elevation` | DEM 고도 단독 제거 |
| 3 | `temperature`, `v_wind`, `u_wind` | 기온 + 바람 변수 제거 |
| 4 | `temperature`, `v_wind`, `u_wind`, `dem_elevation` | DEM 단독 결과에 따라 포함 여부 결정 |
| 5 | `temperature`, `v_wind`, `u_wind`, `wind_slope_alignment`, `dem_elevation` | 기상 + 바람-사면 정렬도 + 조건부 DEM 제거 |

선택된 feature 조합 모델에 대한 permutation importance 재계산 예시는 다음과 같습니다.

```bash
python -u permutation_importance_selected.py \
  --data_dir ./data \
  --run_name ablate_1_drop_temp \
  --batch_size 64 \
  --num_workers 4
```

---

## 10. Research Rules

- feature 선택과 threshold 선택은 validation set에서만 수행합니다.
- test set은 마지막 최종 평가용으로만 사용합니다.
- `duration_hours`, `duration_days`, `burned_area_ha`처럼 산불 종료 후에만 알 수 있는 사후 정보는 모델 입력으로 사용하지 않습니다.
- permutation importance는 feature 삭제 확정 기준이 아니라 ablation 후보 탐색용입니다.
- 최종 feature 조합은 ablation 재학습 후 validation 성능으로 결정합니다.

---
