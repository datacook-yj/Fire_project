# Korea 14-channel Terrain + Accessibility U-Net Docker 실행 방법

## 1. 폴더 구조

압축을 풀면 아래 구조가 되어야 합니다.

korea/
  Dockerfile
  requirements.txt
  .dockerignore
  README.md
  check_data.py
  model.py
  model_14.py
  permutation_importance_14.py
  model_ablation_resume.py
  permutation_importance_selected.py

  data/
    X_train_terrain14_accessibility_2021_2024.npy
    Y_train_2021_2024.npy
    sample_index_2021_2024.csv
    results/

## 2. Docker image build

프로젝트 폴더에서 실행합니다.

docker build -t wildfire-unet:model14 .

## 3. GPU 확인

docker run --rm --gpus all wildfire-unet:model14 nvidia-smi

docker run --rm --gpus all wildfire-unet:model14 \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

## 4. 데이터 확인

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  python -u check_data.py

## 5. 14채널 U-Net 학습

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  bash -lc "set -euo pipefail; mkdir -p /workspace/data/results; python -u model_14.py --data_dir /workspace/data --epochs 30 --batch_size 32 2>&1 | tee /workspace/data/results/train_model14.log; cp -f /workspace/data/unet_terrain14_accessibility_best.pth /workspace/data/results/; cp -f /workspace/data/terrain14_accessibility_train_history.csv /workspace/data/results/"

## 6. 결과 확인

학습 후 결과는 아래에 생성됩니다.

data/unet_terrain14_accessibility_best.pth
data/terrain14_accessibility_train_history.csv

그리고 복사본과 로그는 아래에 저장됩니다.

data/results/unet_terrain14_accessibility_best.pth
data/results/terrain14_accessibility_train_history.csv
data/results/train_model14.log

## 7. 14채널 permutation importance

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  bash -lc "set -euo pipefail; cp -f /workspace/data/results/unet_terrain14_accessibility_best.pth /workspace/data/ 2>/dev/null || true; python -u permutation_importance_14.py --data_dir /workspace/data --batch_size 64 --num_workers 4 2>&1 | tee /workspace/data/results/permutation_14.log"

## 8. Ablation 예시

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  bash -lc "set -euo pipefail; mkdir -p /workspace/data/results; python -u model_ablation_resume.py --data_dir /workspace/data --run_name ablate_1_drop_temp --drop_features temperature --epochs 30 --batch_size 32 --num_workers 4 2>&1 | tee /workspace/data/results/ablate_1_drop_temp.log"

## 9. Ablation 모델의 선택 feature permutation importance

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  bash -lc "set -euo pipefail; python -u permutation_importance_selected.py --data_dir /workspace/data --run_name ablate_1_drop_temp --batch_size 64 --num_workers 4 2>&1 | tee /workspace/data/results/permutation_selected_ablate_1_drop_temp.log"

## 주의사항

- Docker image 안에는 데이터 파일을 넣지 않습니다.
- data 폴더는 docker run에서 volume mount로 연결합니다.
- model_14.py 기준 기본 학습에는 --num_workers 옵션을 넣지 않습니다.
- test set은 최종 평가 전까지 사용하지 않습니다.
