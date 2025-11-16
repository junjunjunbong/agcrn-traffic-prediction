# AGCRN Traffic Prediction Project

교통 루프 검지기 데이터를 사용한 AGCRN (Adaptive Graph Convolutional Recurrent Network) 기반 교통 예측 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 프로젝트 소개

이 프로젝트는 교통 루프 검지기에서 수집된 시계열 데이터를 활용하여 AGCRN 모델로 교통 흐름을 예측합니다. AGCRN은 그래프 구조를 자동으로 학습하는 적응형 그래프 합성곱 순환 신경망으로, 시공간 교통 패턴을 효과적으로 모델링합니다.

## 프로젝트 구조

```
DL_PROJECT/
├─ data/
│  ├─ raw/                    # 원본 CSV 파일
│  ├─ processed/              # 전처리된 데이터 (.npz)
│  └─ meta/                   # 메타데이터 (센서 정보 등)
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # 설정 파일
│  ├─ preprocess.py           # 데이터 전처리
│  ├─ dataset.py             # PyTorch Dataset
│  ├─ model_agcrn.py         # AGCRN 모델 구현
│  ├─ trainer.py             # 학습 코드
│  └─ eval.py                # 평가 코드
├─ experiments/              # 실험 노트북
├─ configs/                  # 설정 파일 (YAML)
├─ logs/                     # 학습 로그
├─ saved_models/             # 저장된 모델
├─ train.py                  # 학습 스크립트
├─ preprocess.py             # 전처리 스크립트
└─ README.md
```

## 사용 방법

### 1. 데이터 전처리

먼저 원본 CSV 파일을 전처리하여 AGCRN에 맞는 형태로 변환합니다:

```bash
python preprocess.py
```

이 스크립트는:
- CSV 파일을 읽어서 (T, N, F) 텐서로 변환
- 결측치 보간 (속도 -1 처리)
- 정규화 및 train/val/test 분할
- `data/processed/` 폴더에 저장

### 2. 모델 학습

```bash
python train.py --data loops_035 --batch_size 64 --lr 0.001 --epochs 100
```

주요 옵션:
- `--data`: 사용할 데이터 이름 (예: loops_035)
- `--batch_size`: 배치 크기
- `--lr`: 학습률
- `--epochs`: 에포크 수

### 3. 평가

```python
from src.eval import evaluate_model, load_model
from src.config import NUM_NODES, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS

# 모델 로드
model = load_model(
    model_path="saved_models/best_agcrn.pt",
    model_config={
        "num_nodes": NUM_NODES,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
)

# 평가
metrics, predictions, targets = evaluate_model(model, test_loader)
print(metrics)
```

## 데이터 구조

- **노드**: 각 `raw_id`를 하나의 노드로 사용 (480개 노드)
- **시간**: 5초 단위 시간 스텝 (약 2160 스텝 ≈ 3시간)
- **특성**: flow, occupancy, harmonicMeanSpeed

## 모델 구조

- **AGCRN**: Adaptive Graph Convolutional Recurrent Network
- 노드 임베딩을 학습하여 그래프 구조를 자동으로 학습
- 시공간 교통 패턴을 동시에 모델링

## 설정 변경

`src/config.py` 파일에서 다음을 변경할 수 있습니다:

- `NODE_MODE`: "raw_id" (480 노드) 또는 "det_pos" (160 노드)
- `FEATURES`: 사용할 특성 선택
- `SEQUENCE_LENGTH`: 입력 시퀀스 길이
- `HORIZON`: 예측 범위

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/agcrn-traffic-prediction.git
cd agcrn-traffic-prediction

# 패키지 설치
pip install -r requirements.txt
```

### 실행

```bash
# 1. 데이터 전처리
python preprocess.py

# 2. 모델 학습
python train.py --data loops_035 --batch_size 64 --lr 0.001 --epochs 100
```

## 📦 요구사항

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyyaml>=5.4.0
```

## 📚 참고 자료

- **AGCRN 논문**: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
- 데이터는 교통 루프 검지기에서 수집된 5초 단위 시계열 데이터

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 👥 기여

이슈나 풀 리퀘스트를 환영합니다!

