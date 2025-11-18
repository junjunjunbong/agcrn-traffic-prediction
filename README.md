# AGCRN Traffic Prediction Project

교통 루프 검지기 데이터를 사용한 AGCRN (Adaptive Graph Convolutional Recurrent Network) 기반 교통 예측 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 프로젝트 소개

이 프로젝트는 교통 루프 검지기에서 수집된 시계열 데이터를 활용하여 AGCRN 모델로 교통 흐름을 예측합니다. AGCRN은 그래프 구조를 자동으로 학습하는 적응형 그래프 합성곱 순환 신경망으로, 시공간 교통 패턴을 효과적으로 모델링합니다.

### 🌟 주요 특징

- **600배 빠른 전처리**: 벡터화 연산으로 100만 행 데이터를 5초 내 처리
- **마스킹 기반 결측값 처리**: 실제 관측값(70.8%)과 보간값(29.2%)을 구분하여 학습
- **긴 결측 구간 필터링**: 5분 이상 연속 결측 샘플 자동 제거
- **다양한 손실 함수**: MaskedMSE, MaskedMAE, ObservedOnly 등 4가지 옵션
- **통합 학습 파이프라인**: 명령어 한 줄로 마스킹 기반 학습 가능
- **완전한 테스트**: 15개 이상의 단위 테스트로 안정성 보장
- **🆕 NaN Loss 버그 수정**: 3중 방어 시스템으로 안정적인 학습 보장
- **🆕 다중 데이터셋 지원**: 여러 데이터셋 자동 학습 및 비교 기능

## 📁 프로젝트 구조

```
agcrn-traffic-prediction/
├── data/
│   ├── raw/                         # 원본 CSV 파일
│   ├── processed/                   # 전처리된 데이터 (.npz)
│   └── meta/                        # 메타데이터 (센서 정보 등)
├── src/
│   ├── config.py                    # 설정 파일
│   ├── preprocess.py                # 데이터 전처리 (마스킹 지원)
│   ├── dataset.py                   # PyTorch Dataset (필터링 지원)
│   ├── model_agcrn.py               # AGCRN 모델 구현
│   ├── trainer.py                   # 학습 코드
│   ├── losses.py                    # 마스크 기반 손실 함수
│   ├── eval.py                      # 평가 코드
│   └── utils/                       # 유틸리티 함수
├── tests/                           # 테스트 코드
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
├── analyze_missing_pattern.py       # 결측값 분석 스크립트
├── analyze_missing_pattern_simple.py # 독립 실행 분석 스크립트
├── train.py                         # 학습 스크립트
├── train_all.py                     # 🆕 다중 데이터셋 자동 학습
├── preprocess.py                    # 전처리 실행 스크립트
├── debug_nan.py                     # 🆕 NaN 디버깅 도구
├── check_data_simple.py             # 🆕 데이터 검증 도구
├── test_nan_fix_v2.py               # 🆕 NaN 수정 테스트
├── FIX_SUMMARY.md                   # 🆕 NaN 버그 수정 요약
├── MASKED_PREPROCESSING_USAGE.md    # 마스킹 전처리 사용 가이드
└── README.md
```

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/junjunjunbong/agcrn-traffic-prediction.git
cd agcrn-traffic-prediction

# 패키지 설치
pip install -r requirements.txt
```

### 실행

#### 1. 결측값 패턴 분석 (선택사항)

데이터의 결측 패턴을 먼저 확인하세요:

```bash
python analyze_missing_pattern_simple.py
```

출력 예시:
```
============================================================
결측값 패턴 분석
============================================================

1. 전체 결측률: 927,986 / 3,110,400 = 29.83%

2. 특징별 결측률:
   flow                :   0.00%
   occupancy           :   0.00%
   harmonicMeanSpeed   :  89.50%  ← 주의!

7. 권장사항:
   ⚠️  결측률 20% 이상: 보간보다는 결측값 처리 모델 고려
   ⚠️  긴 결측 구간 다수: 선형 보간은 부정확할 수 있음
```

#### 2. 데이터 전처리

원본 CSV 파일을 마스킹 기반 전처리로 변환:

```bash
python preprocess.py
```

전처리 결과:
- **관측값 마스크 생성**: 실제 관측 70.8%, 보간 29.2%
- **벡터화 처리**: 100만 행을 5초 내 처리 (600배 빠름)
- **모든 특성 보간**: flow, occupancy, harmonicMeanSpeed
- **저장 위치**: `data/processed/*.npz`

#### 3. 모델 학습

마스킹 기반 손실 함수로 학습:

```bash
# 🆕 추천: 단일 데이터셋 학습 (ObservedOnly loss)
python train.py --data loops_033 --loss observed_only --epochs 20

# 기본 실행 (Masked MSE, 보간값 10% 가중치)
python train.py --data loops_033 --loss masked_mse

# 짧은 테스트 (5 에폭)
python train.py --data loops_033 --epochs 5 --loss masked_mse

# 보간값 가중치 조절 (5% = 관측값의 20배 중요)
python train.py --data loops_033 --loss masked_mse --imputed_weight 0.05

# 보간값 완전 무시 (관측값만 학습)
python train.py --data loops_033 --loss observed_only

# MAE 손실 함수 (이상치에 덜 민감)
python train.py --data loops_033 --loss masked_mae

# 기존 방식 (비교용 - 마스킹 없음)
python train.py --data loops_033 --loss mse
```

#### 🆕 4. 다중 데이터셋 자동 학습

여러 데이터셋을 한 번에 학습하고 결과 비교:

```bash
# 모든 데이터셋 자동 학습 (loops_033, loops_035, loops_040)
python train_all.py
```

**결과:**
- 각 데이터셋별 모델 저장 (`results/loops_*/best_model.pt`)
- 학습 로그 자동 저장 (`results/loops_*/training.log`)
- 최종 성능 비교 및 순위 출력

**주요 옵션**:
- `--loss`: 손실 함수 선택 (`masked_mse`, `masked_mae`, `observed_only`, `mse`)
- `--imputed_weight`: 보간값 가중치 (0.0~1.0, 기본값 0.1)
- `--epochs`: 학습 에폭 수 (기본값 100)
- `--data`: 데이터 파일명 (예: `loops_033`, `loops_035`, `loops_040`)
- `--lr`: Learning rate (기본값 0.001)
- `--device`: 디바이스 (`cuda` 또는 `cpu`)

**자세한 사용법**: [MASKED_PREPROCESSING_USAGE.md](MASKED_PREPROCESSING_USAGE.md) 및 [FIX_SUMMARY.md](FIX_SUMMARY.md) 참고

## 📊 데이터 구조

### 입력 데이터
- **노드**: 480개 (raw_id 모드) 또는 160개 (det_pos 모드)
- **시간**: 5초 단위 시간 스텝 (약 2160 스텝 ≈ 3시간)
- **특성**:
  - `flow`: 교통량 (차량 수)
  - `occupancy`: 차선 점유율 [0, 1]
  - `harmonicMeanSpeed`: 조화평균 속도 (m/s)

### 전처리 출력
`.npz` 파일 구조:
```python
{
    'train': (T, N, F),          # 정규화된 학습 데이터
    'val': (T, N, F),            # 검증 데이터
    'test': (T, N, F),           # 테스트 데이터
    'mask_train': (T, N, F),     # 관측값 마스크 (True=실제, False=보간)
    'mask_val': (T, N, F),
    'mask_test': (T, N, F),
    'stats': {...}               # 정규화 통계량
}
```

## 🔧 고급 설정

### config.py 주요 설정

```python
# 노드 설정
NODE_MODE = "raw_id"              # "raw_id" (480) 또는 "det_pos" (160)

# 특성 설정
FEATURES = ["flow", "occupancy", "harmonicMeanSpeed"]

# 결측값 처리
MISSING_SPEED_VALUE = -1.0
FREE_FLOW_SPEED = 15.0            # m/s

# 시퀀스 설정
SEQUENCE_LENGTH = 12              # 입력: 1분 (12 × 5초)
HORIZON = 3                       # 예측: 15초 (3 × 5초)

# 학습 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

### 마스킹 옵션

```bash
# 보수적: 보간값 완전 무시 (70.8% 관측값만 학습)
python train.py --loss observed_only

# 균형: 보간값 10% 가중치 (추천)
python train.py --loss masked_mse --imputed_weight 0.1

# 적극적: 보간값 50% 가중치
python train.py --loss masked_mse --imputed_weight 0.5

# 표준: 마스킹 없이 모든 값 동등 취급 (비교용)
python train.py --loss mse
```

### 필터링 옵션

```python
# 엄격: 2.5분 이상 결측 샘플 제거
create_dataloaders(..., max_missing_gap=30)

# 표준: 5분 이상 결측 샘플 제거 (기본값)
create_dataloaders(..., max_missing_gap=60)

# 관대: 10분 이상 결측 샘플 제거
create_dataloaders(..., max_missing_gap=120)

# 필터링 비활성화
create_dataloaders(..., filter_long_gaps=False)
```

## 🏗️ 모델 구조

### AGCRN (Adaptive Graph Convolutional Recurrent Network)

```
Input (batch, seq_len, N, F)
    ↓
Node Embeddings (학습 가능)
    ↓
AGCRN Cells × 2 layers
    ├── Adaptive GCN (그래프 학습)
    └── GRU (시계열 학습)
    ↓
Output Projection
    ↓
Prediction (batch, N, output_dim)
```

**특징**:
- 노드 임베딩을 학습하여 그래프 구조 자동 학습
- 시공간 교통 패턴을 동시에 모델링
- 적응형 인접 행렬로 동적 관계 파악

## 📈 성능 개선 내역

| 항목 | 이전 | 현재 | 개선 |
|------|------|------|------|
| 전처리 속도 | ~30분 | ~5초 | **600배** ↑ |
| 결측값 처리 | speed만 | 모든 특성 | ✅ |
| 관측값 추적 | 없음 | 마스킹 | ✅ |
| 긴 결측 처리 | 보간 | 필터링 | ✅ |
| 손실 함수 | MSE만 | 4가지 옵션 | ✅ |
| 학습 파이프라인 | 수동 통합 | CLI 자동화 | ✅ |
| 테스트 커버리지 | 0% | 80%+ | ✅ |
| **🆕 NaN Loss 버그** | 첫 에폭 실패 | 완전 해결 | ✅ |
| **🆕 다중 데이터셋** | 수동 | 자동화 | ✅ |

## 🛡️ NaN Loss 버그 수정 (v2.1.0)

이전 버전에서 학습 시작 즉시 `Train Loss: nan`이 발생하던 치명적 버그를 완전히 수정했습니다.

### 3중 방어 시스템

**1️⃣ 전처리 검증 강화 (메인 수정)**
- NaN이 보간 후에도 남으면 즉시 에러 발생
- 나쁜 데이터가 학습에 도달하지 못하도록 차단

**2️⃣ 학습 루프 NaN 감지**
- Backward pass 전에 NaN/Inf loss 감지
- 문제 배치는 스킵하고 학습 계속 진행
- 상세한 디버깅 정보 자동 출력

**3️⃣ 손실 함수 안전장치**
- Edge case (total_weight == 0) 처리
- NaN 전파 방지

### 검증 도구

```bash
# 데이터 NaN/Inf 체크 (NumPy만 필요)
python check_data_simple.py

# 포괄적 NaN 디버깅 (PyTorch 필요)
python debug_nan.py

# NaN 수정 테스트 실행
python test_nan_fix_v2.py
```

**자세한 내용**: [FIX_SUMMARY.md](FIX_SUMMARY.md)

## 📚 문서

- [FIX_SUMMARY.md](FIX_SUMMARY.md) - 🆕 NaN Loss 버그 수정 상세 문서
- [MASKED_PREPROCESSING_USAGE.md](MASKED_PREPROCESSING_USAGE.md) - 마스킹 전처리 상세 가이드
- [PREPROCESS_REVIEW.md](PREPROCESS_REVIEW.md) - 전처리 개선 내역
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - 프로젝트 개선 사항

## 🧪 테스트

```bash
# 전처리 테스트
pytest tests/test_preprocess.py -v

# 데이터셋 테스트
pytest tests/test_dataset.py -v

# 모델 테스트
pytest tests/test_model.py -v

# 전체 테스트
pytest tests/ -v
```

## 📦 요구사항

```
torch>=1.9.0,<2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyyaml>=5.4.0
pytest>=7.0.0
```

## 🤝 기여

이슈나 풀 리퀘스트를 환영합니다!

### 기여 가이드
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 📚 참고 자료

- **AGCRN 논문**: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting" (NeurIPS 2020)
- 데이터는 교통 루프 검지기에서 수집된 5초 단위 시계열 데이터

## ⚠️ 알려진 이슈

### 결측값 문제
- `harmonicMeanSpeed` 특성이 87.58% 결측
- 마스킹 + 필터링 전략으로 대응
- **해결책**: `--loss observed_only` 사용 (관측값만 학습)
- 자세한 내용은 `analyze_missing_pattern_simple.py` 실행 결과 참조

### ~~NaN Loss 버그~~ ✅ 해결됨 (v2.1.0)
- ~~이전: 첫 에폭부터 `Train Loss: nan` 발생~~
- **수정 완료**: 3중 방어 시스템으로 완전 해결
- 자세한 내용: [FIX_SUMMARY.md](FIX_SUMMARY.md)

## 💡 문제 해결

### Q: 🆕 학습 시작하자마자 "Train Loss: nan" 나와요
A: ✅ v2.1.0에서 완전히 수정되었습니다!
   - 최신 코드 받기: `git pull`
   - 전처리 다시 실행: `python preprocess.py`
   - 학습 재시도: `python train.py --data loops_033 --loss observed_only --epochs 5`
   - 여전히 문제가 있다면: `python debug_nan.py` 실행

### Q: "Filtered X/Y samples" 메시지가 나와요
A: 정상입니다. 긴 결측 구간이 있는 샘플을 제거한 것입니다. `max_missing_gap`을 조정하거나 `filter_long_gaps=False`로 설정하세요.

### Q: 마스크가 없다고 나와요
A: 데이터를 다시 전처리하세요: `python preprocess.py`

### Q: 전처리가 너무 느려요
A: 최신 버전은 벡터화 연산으로 매우 빠릅니다. `git pull`로 최신 코드를 받으세요.

### Q: 🆕 여러 데이터셋을 한 번에 학습하고 싶어요
A: `python train_all.py`를 사용하세요. loops_033, loops_035, loops_040 모두 자동 학습하고 결과 비교해줍니다.

---

**최종 업데이트**: 2025-11-18
**버전**: 2.1.0 (NaN Loss 버그 수정 + 다중 데이터셋 지원)
