# 마스킹 기반 전처리 사용 가이드

## 개요

이 문서는 결측값 마스킹 + 필터링 전략을 사용하는 방법을 설명합니다.

### 구현된 기능

1. **결측값 마스크 생성**: 실제 관측값 vs 보간된 값 추적
2. **긴 결측 구간 필터링**: 5분 이상 연속 결측이 있는 시퀀스 제거
3. **마스크 기반 손실 함수**: 실제 관측값에 더 높은 가중치 부여
4. **유연한 설정**: 마스크 사용 여부, 필터링 강도 조절 가능

---

## 1. 데이터 전처리

### 기본 사용법

```python
from src.preprocess import main

# 전처리 실행 (마스크 자동 생성)
main()
```

### 출력

```
✓ Observation mask created: 70.17% real observations
```

전처리 완료 후 `data/processed/` 폴더에 다음 파일들이 생성됩니다:

```
loops_035_processed.npz:
  - train: 정규화된 학습 데이터 (T, N, F)
  - val: 검증 데이터
  - test: 테스트 데이터
  - mask_train: 학습 데이터 마스크 (True=관측, False=보간)
  - mask_val: 검증 데이터 마스크
  - mask_test: 테스트 데이터 마스크
  - stats: 정규화 통계량
```

---

## 2. 데이터 로딩 및 DataLoader 생성

### 방법 1: create_dataloaders 사용 (추천)

```python
from src.dataset import create_dataloaders

# 마스크 + 필터링 사용 (기본값)
train_loader, val_loader, test_loader = create_dataloaders(
    data_name='loops_035',
    batch_size=64,
    use_masks=True,          # 마스크 사용
    filter_long_gaps=True,   # 긴 결측 구간 필터링
    max_missing_gap=60       # 최대 허용 연속 결측 (60 = 5분)
)
```

**출력 예시:**
```
Loaded data with observation masks
Dataset created: 1450 samples from shape (1512, 480, 3)
  Observation rate: 70.84%
  Filtered 50/1500 samples with gaps > 60 timesteps
```

### 방법 2: 마스크 사용 안 함 (이전 방식)

```python
# 마스크 없이 사용
train_loader, val_loader, test_loader = create_dataloaders(
    data_name='loops_035',
    use_masks=False  # 마스크 미사용
)
```

### 방법 3: 필터링만 비활성화

```python
# 마스크는 사용하되 필터링 안 함
train_loader, val_loader, test_loader = create_dataloaders(
    data_name='loops_035',
    use_masks=True,
    filter_long_gaps=False  # 모든 샘플 사용
)
```

---

## 3. 배치 데이터 구조

### 마스크 사용시

```python
for batch in train_loader:
    x, y, masks = batch

    if masks is not None:
        mask_x, mask_y = masks

        print(x.shape)       # (batch, seq_len, N, F) = (64, 12, 480, 3)
        print(y.shape)       # (batch, horizon, N, F) = (64, 3, 480, 3)
        print(mask_x.shape)  # (64, 12, 480, 3)
        print(mask_y.shape)  # (64, 3, 480, 3)
```

### 마스크 미사용시

```python
for batch in train_loader:
    x, y, masks = batch

    print(masks)  # None
```

---

## 4. 손실 함수 사용

### 옵션 1: MaskedMSELoss (추천)

실제 관측값에 더 높은 가중치 부여

```python
from src.losses import MaskedMSELoss

criterion = MaskedMSELoss(
    imputed_weight=0.1  # 보간된 값에 10% 가중치
)

# 학습 루프
for x, y, masks in train_loader:
    pred = model(x)

    if masks is not None:
        _, mask_y = masks
        loss = criterion(pred, y[:, :, :, 0], mask_y[:, :, :, 0])  # 첫 번째 특징만 예측
    else:
        loss = criterion(pred, y[:, :, :, 0])
```

**가중치 설정:**
- `imputed_weight=0.0`: 보간값 완전 무시 (가장 보수적)
- `imputed_weight=0.1`: 보간값에 10% 가중치 (추천)
- `imputed_weight=0.5`: 보간값에 50% 가중치
- `imputed_weight=1.0`: 모든 값 동일 가중치 (일반 MSE와 동일)

### 옵션 2: ObservedOnlyLoss (가장 보수적)

보간된 값은 완전히 무시

```python
from src.losses import ObservedOnlyLoss

criterion = ObservedOnlyLoss(loss_fn='mse')

# 사용법은 동일
loss = criterion(pred, target, mask)
```

### 옵션 3: 일반 손실 함수

```python
import torch.nn as nn

criterion = nn.MSELoss()

# 마스크 무시
loss = criterion(pred, target)
```

---

## 5. 실전 예시

### 전체 학습 루프

```python
import torch
from src.dataset import create_dataloaders
from src.losses import MaskedMSELoss
from src.model_agcrn import AGCRN

# 1. 데이터 로더 생성
train_loader, val_loader, test_loader = create_dataloaders(
    'loops_035',
    batch_size=64,
    use_masks=True,
    filter_long_gaps=True,
    max_missing_gap=60
)

# 2. 모델 및 손실 함수 초기화
model = AGCRN()
criterion = MaskedMSELoss(imputed_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 학습
model.train()
for epoch in range(100):
    epoch_loss = 0

    for x, y, masks in train_loader:
        x, y = x.cuda(), y.cuda()

        # Forward
        pred = model(x)  # (batch, N, output_dim)

        # 손실 계산 (마스크 활용)
        if masks is not None:
            _, mask_y = masks
            mask_y = mask_y.cuda()
            # 첫 번째 타임스텝, 첫 번째 특징만 예측
            loss = criterion(
                pred[:, :, 0],           # (batch, N)
                y[:, 0, :, 0],           # (batch, N)
                mask_y[:, 0, :, 0]       # (batch, N)
            )
        else:
            loss = criterion(pred[:, :, 0], y[:, 0, :, 0])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.4f}')
```

---

## 6. 설정 권장사항

### 시나리오별 추천 설정

#### 시나리오 1: 안전 우선 (결측률 > 20%)

```python
train_loader = create_dataloaders(
    'loops_035',
    use_masks=True,
    filter_long_gaps=True,
    max_missing_gap=30  # 2.5분 (더 엄격)
)

criterion = ObservedOnlyLoss(loss_fn='mse')  # 보간값 완전 무시
```

#### 시나리오 2: 균형잡힌 접근 (결측률 10-20%)

```python
train_loader = create_dataloaders(
    'loops_035',
    use_masks=True,
    filter_long_gaps=True,
    max_missing_gap=60  # 5분
)

criterion = MaskedMSELoss(imputed_weight=0.1)  # 보간값 10% 가중치
```

#### 시나리오 3: 데이터 최대 활용 (결측률 < 10%)

```python
train_loader = create_dataloaders(
    'loops_035',
    use_masks=True,
    filter_long_gaps=False  # 필터링 안 함
)

criterion = MaskedMSELoss(imputed_weight=0.5)  # 보간값 50% 가중치
```

---

## 7. 기대 효과

### 개선 전 (단순 선형 보간)

```
✗ 89.5% 속도 데이터 결측 → 대부분 보간값
✗ 83분 연속 결측도 선형 보간
✗ 가짜 패턴 학습 가능
```

### 개선 후 (마스킹 + 필터링)

```
✓ 70.8% 실제 관측값 사용
✓ 긴 결측 구간 샘플 제거 (예: 2,573개 5분+ 구간)
✓ 손실 함수가 실제 관측값 우선
✓ 더 신뢰할 수 있는 모델 학습
```

---

## 8. 문제 해결

### Q: "Filtered X/Y samples" 메시지가 나와요

A: 정상입니다. 긴 결측 구간이 있는 샘플을 제거한 것입니다.
   - 너무 많이 제거되면 `max_missing_gap`을 늘리세요
   - 예: `max_missing_gap=120` (10분)

### Q: 마스크가 없다고 나와요

A: 데이터를 다시 전처리하세요:
   ```python
   from src.preprocess import main
   main()
   ```

### Q: 모든 샘플이 필터링되어요

A: `max_missing_gap`이 너무 작거나 데이터 품질이 낮습니다:
   ```python
   # 필터링 비활성화
   create_dataloaders(..., filter_long_gaps=False)
   ```

---

## 9. 요약

| 구성 요소 | 기능 | 기본값 |
|---------|------|--------|
| `use_masks` | 마스크 사용 여부 | `True` |
| `filter_long_gaps` | 긴 결측 필터링 | `True` |
| `max_missing_gap` | 최대 허용 연속 결측 | `60` (5분) |
| `imputed_weight` | 보간값 가중치 | `0.1` (10%) |

**핵심 메시지**:
이 전처리 전략은 29% 결측률, 89% 속도 결측이라는 열악한 데이터 품질 속에서도
신뢰할 수 있는 모델 학습을 가능하게 합니다.
