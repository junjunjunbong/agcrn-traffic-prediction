# 데이터 전처리 코드 리뷰 및 개선 계획

**작성일**: 2025-11-16
**파일**: `src/preprocess.py`
**상태**: 🔴 Critical Issues Found

---

## 📊 현재 코드 분석

### 전처리 흐름

```
CSV 파일 (loops*.csv, ~100만 행)
    ↓
1. load_csv_data()              - DataFrame 로드
    ↓
2. create_node_mapping()        - 노드 매핑 생성 (raw_id or det_pos)
    ↓
3. create_time_index()          - 시간 인덱스 생성
    ↓
4. convert_to_tensor()          - (T, N, F) 텐서 변환 ⚠️ 문제!
    ↓
5. interpolate_missing_speed()  - harmonicMeanSpeed 보간 ⚠️ 문제!
    ↓
6. split_data()                 - Train/Val/Test 분할 (70/15/15)
    ↓
7. normalize_data()             - Z-score 정규화 ⚠️ 문제!
    ↓
8. Save to .npz                 - 저장
```

---

## 🔴 Critical Issues (즉시 수정 필요)

### Issue #1: det_pos 모드에서 데이터 덮어쓰기

**위치**: `convert_to_tensor()`, Line 101-123

**문제**:
```python
for _, row in df.iterrows():
    if NODE_MODE == "det_pos":
        node_id = f"det_pos_{row['det_pos']}"
    # ...
    X[t_idx, n_idx, f_idx] = val  # ❌ 같은 위치 여러 차선 → 마지막 것만 저장
```

**영향**:
- det_pos=0에 3개 차선(lane_idx=0,1,2)이 있으면 lane_idx=2만 저장됨
- 실제로는 차선별 데이터를 집계해야 함

**예시**:
```
det_pos=0, lane_idx=0: flow=10, occupancy=0.3
det_pos=0, lane_idx=1: flow=15, occupancy=0.4
det_pos=0, lane_idx=2: flow=12, occupancy=0.35

현재 코드: flow=12, occupancy=0.35 (마지막 것만)
올바른 처리: flow=37 (합), occupancy=0.35 (평균)
```

**해결 방안**:
```python
# Before: iterrows로 덮어쓰기
for _, row in df.iterrows():
    X[t_idx, n_idx, f_idx] = val

# After: groupby로 집계
if NODE_MODE == "det_pos":
    grouped = df.groupby(['begin', 'det_pos']).agg({
        'flow': 'sum',           # 교통량은 합산
        'occupancy': 'mean',     # 점유율은 평균
        'harmonicMeanSpeed': 'mean'  # 속도는 평균
    })
```

**우선순위**: 🔴 High (데이터 정확성 문제)

---

### Issue #2: iterrows() 성능 문제

**위치**: `convert_to_tensor()`, Line 101

**문제**:
- iterrows()는 row-by-row iteration으로 매우 느림
- 100만 행 처리 시 **10-30분** 소요 예상

**벤치마크**:
```python
# iterrows() 방식
for _, row in df.iterrows():  # ~30분
    X[t_idx, n_idx, f_idx] = row[feature]

# vectorized 방식
pivot = df.pivot_table(...)   # ~1-5초 (600배 빠름!)
X[:, :, f_idx] = pivot.values
```

**해결 방안**:
```python
def convert_to_tensor_fast(df, node_to_idx, features):
    """
    Pivot table을 사용한 빠른 변환
    """
    # 노드 컬럼 설정
    node_col = 'raw_id' if NODE_MODE == 'raw_id' else 'det_pos'

    # 각 feature별로 pivot
    tensor_list = []
    for feature in features:
        pivot = df.pivot_table(
            values=feature,
            index='begin',
            columns=node_col,
            aggfunc='mean' if NODE_MODE == 'raw_id' else
                   ('sum' if feature == 'flow' else 'mean')
        )
        tensor_list.append(pivot.values)

    # (T, N, F) 형태로 stack
    X = np.stack(tensor_list, axis=2)
    return X
```

**우선순위**: 🔴 High (사용자 경험 심각 저하)

---

### Issue #3: flow/occupancy 결측값 미처리

**위치**: `interpolate_missing_speed()`, Line 129-182

**문제**:
- harmonicMeanSpeed만 보간
- flow, occupancy는 NaN 그대로 방치
- 모델 학습 시 NaN으로 인한 에러 또는 성능 저하

**데이터 확인 필요**:
```python
# CSV에서 빈 값이 있는지 확인
df['flow'].isna().sum()        # ?
df['occupancy'].isna().sum()   # ?
```

**해결 방안**:
```python
def interpolate_all_features(X: np.ndarray) -> np.ndarray:
    """
    모든 특성에 대해 결측값 보간
    """
    X_interp = X.copy()

    for f_idx in range(X.shape[2]):  # 각 특성
        for n in range(X.shape[1]):  # 각 노드
            series = X[:, n, f_idx]

            if np.all(np.isnan(series)):
                # 모든 값이 NaN인 경우 0으로
                X_interp[:, n, f_idx] = 0
                continue

            # 시계열 선형 보간
            interpolated = pd.Series(series).interpolate(
                method='linear',
                limit_direction='both',
                fill_value=0
            )
            X_interp[:, n, f_idx] = interpolated.values

    return X_interp
```

**우선순위**: 🔴 High (모델 학습 실패 가능)

---

## 🟡 Important Issues (단기 개선)

### Issue #4: 결측값 추론 로직 버그

**위치**: `interpolate_missing_speed()`, Line 162-163

**문제**:
```python
# flow/occupancy가 NaN인 경우 0으로 설정
flow_val = X[t, n, flow_idx] if not np.isnan(X[t, n, flow_idx]) else 0
occ_val = X[t, n, occ_idx] if not np.isnan(X[t, n, occ_idx]) else 0

# 문제: NaN → 0 → "no vehicles" 조건 충족 → 부적절한 추론
if flow_val < 0.1 and occ_val < 0.1:
    X_interp[t, n, speed_idx] = FREE_FLOW_SPEED
```

**해결 방안**:
```python
# NaN이면 건너뛰기
if np.isnan(X[t, n, flow_idx]) or np.isnan(X[t, n, occ_idx]):
    # 해당 노드의 평균 사용
    valid_speeds = speed_series[~np.isnan(speed_series)]
    X_interp[t, n, speed_idx] = np.mean(valid_speeds) if len(valid_speeds) > 0 else FREE_FLOW_SPEED
    continue

flow_val = X[t, n, flow_idx]
occ_val = X[t, n, occ_idx]
# ... 기존 로직
```

**우선순위**: 🟡 Medium

---

### Issue #5: 데이터 검증 부족

**위치**: 전체

**문제**:
- 입력 데이터 검증 없음
- 변환 후 데이터 검증 없음
- 에러 발생 시 원인 파악 어려움

**추가 필요**:
```python
def validate_input_data(df: pd.DataFrame) -> None:
    """입력 CSV 데이터 검증"""
    # 1. 필수 컬럼 확인
    required_cols = ['begin', 'end', 'raw_id', 'det_pos', 'flow',
                     'occupancy', 'harmonicMeanSpeed']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2. 시간 연속성 확인
    time_steps = sorted(df['begin'].unique())
    time_diff = np.diff(time_steps)
    if not np.allclose(time_diff, TIME_STEP_SIZE):
        warnings.warn("Non-uniform time steps detected")

    # 3. 값 범위 확인
    if df['flow'].min() < 0:
        raise ValueError("flow cannot be negative")

    if not df['occupancy'].between(0, 1, inclusive='both').all():
        warnings.warn("occupancy values outside [0, 1] range")

def validate_tensor(X: np.ndarray, name: str) -> None:
    """텐서 검증"""
    # 1. Shape 확인
    assert X.ndim == 3, f"{name} must be 3D (T, N, F)"

    # 2. NaN 확인
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        raise ValueError(f"{name} contains {nan_count} NaN values")

    # 3. Inf 확인
    if np.any(np.isinf(X)):
        raise ValueError(f"{name} contains inf values")

    print(f"✓ {name} validation passed: shape={X.shape}")
```

**우선순위**: 🟡 Medium

---

### Issue #6: 정규화 전 NaN 검증 부족

**위치**: `normalize_data()`, Line 185-217

**문제**:
```python
# np.nanmean 사용 → NaN이 있어도 에러 없이 진행
mean = np.nanmean(train_feat)
std = np.nanstd(train_feat)

# 문제: NaN이 많으면 통계가 부정확
```

**해결 방안**:
```python
def normalize_data(X_train, X_val, X_test):
    """
    Z-score normalization with strict validation
    """
    # NaN 검증 (정규화 전에 모든 NaN 제거되어야 함)
    for split_name, split_data in [('train', X_train), ('val', X_val), ('test', X_test)]:
        nan_count = np.isnan(split_data).sum()
        if nan_count > 0:
            raise ValueError(f"{split_name} contains {nan_count} NaN values before normalization")

    stats = {}
    # ... 기존 로직 (np.mean 사용, np.nanmean 아님)
```

**우선순위**: 🟡 Medium

---

## 🟢 Nice to Have (장기 개선)

### Issue #7: 메모리 효율성

**현재**: 전체 DataFrame을 메모리에 로드 (~100MB+)

**개선 방안**:
```python
# Chunk 단위 처리
chunks = []
for chunk in pd.read_csv(csv_path, chunksize=50000):
    processed = process_chunk(chunk)
    chunks.append(processed)
X = np.concatenate(chunks, axis=0)
```

**우선순위**: 🟢 Low

---

### Issue #8: 병렬 처리 부재

**개선 방안**:
```python
from multiprocessing import Pool

def process_node(n):
    """단일 노드 보간"""
    # ...

# 병렬 처리
with Pool(processes=4) as pool:
    results = pool.map(process_node, range(num_nodes))
```

**우선순위**: 🟢 Low

---

## 📋 개선 계획

### Phase 1: Critical Fixes (즉시)

**목표**: 기능적 문제 해결

1. ✅ **vectorized convert_to_tensor 구현**
   - pivot_table 사용
   - 600배 성능 향상
   - 예상 소요: 1-2시간

2. ✅ **det_pos 모드 집계 수정**
   - groupby + agg 사용
   - flow: sum, occupancy/speed: mean
   - 예상 소요: 30분

3. ✅ **모든 특성 결측값 처리**
   - interpolate_all_features() 구현
   - 3개 특성 모두 보간
   - 예상 소요: 1시간

4. ✅ **데이터 검증 추가**
   - validate_input_data()
   - validate_tensor()
   - 예상 소요: 1시간

### Phase 2: Quality Improvements (단기)

**목표**: 안정성 및 신뢰성 향상

5. ✅ **결측값 추론 로직 수정**
   - NaN 처리 버그 수정
   - 예상 소요: 30분

6. ✅ **정규화 검증 강화**
   - NaN 체크 추가
   - 예상 소요: 30분

7. ✅ **전처리 테스트 추가**
   - test_preprocess.py
   - 유닛 테스트 작성
   - 예상 소요: 2시간

### Phase 3: Optimization (장기)

**목표**: 성능 및 확장성

8. ⏳ **메모리 최적화**
   - Chunk 처리
   - 예상 소요: 2시간

9. ⏳ **병렬 처리 추가**
   - multiprocessing
   - 예상 소요: 2-3시간

---

## 🎯 수정 후 예상 효과

| 항목 | Before | After | 개선 |
|------|--------|-------|------|
| 처리 시간 (100만 행) | ~30분 | ~5초 | **360배** ↑ |
| det_pos 정확도 | 데이터 손실 | 올바른 집계 | ✅ |
| 결측값 처리 | speed만 | 모든 특성 | ✅ |
| 데이터 검증 | 없음 | 완전 검증 | ✅ |
| 테스트 커버리지 | 0% | ~80% | ✅ |
| 에러 디버깅 | 어려움 | 명확한 메시지 | ✅ |

---

## 📝 구현 체크리스트

### Phase 1 구현 항목

- [ ] `convert_to_tensor_vectorized()` 함수 작성
- [ ] det_pos 모드 집계 로직 추가
- [ ] `interpolate_all_features()` 함수 작성
- [ ] `validate_input_data()` 함수 추가
- [ ] `validate_tensor()` 함수 추가
- [ ] 기존 `process_single_file()` 함수 업데이트
- [ ] 에러 메시지 개선
- [ ] 진행 상황 로깅 추가

### 테스트 항목

- [ ] test_convert_to_tensor_raw_id()
- [ ] test_convert_to_tensor_det_pos()
- [ ] test_interpolate_all_features()
- [ ] test_validate_input_data()
- [ ] test_normalize_data()
- [ ] test_split_data()
- [ ] test_full_pipeline()

### 문서화

- [ ] docstring 추가/개선
- [ ] README에 전처리 가이드 추가
- [ ] 예제 노트북 작성

---

## 🔍 참고 사항

### 데이터 특성

- **시간 간격**: 5초
- **노드 수**: 480 (raw_id) 또는 160 (det_pos)
- **특성**: flow, occupancy, harmonicMeanSpeed
- **CSV 크기**: ~75MB per file
- **행 수**: ~100만 행

### 전처리 파라미터

```python
# config.py
NODE_MODE = "raw_id"              # or "det_pos"
FEATURES = ["flow", "occupancy", "harmonicMeanSpeed"]
TIME_STEP_SIZE = 5.0              # seconds
MISSING_SPEED_VALUE = -1.0
FREE_FLOW_SPEED = 15.0            # m/s
CONGESTED_SPEED = 2.5             # m/s
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

---

**Last Updated**: 2025-11-16
**Status**: Ready for Implementation
**Estimated Total Time**: Phase 1 = 4-5 hours
