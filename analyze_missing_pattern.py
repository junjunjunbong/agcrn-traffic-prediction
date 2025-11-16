"""
결측값 패턴 분석 스크립트

결측값 보간이 정당한지 판단하기 위해:
1. 결측률 분석
2. 결측 패턴 분석 (랜덤 vs 체계적)
3. 보간 전후 통계 비교
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocess import load_csv_data, create_node_mapping, convert_to_tensor_vectorized, create_time_index
from src.config import RAW_DATA_DIR


def analyze_missing_pattern():
    """결측값 패턴 분석"""

    # 데이터 로드
    csv_files = list(RAW_DATA_DIR.glob("loops*.csv"))
    if not csv_files:
        print("No data files found")
        return

    df = load_csv_data(csv_files[0])
    sensors_df, node_to_idx = create_node_mapping(df)
    time_steps = create_time_index(df)
    X = convert_to_tensor_vectorized(df, node_to_idx, time_steps)

    T, N, F = X.shape
    feature_names = ['flow', 'occupancy', 'harmonicMeanSpeed']

    print("="*60)
    print("결측값 패턴 분석")
    print("="*60)

    # 1. 전체 결측률
    total_missing = np.isnan(X).sum()
    total_size = X.size
    print(f"\n1. 전체 결측률: {total_missing:,} / {total_size:,} = {100*total_missing/total_size:.2f}%")

    # 2. 특징별 결측률
    print("\n2. 특징별 결측률:")
    for f_idx, feat_name in enumerate(feature_names):
        missing = np.isnan(X[:, :, f_idx]).sum()
        total = X[:, :, f_idx].size
        print(f"   {feat_name:20s}: {100*missing/total:6.2f}%")

    # 3. 노드별 결측률 분포
    print("\n3. 노드별 결측률:")
    node_missing_rate = np.isnan(X).mean(axis=(0, 2)) * 100
    print(f"   평균: {node_missing_rate.mean():.2f}%")
    print(f"   최소: {node_missing_rate.min():.2f}%")
    print(f"   최대: {node_missing_rate.max():.2f}%")
    print(f"   표준편차: {node_missing_rate.std():.2f}%")

    high_missing_nodes = (node_missing_rate > 30).sum()
    print(f"   30% 이상 결측: {high_missing_nodes} / {N} 노드")

    # 4. 시간별 결측률
    print("\n4. 시간별 결측률:")
    time_missing_rate = np.isnan(X).mean(axis=(1, 2)) * 100
    print(f"   평균: {time_missing_rate.mean():.2f}%")
    print(f"   최대: {time_missing_rate.max():.2f}%")

    high_missing_times = (time_missing_rate > 50).sum()
    print(f"   50% 이상 결측 시간: {high_missing_times} / {T}")

    # 5. 연속 결측 패턴
    print("\n5. 연속 결측 패턴 (harmonicMeanSpeed):")
    speed_data = X[:, :, 2]  # harmonicMeanSpeed

    consecutive_missing = []
    for n in range(N):
        series = speed_data[:, n]
        is_missing = np.isnan(series)

        # 연속 결측 구간 찾기
        changes = np.diff(np.concatenate([[False], is_missing, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            consecutive_missing.append(end - start)

    if consecutive_missing:
        consecutive_missing = np.array(consecutive_missing)
        print(f"   평균 연속 결측: {consecutive_missing.mean():.1f} 타임스텝")
        print(f"   최대 연속 결측: {consecutive_missing.max()} 타임스텝")
        print(f"   중앙값: {np.median(consecutive_missing):.1f} 타임스텝")

        # 긴 연속 결측 (5분 이상 = 60 타임스텝)
        long_gaps = (consecutive_missing >= 60).sum()
        print(f"   5분 이상 연속 결측: {long_gaps}회")

    # 6. 보간 정당성 판단
    print("\n6. 보간 정당성 평가:")
    print("   ✓ 전체 결측률 < 10%: ", "예" if total_missing/total_size < 0.1 else "아니오")
    print("   ✓ 대부분 짧은 결측 (< 1분): ", "예" if np.median(consecutive_missing) < 12 else "아니오")
    print("   ✓ 노드 균등 분포: ", "예" if node_missing_rate.std() < 10 else "아니오")

    # 7. 권장사항
    print("\n7. 권장사항:")
    if total_missing/total_size > 0.2:
        print("   ⚠ 결측률 20% 이상: 보간보다는 결측값 처리 모델 고려")
    if (consecutive_missing >= 60).sum() > 100:
        print("   ⚠ 긴 결측 구간 다수: 선형 보간은 부정확할 수 있음")
    if high_missing_nodes > N * 0.1:
        print(f"   ⚠ 고결측 노드 {high_missing_nodes}개: 해당 노드 제거 고려")

    if total_missing/total_size < 0.1 and np.median(consecutive_missing) < 12:
        print("   ✓ 현재 보간 전략 적절함")

    print("="*60)


if __name__ == "__main__":
    analyze_missing_pattern()
