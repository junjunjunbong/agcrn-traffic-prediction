"""
결측값 패턴 분석 스크립트 (독립 버전)

결측값 보간이 정당한지 판단하기 위해:
1. 결측률 분석
2. 결측 패턴 분석 (랜덤 vs 체계적)
3. 보간 전후 통계 비교
"""
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_missing_pattern():
    """결측값 패턴 분석"""

    # 데이터 로드
    raw_data_dir = Path("data/raw")
    csv_files = list(raw_data_dir.glob("loops*.csv"))

    if not csv_files:
        print("No data files found in data/raw/")
        return

    print(f"Found {len(csv_files)} CSV file(s)")
    csv_path = csv_files[0]  # 첫 번째 파일 분석

    print(f"Analyzing: {csv_path.name}")
    print("Loading data...")
    df = pd.read_csv(csv_path)

    print(f"Loaded: {len(df):,} rows")

    # 간단한 텐서 변환
    print("Converting to tensor format...")
    features = ['flow', 'occupancy', 'harmonicMeanSpeed']

    # harmonicMeanSpeed에서 -1을 NaN으로 변환
    df.loc[df['harmonicMeanSpeed'] == -1, 'harmonicMeanSpeed'] = np.nan

    # raw_id 모드로 피벗
    node_col = 'raw_id'
    unique_nodes = sorted(df[node_col].unique())
    unique_times = sorted(df['begin'].unique())

    N = len(unique_nodes)
    T = len(unique_times)
    F = len(features)

    print(f"Tensor shape: T={T}, N={N}, F={F}")

    # 각 특징별로 피벗 테이블 생성
    X = np.zeros((T, N, F))

    for f_idx, feature in enumerate(features):
        pivot = df.pivot_table(
            values=feature,
            index='begin',
            columns=node_col,
            aggfunc='mean'
        )
        # 모든 시간과 노드로 reindex
        pivot = pivot.reindex(index=unique_times, columns=unique_nodes)
        X[:, :, f_idx] = pivot.values

    print("\n" + "="*60)
    print("결측값 패턴 분석")
    print("="*60)

    # 1. 전체 결측률
    total_missing = np.isnan(X).sum()
    total_size = X.size
    print(f"\n1. 전체 결측률: {total_missing:,} / {total_size:,} = {100*total_missing/total_size:.2f}%")

    # 2. 특징별 결측률
    print("\n2. 특징별 결측률:")
    for f_idx, feat_name in enumerate(features):
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
    print(f"   30% 이상 결측: {high_missing_nodes} / {N} 노드 ({100*high_missing_nodes/N:.1f}%)")

    # 4. 시간별 결측률
    print("\n4. 시간별 결측률:")
    time_missing_rate = np.isnan(X).mean(axis=(1, 2)) * 100
    print(f"   평균: {time_missing_rate.mean():.2f}%")
    print(f"   최소: {time_missing_rate.min():.2f}%")
    print(f"   최대: {time_missing_rate.max():.2f}%")

    high_missing_times = (time_missing_rate > 50).sum()
    print(f"   50% 이상 결측 시간: {high_missing_times} / {T} ({100*high_missing_times/T:.2f}%)")

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
        print(f"   평균 연속 결측: {consecutive_missing.mean():.1f} 타임스텝 ({consecutive_missing.mean()*5:.1f}초)")
        print(f"   최대 연속 결측: {consecutive_missing.max()} 타임스텝 ({consecutive_missing.max()*5/60:.1f}분)")
        print(f"   중앙값: {np.median(consecutive_missing):.1f} 타임스텝 ({np.median(consecutive_missing)*5:.1f}초)")

        # 긴 연속 결측 (5분 이상 = 60 타임스텝)
        long_gaps = (consecutive_missing >= 60).sum()
        print(f"   5분(60스텝) 이상 연속 결측: {long_gaps}회")

        # 1분 이상
        medium_gaps = (consecutive_missing >= 12).sum()
        print(f"   1분(12스텝) 이상 연속 결측: {medium_gaps}회")

    # 6. 보간 정당성 판단
    print("\n6. 보간 정당성 평가:")
    check1 = total_missing/total_size < 0.1
    check2 = np.median(consecutive_missing) < 12 if len(consecutive_missing) > 0 else True
    check3 = node_missing_rate.std() < 10

    print(f"   ✓ 전체 결측률 < 10%: {'예' if check1 else '아니오 ❌'}")
    print(f"   ✓ 대부분 짧은 결측 (< 1분): {'예' if check2 else '아니오 ❌'}")
    print(f"   ✓ 노드 균등 분포 (std < 10%): {'예' if check3 else '아니오 ❌'}")

    # 7. 권장사항
    print("\n7. 권장사항:")

    has_issues = False

    if total_missing/total_size > 0.2:
        print("   ⚠️  결측률 20% 이상: 보간보다는 결측값 처리 모델 고려")
        has_issues = True

    if len(consecutive_missing) > 0 and (consecutive_missing >= 60).sum() > 100:
        print("   ⚠️  긴 결측 구간 다수: 선형 보간은 부정확할 수 있음")
        has_issues = True

    if high_missing_nodes > N * 0.1:
        print(f"   ⚠️  고결측 노드 {high_missing_nodes}개: 해당 노드 제거 고려")
        has_issues = True

    if not has_issues and check1 and check2 and check3:
        print("   ✅ 현재 3단계 보간 전략 적절함!")
    elif check1 and check2:
        print("   ✅ 보간 가능하지만 일부 조정 필요")
    else:
        print("   ⚠️  보간 전략 재검토 필요")

    print("="*60)

    # 추가 통계
    print("\n8. 추가 통계:")
    print(f"   총 결측 구간 수: {len(consecutive_missing):,}")
    if len(consecutive_missing) > 0:
        print(f"   결측 구간 길이 분포:")
        print(f"     - 5초 이하: {(consecutive_missing == 1).sum():,}회")
        print(f"     - 6-30초: {((consecutive_missing > 1) & (consecutive_missing <= 6)).sum():,}회")
        print(f"     - 31초-1분: {((consecutive_missing > 6) & (consecutive_missing < 12)).sum():,}회")
        print(f"     - 1-5분: {((consecutive_missing >= 12) & (consecutive_missing < 60)).sum():,}회")
        print(f"     - 5분 이상: {(consecutive_missing >= 60).sum():,}회")


if __name__ == "__main__":
    analyze_missing_pattern()
