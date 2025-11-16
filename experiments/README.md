# Experiments

이 디렉토리는 실험용 Jupyter 노트북과 탐색적 데이터 분석(EDA) 파일을 저장하는 공간입니다.

## 구조

```
experiments/
├── notebooks/          # Jupyter notebooks
├── eda/               # Exploratory Data Analysis
├── visualizations/    # 시각화 결과
└── results/           # 실험 결과
```

## 사용 예시

### 1. 데이터 탐색

```python
# data_exploration.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../data/raw/loops035.csv')
# ... analysis code
```

### 2. 모델 실험

```python
# model_experiments.ipynb
from src.model_agcrn import AGCRN
from src.dataset import create_dataloaders

# ... experiment code
```

## 노트북 관리

- 노트북 이름은 날짜와 설명을 포함: `2025-01-15_traffic_pattern_analysis.ipynb`
- 중요한 실험 결과는 문서화
- 재현 가능한 코드 작성
