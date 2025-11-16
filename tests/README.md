# Tests

프로젝트의 테스트 코드를 포함합니다.

## 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/

# 특정 파일 테스트
pytest tests/test_model.py

# Coverage 포함
pytest tests/ --cov=src --cov-report=html
```

## 테스트 구조

```
tests/
├── test_preprocess.py    # 데이터 전처리 테스트
├── test_dataset.py       # Dataset 클래스 테스트
├── test_model.py         # 모델 테스트
├── test_trainer.py       # 학습 파이프라인 테스트
└── test_utils.py         # 유틸리티 함수 테스트
```

## 테스트 작성 가이드

### 1. 기본 구조

```python
import pytest
import numpy as np
from src.model_agcrn import AGCRN

def test_model_forward():
    """Test model forward pass"""
    model = AGCRN(num_nodes=10, input_dim=3)
    x = np.random.randn(2, 12, 10, 3)  # (batch, seq, nodes, features)
    output = model(x)
    assert output.shape == (2, 10, 1)
```

### 2. Fixture 사용

```python
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return np.random.randn(100, 10, 3)

def test_with_fixture(sample_data):
    assert sample_data.shape == (100, 10, 3)
```

## 요구 패키지

```bash
pip install pytest pytest-cov
```
