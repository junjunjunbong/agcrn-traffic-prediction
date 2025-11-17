# AGCRN 교통 예측 프로젝트 보안 취약점 및 개선사항 분석

**분석 날짜**: 2025-11-17
**프로젝트 버전**: 2.0.0
**분석 범위**: 보안 취약점, 코드 품질, 설정 관리

---

## 📋 목차

1. [요약](#요약)
2. [보안 취약점](#보안-취약점)
3. [코드 품질 문제](#코드-품질-문제)
4. [설정 및 환경 문제](#설정-및-환경-문제)
5. [우선순위별 개선 권장사항](#우선순위별-개선-권장사항)
6. [보안 체크리스트](#보안-체크리스트)

---

## 요약

### 발견된 주요 이슈

| 위험도 | 개수 | 주요 항목 |
|--------|------|-----------|
| **HIGH** | 2 | Pickle Deserialization, Unsafe Model Loading |
| **MEDIUM** | 4 | Path Traversal, 입력 검증, 의존성 취약점, 에러 핸들링 |
| **LOW** | 6 | 환경변수 관리, 로깅 일관성, 타입 체킹 등 |

### 즉시 조치 필요 항목

1. **Pickle deserialization 취약점** (dataset.py:158)
2. **Unsafe PyTorch model loading** (eval.py:145)
3. **Path traversal 방지** (여러 파일 I/O 부분)

---

## 보안 취약점

### 🔴 HIGH 위험도

#### 1. Pickle Deserialization 취약점

**위치**: `src/dataset.py:158`
**취약점 유형**: Arbitrary Code Execution (RCE)
**CVSS Score**: 9.8 (Critical)

**현재 코드**:
```python
data = np.load(npz_path, allow_pickle=True)
```

**문제점**:
- `allow_pickle=True`는 신뢰할 수 없는 소스의 파일을 역직렬화할 때 임의의 코드 실행 가능
- 공격자가 악의적인 npz 파일을 제공하면 시스템이 손상될 수 있음
- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)

**개선 방안**:
```python
# 방법 1: allow_pickle=False 사용 (권장)
data = np.load(npz_path, allow_pickle=False)

# 방법 2: 파일 무결성 검증
import hashlib

def verify_file_integrity(file_path: Path, expected_hash: str) -> None:
    """파일 해시를 검증하여 무결성 확인"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash != expected_hash:
        raise ValueError(f"File integrity check failed for {file_path}")

# 사용 예시
verify_file_integrity(npz_path, EXPECTED_HASHES.get(npz_path.name))
data = np.load(npz_path, allow_pickle=False)
```

**참고 자료**:
- [NumPy Security Advisory](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
- [Python Pickle Security Issues](https://docs.python.org/3/library/pickle.html#module-pickle)

---

#### 2. Unsafe PyTorch Model Loading

**위치**: `src/eval.py:145`
**취약점 유형**: Arbitrary Code Execution (RCE)
**CVSS Score**: 9.8 (Critical)

**현재 코드**:
```python
checkpoint = torch.load(model_path, map_location='cpu')
```

**문제점**:
- PyTorch 1.x의 `torch.load()`는 기본적으로 pickle을 사용하여 임의 코드 실행 가능
- 신뢰할 수 없는 모델 파일 로드 시 위험
- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/serialization.html#security)

**개선 방안**:
```python
# PyTorch 2.0 이상에서는 weights_only=True 사용 (권장)
try:
    checkpoint = torch.load(
        model_path,
        map_location='cpu',
        weights_only=True  # PyTorch 2.0+
    )
except TypeError:
    # PyTorch 1.x 호환성
    import warnings
    warnings.warn("PyTorch 1.x detected. Consider upgrading to 2.0+ for better security.")
    checkpoint = torch.load(model_path, map_location='cpu')

# 추가: 모델 파일 검증
def load_safe_checkpoint(model_path: Path, expected_keys: set) -> dict:
    """안전하게 체크포인트 로드"""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location='cpu',
        weights_only=True
    )

    # 예상되는 키만 있는지 검증
    unexpected_keys = set(checkpoint.keys()) - expected_keys
    if unexpected_keys:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected_keys}")

    return checkpoint

# 사용 예시
EXPECTED_KEYS = {'model_state_dict', 'optimizer_state_dict', 'epoch', 'loss'}
checkpoint = load_safe_checkpoint(model_path, EXPECTED_KEYS)
```

**참고 자료**:
- [PyTorch Save/Load Security](https://pytorch.org/docs/stable/notes/serialization.html#security)

---

### 🟡 MEDIUM 위험도

#### 3. Path Traversal 취약점

**위치**: 여러 파일
**취약점 유형**: Path Traversal / Directory Traversal
**CVSS Score**: 7.5 (High)

**영향받는 코드**:

1. **src/preprocess.py:111** - CSV 파일 읽기
```python
df = pd.read_csv(csv_path)  # 경로 검증 없음
```

2. **src/preprocess.py:456-471** - NPZ 파일 저장
```python
np.savez(output_path, ...)  # 경로 검증 없음
sensors_df.to_csv(sensors_path, index=False)
```

3. **src/trainer.py:190-196** - 모델 저장
```python
torch.save({...}, save_path)  # 경로 검증 없음
```

4. **src/trainer.py:216-217** - History JSON 저장
```python
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
```

5. **src/eval.py:125** - 그래프 저장
```python
plt.savefig(save_path)  # 경로 검증 없음
```

**문제점**:
- 사용자가 `../../etc/passwd` 같은 경로를 입력하면 시스템 파일 접근 가능
- 허가되지 않은 디렉토리에 파일 쓰기 가능
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)

**개선 방안**:
```python
from pathlib import Path
from typing import List

def validate_file_path(
    file_path: Path,
    allowed_dirs: List[Path],
    must_exist: bool = False
) -> Path:
    """
    파일 경로가 허용된 디렉토리 내에 있는지 검증

    Args:
        file_path: 검증할 파일 경로
        allowed_dirs: 허용된 디렉토리 목록
        must_exist: True면 파일이 존재해야 함

    Returns:
        검증된 절대 경로

    Raises:
        ValueError: 경로가 허용되지 않은 경우
        FileNotFoundError: must_exist=True이고 파일이 없는 경우
    """
    file_path = Path(file_path).resolve()

    # 허용된 디렉토리 중 하나에 속하는지 확인
    for allowed_dir in allowed_dirs:
        allowed_dir = Path(allowed_dir).resolve()
        try:
            file_path.relative_to(allowed_dir)
            # 파일 존재 여부 확인
            if must_exist and not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return file_path
        except ValueError:
            continue

    raise ValueError(
        f"File path '{file_path}' is not in allowed directories: {allowed_dirs}"
    )

# config.py에 허용된 디렉토리 정의
ALLOWED_DATA_DIRS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "logs"
]

# 사용 예시
from src.config import ALLOWED_DATA_DIRS

# 읽기
safe_csv_path = validate_file_path(csv_path, [RAW_DATA_DIR], must_exist=True)
df = pd.read_csv(safe_csv_path)

# 쓰기
safe_output_path = validate_file_path(output_path, ALLOWED_DATA_DIRS)
safe_output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(safe_output_path, ...)
```

---

#### 4. 입력 검증 부재

**위치**: 여러 파일
**취약점 유형**: Input Validation, DoS
**CVSS Score**: 6.5 (Medium)

##### 4-1. 파일 크기 제한 없음

**위치**: `src/preprocess.py:111`

**현재 코드**:
```python
df = pd.read_csv(csv_path)
# 파일 크기 제한 없음 - 메모리 부족 DoS 가능
```

**개선 방안**:
```python
import os
from pathlib import Path

def load_csv_data(
    csv_path: Path,
    max_size_mb: int = 500,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    CSV 파일을 안전하게 로드 (크기 검증 포함)

    Args:
        csv_path: CSV 파일 경로
        max_size_mb: 최대 파일 크기 (MB)
        chunk_size: 청크 단위 읽기 크기

    Returns:
        로드된 DataFrame

    Raises:
        FileNotFoundError: 파일이 없는 경우
        ValueError: 파일이 너무 큰 경우
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 파일 크기 확인
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        )

    logger.info(f"Loading CSV file ({file_size_mb:.2f}MB): {csv_path}")

    # 대용량 파일은 청크로 읽기
    if file_size_mb > 100:
        logger.info(f"Large file detected, reading in chunks of {chunk_size}")
        chunks = pd.read_csv(csv_path, chunksize=chunk_size)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df):,} rows")
    return df
```

##### 4-2. 명령행 인자 검증 없음

**위치**: `train.py:21-33`

**현재 코드**:
```python
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--lr', type=float, default=LEARNING_RATE)
parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
parser.add_argument('--imputed_weight', type=float, default=0.1)
# 범위 검증 없음
```

**개선 방안**:
```python
def validate_args(args) -> argparse.Namespace:
    """
    명령행 인자 검증

    Args:
        args: argparse로 파싱된 인자

    Returns:
        검증된 인자

    Raises:
        ValueError: 인자가 유효하지 않은 경우
    """
    errors = []

    # Batch size 검증
    if not (1 <= args.batch_size <= 1024):
        errors.append(f"Invalid batch_size: {args.batch_size} (must be 1-1024)")

    # Learning rate 검증
    if not (0 < args.lr <= 1.0):
        errors.append(f"Invalid learning rate: {args.lr} (must be 0-1.0)")

    # Epochs 검증
    if not (1 <= args.epochs <= 10000):
        errors.append(f"Invalid epochs: {args.epochs} (must be 1-10000)")

    # Imputed weight 검증
    if not (0.0 <= args.imputed_weight <= 1.0):
        errors.append(f"Invalid imputed_weight: {args.imputed_weight} (must be 0.0-1.0)")

    # Loss function 검증
    valid_losses = ['mse', 'masked_mse', 'masked_mae', 'observed_only']
    if args.loss not in valid_losses:
        errors.append(f"Invalid loss: {args.loss} (must be one of {valid_losses})")

    # Device 검증
    if args.device not in ['cuda', 'cpu']:
        errors.append(f"Invalid device: {args.device} (must be 'cuda' or 'cpu')")

    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return args

# train.py에서 사용
args = parser.parse_args()
args = validate_args(args)
logger.info(f"Arguments validated: {vars(args)}")
```

---

#### 5. 의존성 버전 취약점

**위치**: `requirements.txt`
**취약점 유형**: Dependency Vulnerabilities
**CVSS Score**: 6.0 (Medium)

**현재 상태**:
```
torch>=1.9.0,<2.0.0  # 너무 넓은 범위 - 알려진 취약점 포함 가능
numpy>=1.21.0,<2.0.0
pandas>=1.3.0,<2.0.0
pyyaml>=5.4.0  # 상한선 없음 - 호환성 문제 가능
```

**문제점**:
- 버전 범위가 너무 넓어 알려진 보안 취약점이 있는 버전 설치 가능
- PyYAML 5.4.0은 CVE-2020-1747 등 취약점 존재
- 상한선이 없으면 호환성 문제 발생 가능

**개선 방안**:
```
# requirements.txt 개선 버전
# Deep Learning
torch==1.13.1  # 또는 최신 안정 버전 (보안 패치 적용)
torchvision==0.14.1

# Data Processing
numpy==1.24.3  # CVE 해결된 버전
pandas==1.5.3

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Progress & Utilities
tqdm==4.65.0
pyyaml>=6.0,<7.0  # CVE-2020-1747, CVE-2020-14343 해결

# Testing
pytest==7.3.1
pytest-cov==4.1.0

# Code Quality
black==23.3.0
flake8==6.0.0
isort==5.12.0

# Logging & Monitoring
tensorboard==2.13.0

# Security
safety>=2.3.0  # 의존성 취약점 스캔
bandit>=1.7.0  # 코드 보안 스캔
```

**보안 스캔 설정**:
```bash
# 의존성 취약점 스캔
pip install safety
safety check

# 코드 보안 스캔
pip install bandit
bandit -r src/ -f txt -o bandit_report.txt

# GitHub Actions에 보안 스캔 추가
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Safety
        run: |
          pip install safety
          safety check --json
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r src/
```

**참고 자료**:
- [PyYAML CVE-2020-1747](https://nvd.nist.gov/vuln/detail/CVE-2020-1747)
- [Safety - Python Dependency Checker](https://pyup.io/safety/)

---

#### 6. 에러 핸들링 부재

**위치**: `src/trainer.py`
**취약점 유형**: Error Handling
**CVSS Score**: 5.0 (Medium)

**문제점**:
- 전체 train() 메서드에 try-except 없음
- GPU OOM 에러 처리 없음
- NaN loss 발생 시 처리 없음
- 모델 저장 실패 시 처리 없음

**현재 코드**:
```python
def train_epoch(self) -> float:
    self.model.train()
    # GPU OOM, NaN loss 등 예외 처리 없음
    for batch_data in pbar:
        x, y, masks = batch_data
        x = x.to(self.device)  # CUDA OOM 가능
        ...
```

**개선 방안**:
```python
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class Trainer:
    def train_epoch(self) -> float:
        """한 에폭 학습 (에러 핸들링 포함)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        try:
            pbar = tqdm(self.train_loader, desc="Training")
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    x, y, masks = batch_data
                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(x)

                    # Loss 계산
                    loss = self._compute_loss(output, y, masks)

                    # NaN/Inf 체크
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(
                            f"Invalid loss at batch {batch_idx}: {loss.item()}, skipping"
                        )
                        continue

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=5.0
                    )

                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': loss.item()})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(
                            f"GPU OOM at batch {batch_idx}. "
                            "Consider reducing batch size."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Training epoch failed: {str(e)}")
            raise

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(
        self,
        num_epochs: int,
        save_path: Optional[Path] = None
    ) -> Dict[str, list]:
        """메인 학습 루프 (에러 핸들링 포함)"""
        history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

        try:
            logger.info(f"Starting training for {num_epochs} epochs")
            logger.info(f"Device: {self.device}")

            for epoch in range(num_epochs):
                try:
                    train_loss = self.train_epoch()
                    val_loss = self.validate()

                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)

                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
                    )

                    # 모델 저장 (에러 핸들링 포함)
                    if val_loss < self.best_val_loss and save_path:
                        self.best_val_loss = val_loss
                        try:
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                            }, save_path)
                            logger.info(f"✓ Saved best model: {save_path}")
                        except IOError as e:
                            logger.error(f"Failed to save model: {e}")
                            # 계속 학습

                except KeyboardInterrupt:
                    logger.info("Training interrupted by user at epoch {epoch+1}")
                    break

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # History 저장 시도 (실패해도 에러 발생 안 함)
            if save_path:
                history_path = save_path.parent / "training_history.json"
                try:
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=2)
                    logger.info(f"✓ Saved training history: {history_path}")
                except IOError as e:
                    logger.warning(f"Failed to save history: {e}")

        return history
```

---

### 🔵 LOW 위험도

#### 7. 환경변수 관리 부재

**위치**: `src/config.py`
**취약점 유형**: Configuration Management
**CVSS Score**: 3.0 (Low)

**문제점**:
- 모든 설정이 하드코딩됨 - 환경별 설정 불가
- 민감한 정보 관리 방법 없음
- 개발/프로덕션 환경 구분 없음

**개선 방안**:

**1. `.env` 파일 지원 추가**:

```bash
# .env.example (템플릿)
# Environment
AGCRN_ENV=development  # development, production, testing

# Device
AGCRN_DEVICE=cuda

# Training
AGCRN_BATCH_SIZE=64
AGCRN_LEARNING_RATE=0.001
AGCRN_NUM_EPOCHS=100

# Data
AGCRN_DATA_DIR=./data
AGCRN_NODE_MODE=raw_id

# Logging
AGCRN_LOG_LEVEL=INFO
AGCRN_LOG_FILE=logs/agcrn.log

# Optional: API Keys (if needed)
# MODEL_API_KEY=your_api_key_here
```

**2. config.py 수정**:

```python
# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

# .env 파일 로드
load_dotenv()

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv('AGCRN_DATA_DIR', PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
META_DATA_DIR = DATA_DIR / "meta"

# Environment
ENVIRONMENT = os.getenv('AGCRN_ENV', 'development')

# Device
DEVICE = os.getenv('AGCRN_DEVICE',
                   "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = int(os.getenv('AGCRN_BATCH_SIZE', '64'))
LEARNING_RATE = float(os.getenv('AGCRN_LEARNING_RATE', '0.001'))
NUM_EPOCHS = int(os.getenv('AGCRN_NUM_EPOCHS', '100'))

# Data
NODE_MODE = os.getenv('AGCRN_NODE_MODE', 'raw_id')

# Logging
LOG_LEVEL = os.getenv('AGCRN_LOG_LEVEL', 'INFO')
LOG_FILE = Path(os.getenv('AGCRN_LOG_FILE', PROJECT_ROOT / 'logs' / 'agcrn.log'))

# Validation
if ENVIRONMENT not in ['development', 'production', 'testing']:
    raise ValueError(f"Invalid environment: {ENVIRONMENT}")

if BATCH_SIZE < 1 or BATCH_SIZE > 1024:
    raise ValueError(f"Invalid batch size: {BATCH_SIZE}")

logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Device: {DEVICE}")
```

**3. requirements.txt에 추가**:
```
python-dotenv>=1.0.0
```

---

#### 8. 하드코딩된 정보

**위치**: `setup.py:28-30`
**취약점 유형**: Information Disclosure
**CVSS Score**: 2.0 (Low)

**현재 코드**:
```python
author="Your Name",
author_email="your.email@example.com",
url="https://github.com/your-username/agcrn-traffic-prediction",
```

**개선 방안**:
```python
# setup.py
import os

setup(
    name="agcrn-traffic-prediction",
    version="2.0.0",
    author=os.getenv('PACKAGE_AUTHOR', 'AGCRN Team'),
    author_email=os.getenv('PACKAGE_EMAIL', 'contact@example.com'),
    url=os.getenv('PACKAGE_URL', 'https://github.com/junjunjunbong/agcrn-traffic-prediction'),
    ...
)
```

---

#### 9. .gitignore 누락

**위치**: `.gitignore`
**위험도**: LOW

**문제점**:
- `.env` 파일이 gitignore에 없음
- 민감한 설정 파일 패턴 누락
- 보안 스캔 결과 파일 누락

**개선 방안**:

```gitignore
# .gitignore에 추가

# Environment & Secrets
.env
.env.*
!.env.example
*.key
*.pem
*.pfx
credentials.json
secrets.yaml
config.local.yaml

# Security scans
.safety/
bandit_report.txt
bandit_report.json
safety_report.json

# Temporary files
*.tmp
*.temp
.DS_Store

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs (민감한 정보 포함 가능)
logs/*.log
!logs/.gitkeep
```

---

## 코드 품질 문제

### 1. 에러 핸들링 일관성 부족

**영향받는 파일**:
- `src/trainer.py` - try-except 거의 없음
- `src/model_agcrn.py` - 입력 검증 없음
- `src/losses.py` - 일부만 검증

**세부 사항은 위 "6. 에러 핸들링 부재" 참고**

---

### 2. 입력 검증 부족

#### 모델 입력 검증

**위치**: `src/model_agcrn.py:119-146`

**현재 코드**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    sequence_length = x.shape[1]
    # 입력 shape 검증 없음 - 잘못된 입력 시 cryptic error
```

**개선 방안**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass with input validation

    Args:
        x: Input tensor (batch, sequence_length, num_nodes, input_dim)

    Returns:
        Output tensor (batch, num_nodes, output_dim)

    Raises:
        ValueError: 입력이 유효하지 않은 경우
    """
    # Shape 검증
    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D input (batch, seq, nodes, features), got {x.ndim}D"
        )

    batch_size, sequence_length, num_nodes, input_dim = x.shape

    # 노드 수 검증
    if num_nodes != self.num_nodes:
        raise ValueError(
            f"Expected {self.num_nodes} nodes, got {num_nodes}"
        )

    # 특성 수 검증
    if input_dim != self.input_dim:
        raise ValueError(
            f"Expected {self.input_dim} features, got {input_dim}"
        )

    # NaN/Inf 검증
    if torch.isnan(x).any():
        raise ValueError("Input contains NaN values")

    if torch.isinf(x).any():
        raise ValueError("Input contains Inf values")

    # ... 나머지 forward 로직 ...
```

---

#### Loss 함수 검증

**위치**: `src/losses.py`

**개선 방안**:
```python
class MaskedMSELoss(nn.Module):
    """MSE loss with mask support and validation"""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked MSE loss with input validation

        Args:
            pred: Predictions
            target: Ground truth
            mask: Optional mask (True = observed, False = imputed)

        Returns:
            Scalar loss

        Raises:
            ValueError: 입력 shape 불일치
        """
        # Shape 검증
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        if mask is not None and mask.shape != pred.shape:
            raise ValueError(
                f"Mask shape {mask.shape} must match pred shape {pred.shape}"
            )

        # ... 나머지 로직 ...
```

---

### 3. 로깅 일관성 부족

**현재 상태**:
- `src/preprocess.py`: logging 모듈 사용 ✓
- `src/trainer.py`: print만 사용
- `src/eval.py`: print만 사용
- 일관성 없음

**개선 방안**:

**1. 모든 파일에 logging 적용**:

```python
# src/trainer.py
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, ...):
        # print 대신 logger 사용
        logger.info(f"Initializing trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> float:
        logger.debug("Starting training epoch")
        # ...

    def train(self, num_epochs: int, ...) -> Dict:
        logger.info(f"Starting training for {num_epochs} epochs")
        # ...
        logger.info(f"✓ Best validation loss: {self.best_val_loss:.6f}")
```

**2. 중앙화된 로깅 설정**:

```python
# src/utils/logger.py
import logging
from pathlib import Path

def setup_logging(log_file: Path, level: str = "INFO"):
    """중앙화된 로깅 설정"""

    # 로그 디렉토리 생성
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 로그 레벨
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # 포매터
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

# train.py에서 사용
from src.utils.logger import setup_logging
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logging(LOG_FILE, LOG_LEVEL)
```

---

### 4. 타입 체킹 일관성 부족

**현재 상태**:
- `src/preprocess.py`: 완전한 타입 힌트 ✓
- `src/dataset.py`: 완전한 타입 힌트 ✓
- `src/trainer.py`: 부분적
- `src/losses.py`: 부분적
- `src/model_agcrn.py`: 부분적
- `src/eval.py`: 부분적

**개선 방안**:

**1. 모든 함수에 타입 힌트 추가**:

```python
# src/trainer.py
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ) -> None:
        ...

    def train_epoch(self) -> float:
        ...

    def validate(self) -> float:
        ...

    def train(
        self,
        num_epochs: int,
        save_path: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        ...
```

**2. mypy 설정 및 CI 통합**:

```ini
# mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
no_implicit_optional = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True
```

```bash
# 타입 체크 실행
mypy src/ --strict

# CI에 추가 (.github/workflows/ci.yml)
- name: Type Check
  run: mypy src/
```

---

### 5. 리소스 누수 위험

**현재 상태**: 대체로 양호 (with 문 사용)

**개선 필요 영역**:

```python
# src/trainer.py:216-217
# 현재 (Good - with 문 사용)
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

# 개선 (에러 핸들링 추가)
try:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
except IOError as e:
    logger.error(f"Failed to save training history: {e}")
    # 계속 진행 (critical하지 않음)
```

---

### 6. 하드코딩된 값들

#### Magic Numbers

**위치**: `src/trainer.py:99`

**현재 코드**:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
# 5.0이 하드코딩됨
```

**개선 방안**:
```python
# config.py에 추가
GRADIENT_CLIP_NORM = 5.0
GRADIENT_CLIP_TYPE = 'norm'  # 'norm' or 'value'

# trainer.py
from src.config import GRADIENT_CLIP_NORM

if GRADIENT_CLIP_TYPE == 'norm':
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        max_norm=GRADIENT_CLIP_NORM
    )
```

#### 분산된 설정값

**개선 방안**:
```python
# config.py - 모든 하이퍼파라미터 집중 관리

# === Training ===
BATCH_SIZE = int(os.getenv('AGCRN_BATCH_SIZE', '64'))
LEARNING_RATE = float(os.getenv('AGCRN_LEARNING_RATE', '0.001'))
NUM_EPOCHS = int(os.getenv('AGCRN_NUM_EPOCHS', '100'))
GRADIENT_CLIP_NORM = float(os.getenv('AGCRN_GRAD_CLIP', '5.0'))

# === DataLoader ===
NUM_WORKERS = int(os.getenv('AGCRN_NUM_WORKERS', '0'))
PIN_MEMORY = os.getenv('AGCRN_PIN_MEMORY', 'True').lower() == 'true'
PREFETCH_FACTOR = int(os.getenv('AGCRN_PREFETCH_FACTOR', '2'))

# === Dataset ===
MAX_MISSING_GAP = int(os.getenv('AGCRN_MAX_MISSING_GAP', '60'))
STRIDE = int(os.getenv('AGCRN_STRIDE', '1'))
FILTER_LONG_GAPS = os.getenv('AGCRN_FILTER_GAPS', 'True').lower() == 'true'

# === Evaluation ===
NUM_NODES_TO_PLOT = int(os.getenv('AGCRN_PLOT_NODES', '5'))
NUM_SAMPLES_TO_PLOT = int(os.getenv('AGCRN_PLOT_SAMPLES', '100'))
PLOT_DPI = int(os.getenv('AGCRN_PLOT_DPI', '100'))

# === Model ===
HIDDEN_DIM = int(os.getenv('AGCRN_HIDDEN_DIM', '64'))
NUM_LAYERS = int(os.getenv('AGCRN_NUM_LAYERS', '2'))
CHEB_K = int(os.getenv('AGCRN_CHEB_K', '2'))
EMBED_DIM = int(os.getenv('AGCRN_EMBED_DIM', '10'))
```

---

### 7. 기타 코드 품질 이슈

#### eval.py의 마스크 미처리

**위치**: `src/eval.py:40`

**현재 코드**:
```python
def evaluate_model(model, test_loader, ...):
    for x, y in test_loader:  # 마스크를 받지 않음
        # dataset.py는 (x, y, masks) 반환하는데 여기서는 무시
```

**개선 방안**:
```python
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str = DEVICE,
    use_masks: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    모델 평가 (마스크 지원)

    Args:
        model: 평가할 모델
        test_loader: 테스트 DataLoader
        criterion: 손실 함수
        device: 디바이스
        use_masks: 마스크 사용 여부

    Returns:
        (평균 손실, 예측값, 실제값)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 마스크 처리
            if use_masks and len(batch_data) == 3:
                x, y, masks = batch_data
                mask_target = masks[1][:, -1, :, :] if masks else None
            else:
                x, y = batch_data[0], batch_data[1]
                mask_target = None

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            y_target = y[:, -1, :, :]

            # 손실 계산 (마스크 지원 여부에 따라)
            if mask_target is not None and hasattr(criterion, 'forward'):
                params = criterion.forward.__code__.co_varnames
                if 'mask' in params:
                    loss = criterion(output, y_target, mask_target.to(device))
                else:
                    loss = criterion(output, y_target)
            else:
                loss = criterion(output, y_target)

            total_loss += loss.item()
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return avg_loss, predictions, targets
```

---

#### 복잡한 동적 검사 로직

**위치**: `src/trainer.py:82-95`

**현재 코드**:
```python
# 복잡하고 읽기 어려운 동적 검사
if hasattr(self.criterion, 'forward') and \
   'mask' in self.criterion.forward.__code__.co_varnames:
    loss = self.criterion(output, y_target, mask_target)
else:
    loss = self.criterion(output, y_target)
```

**개선 방안 1: ABC 사용**:
```python
from abc import ABC, abstractmethod

class Loss(ABC):
    """기본 손실 함수 인터페이스"""

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

class MaskedLoss(ABC):
    """마스크 지원 손실 함수 인터페이스"""

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

# Trainer에서
class Trainer:
    def __init__(self, ..., criterion_uses_mask: bool = False):
        self.criterion_uses_mask = criterion_uses_mask

    def train_epoch(self):
        if self.criterion_uses_mask:
            loss = self.criterion(output, y_target, mask_target)
        else:
            loss = self.criterion(output, y_target)
```

**개선 방안 2: Duck typing with try-except**:
```python
def compute_loss(
    self,
    output: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """손실 계산 (마스크 자동 감지)"""
    try:
        # 마스크 지원 시도
        return self.criterion(output, target, mask)
    except TypeError:
        # 마스크 미지원 시 fallback
        return self.criterion(output, target)
```

---

## 설정 및 환경 문제

### 1. 환경변수 관리

위의 "7. 환경변수 관리 부재" 참고

---

### 2. 설정 파일 보안

**현재 상태**: 모든 설정이 코드에 하드코딩
**위험도**: LOW

**개선 방안**:

**1. YAML 기반 설정 파일**:

```yaml
# configs/default.yaml
environment: development

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 100
  gradient_clip_norm: 5.0
  device: cuda

model:
  hidden_dim: 64
  num_layers: 2
  cheb_k: 2
  embed_dim: 10
  output_dim: 1

data:
  node_mode: raw_id
  features:
    - flow
    - occupancy
    - harmonicMeanSpeed
  sequence_length: 12
  horizon: 3

dataset:
  max_missing_gap: 60
  stride: 1
  filter_long_gaps: true

logging:
  level: INFO
  file: logs/agcrn.log
```

**2. 설정 로더**:

```python
# src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 없는 경우
        yaml.YAMLError: YAML 파싱 실패
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)  # safe_load 사용!

        logger.info(f"Loaded config from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise

def merge_configs(
    default_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    설정 딕셔너리 병합 (override가 우선)

    Args:
        default_config: 기본 설정
        override_config: 오버라이드 설정

    Returns:
        병합된 설정
    """
    merged = default_config.copy()

    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged

# 사용 예시
def get_config(config_name: str = 'default') -> Dict[str, Any]:
    """설정 로드 (환경별 오버라이드 지원)"""
    config_dir = Path(__file__).parent.parent.parent / 'configs'

    # 기본 설정 로드
    default_config = load_config(config_dir / 'default.yaml')

    # 환경별 설정 오버라이드
    env_config_path = config_dir / f'{config_name}.yaml'
    if env_config_path.exists():
        env_config = load_config(env_config_path)
        config = merge_configs(default_config, env_config)
    else:
        config = default_config

    return config
```

**3. train.py에서 사용**:

```python
from src.utils.config_loader import get_config

# 설정 로드
config = get_config(args.config)  # --config development/production

# 설정 사용
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
```

---

## 우선순위별 개선 권장사항

### 🔴 즉시 수정 필요 (HIGH Priority)

#### 1. Pickle Deserialization 취약점 수정
**파일**: `src/dataset.py:158`
**예상 작업 시간**: 5분
**난이도**: 쉬움

```python
# 변경 전
data = np.load(npz_path, allow_pickle=True)

# 변경 후
data = np.load(npz_path, allow_pickle=False)
```

#### 2. Unsafe PyTorch Model Loading 수정
**파일**: `src/eval.py:145`
**예상 작업 시간**: 10분
**난이도**: 쉬움

```python
# 변경 전
checkpoint = torch.load(model_path, map_location='cpu')

# 변경 후
try:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
except TypeError:
    # PyTorch 1.x 호환성
    import warnings
    warnings.warn("Using unsafe torch.load - upgrade to PyTorch 2.0+")
    checkpoint = torch.load(model_path, map_location='cpu')
```

#### 3. Path Traversal 방지
**파일**: 여러 파일
**예상 작업 시간**: 2시간
**난이도**: 중간

- `validate_file_path()` 함수 구현
- 모든 파일 I/O에 검증 추가
- 테스트 케이스 작성

---

### 🟡 단기 개선 (MEDIUM Priority, 1-2주)

#### 4. 입력 검증 추가
**예상 작업 시간**: 4시간
**난이도**: 중간

- [ ] 파일 크기 검증 (`load_csv_data()`)
- [ ] 명령행 인자 검증 (`validate_args()`)
- [ ] 모델 입력 검증 (shape, NaN/Inf 체크)
- [ ] Loss 함수 입력 검증

#### 5. 에러 핸들링 개선
**예상 작업 시간**: 6시간
**난이도**: 중간

- [ ] Trainer에 try-except 추가
- [ ] GPU OOM 처리
- [ ] NaN loss 처리
- [ ] 모델 저장 실패 처리
- [ ] KeyboardInterrupt 처리

#### 6. 의존성 버전 고정 및 보안 스캔
**예상 작업 시간**: 2시간
**난이도**: 쉬움

- [ ] requirements.txt 버전 고정
- [ ] safety 설치 및 스캔
- [ ] bandit 설치 및 스캔
- [ ] GitHub Actions 보안 워크플로우 추가

---

### 🔵 장기 개선 (LOW Priority, 1-2개월)

#### 7. 로깅 통일
**예상 작업 시간**: 4시간
**난이도**: 쉬움

- [ ] 모든 파일에 logging 모듈 적용
- [ ] print 문 제거
- [ ] 중앙화된 로깅 설정 (`setup_logging()`)
- [ ] 로그 레벨 환경변수로 제어

#### 8. 타입 힌트 완성
**예상 작업 시간**: 6시간
**난이도**: 중간

- [ ] 모든 함수에 타입 힌트 추가
- [ ] mypy 설정 파일 작성
- [ ] CI에 타입 체크 추가
- [ ] 타입 에러 수정

#### 9. 환경변수 지원
**예상 작업 시간**: 3시간
**난이도**: 쉬움

- [ ] python-dotenv 설치
- [ ] .env.example 작성
- [ ] config.py에 환경변수 지원 추가
- [ ] .gitignore에 .env 추가

#### 10. 설정 파일 외부화
**예상 작업 시간**: 4시간
**난이도**: 중간

- [ ] YAML 설정 파일 작성
- [ ] config_loader.py 구현
- [ ] 환경별 설정 오버라이드
- [ ] train.py에 --config 옵션 추가

---

## 보안 체크리스트

### 즉시 조치 필요 (HIGH)

- [ ] **Pickle deserialization 취약점 수정**
  - [ ] src/dataset.py:158 - `allow_pickle=False`
  - [ ] 데이터 파일 재생성 (.npz 파일이 pickle 객체 포함 시)

- [ ] **PyTorch model loading 보안**
  - [ ] src/eval.py:145 - `weights_only=True` 추가
  - [ ] src/trainer.py 모델 로드 부분도 확인

- [ ] **Path traversal 방지**
  - [ ] `validate_file_path()` 함수 구현
  - [ ] src/preprocess.py 파일 I/O 검증 추가
  - [ ] src/trainer.py 파일 I/O 검증 추가
  - [ ] src/eval.py 파일 I/O 검증 추가

### 단기 개선 (MEDIUM)

- [ ] **입력 검증**
  - [ ] 파일 크기 제한 (`load_csv_data()`)
  - [ ] 명령행 인자 범위 검증 (`validate_args()`)
  - [ ] 모델 입력 검증 (shape, NaN/Inf)

- [ ] **에러 핸들링**
  - [ ] Trainer.train_epoch() try-except 추가
  - [ ] GPU OOM 처리
  - [ ] NaN loss 처리
  - [ ] 모델 저장 실패 처리

- [ ] **의존성 보안**
  - [ ] requirements.txt 버전 고정
  - [ ] `pip install safety && safety check` 실행
  - [ ] `pip install bandit && bandit -r src/` 실행
  - [ ] GitHub Actions에 보안 스캔 추가

### 장기 개선 (LOW)

- [ ] **로깅**
  - [ ] 모든 print를 logger로 변경
  - [ ] setup_logging() 구현
  - [ ] 로그 레벨 환경변수로 제어

- [ ] **타입 체킹**
  - [ ] 모든 함수에 타입 힌트 추가
  - [ ] mypy.ini 작성
  - [ ] CI에 mypy 추가

- [ ] **환경변수**
  - [ ] .env.example 작성
  - [ ] config.py에 os.getenv() 추가
  - [ ] .gitignore에 .env 추가

- [ ] **설정 외부화**
  - [ ] configs/default.yaml 작성
  - [ ] config_loader.py 구현
  - [ ] train.py에 --config 옵션 추가

### 보안 모니터링

- [ ] **정기적인 보안 스캔**
  - [ ] 매주 `safety check` 실행
  - [ ] 매월 의존성 업데이트 검토
  - [ ] GitHub Dependabot 활성화

- [ ] **코드 리뷰**
  - [ ] PR에 보안 체크리스트 추가
  - [ ] 파일 I/O 코드 집중 리뷰
  - [ ] 민감한 정보 하드코딩 방지

---

## 참고 자료

### 보안 가이드라인

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/archive/2023/2023_top25_list.html)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

### 도구

- [Safety](https://pyup.io/safety/) - Python dependency vulnerability scanner
- [Bandit](https://bandit.readthedocs.io/) - Python code security scanner
- [mypy](https://mypy.readthedocs.io/) - Static type checker
- [pre-commit](https://pre-commit.com/) - Git hook framework

### 취약점 데이터베이스

- [NVD (National Vulnerability Database)](https://nvd.nist.gov/)
- [CVE Details](https://www.cvedetails.com/)
- [GitHub Advisory Database](https://github.com/advisories)

---

## 요약

### 주요 발견 사항

1. **2개의 HIGH 위험 취약점** (즉시 수정 필요)
   - Pickle deserialization (RCE 가능)
   - Unsafe model loading (RCE 가능)

2. **4개의 MEDIUM 위험 이슈** (단기 개선)
   - Path traversal
   - 입력 검증 부재
   - 의존성 취약점
   - 에러 핸들링 부족

3. **6개의 LOW 위험 이슈** (장기 개선)
   - 환경변수 관리
   - 로깅 일관성
   - 타입 체킹
   - 설정 외부화 등

### 권장 조치 순서

1. **1일차**: HIGH 취약점 즉시 수정 (Pickle, Model loading)
2. **1주차**: Path traversal 방지 + 입력 검증
3. **2주차**: 에러 핸들링 + 의존성 고정
4. **1개월차**: 로깅/타입 체킹/환경변수 지원
5. **지속적**: 보안 스캔 자동화 + 모니터링

### 예상 개선 효과

- **보안**: RCE 취약점 제거, 입력 검증으로 DoS 방지
- **안정성**: 에러 핸들링으로 크래시 감소
- **유지보수성**: 로깅/타입 체킹으로 디버깅 용이
- **배포**: 환경변수로 다양한 환경 지원

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-11-17
**작성자**: Claude Code Security Review
