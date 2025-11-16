# GitHub 업로드 가이드

## 문제 해결

현재 Git 저장소가 잘못된 위치에 초기화되어 있습니다. 다음 단계를 따라 프로젝트 디렉토리에서 Git을 올바르게 설정하세요.

## 단계별 가이드

### 1. 프로젝트 디렉토리로 이동

터미널/PowerShell에서 프로젝트 디렉토리로 이동:

```powershell
cd "i:\내 드라이브\0. 수업\25-2 학기\딥러닝의 기초\DL_project"
```

### 2. 기존 Git 저장소 제거 (있다면)

```powershell
# 프로젝트 디렉토리에 .git 폴더가 있는지 확인
if (Test-Path ".git") {
    Remove-Item -Recurse -Force .git
}
```

### 3. Git 저장소 초기화

```powershell
git init
```

### 4. 프로젝트 파일만 추가

```powershell
# 소스 코드 및 설정 파일 추가
git add README.md LICENSE requirements.txt .gitignore
git add preprocess.py train.py
git add src/
git add configs/

# .gitignore가 제대로 작동하는지 확인
git status
```

### 5. 첫 커밋 생성

```powershell
git commit -m "Initial commit: AGCRN traffic prediction project"
```

### 6. GitHub 저장소 생성 및 연결

1. GitHub에서 새 저장소 생성 (https://github.com/new)
   - 저장소 이름: 예) `agcrn-traffic-prediction`
   - Public 또는 Private 선택
   - **README, .gitignore, LICENSE는 추가하지 마세요** (이미 있으므로)

2. 원격 저장소 연결 및 푸시

```powershell
# 원격 저장소 추가 (YOUR_USERNAME을 본인의 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/agcrn-traffic-prediction.git

# 기본 브랜치 이름을 main으로 변경 (필요시)
git branch -M main

# 코드 푸시
git push -u origin main
```

## 추가 확인사항

### .gitignore 확인

다음 파일/폴더는 자동으로 제외됩니다:
- `data/processed/` - 전처리된 데이터
- `data/meta/` - 메타데이터
- `saved_models/` - 학습된 모델
- `logs/` - 로그 파일
- `__pycache__/` - Python 캐시
- `*.csv` - 원본 데이터 파일 (큰 파일이므로 제외 권장)

### 원본 데이터 파일 처리

큰 CSV 파일(`data/*.csv`)은 GitHub에 올리지 않는 것이 좋습니다. 

만약 데이터 파일도 포함하고 싶다면:
1. `.gitignore`에서 `data/` 라인을 제거하거나
2. Git LFS 사용: `git lfs track "*.csv"`

## 문제 해결

### "fatal: Unable to create 'index.lock'" 오류

```powershell
# lock 파일 삭제
Remove-Item .git/index.lock -ErrorAction SilentlyContinue
```

### 한글 경로 문제

PowerShell에서 한글 경로가 제대로 인식되지 않으면:
- VS Code의 통합 터미널 사용
- 또는 Git Bash 사용

## 완료 후 확인

GitHub 저장소 페이지에서 다음 파일들이 보여야 합니다:
- ✅ README.md
- ✅ LICENSE
- ✅ requirements.txt
- ✅ .gitignore
- ✅ src/ 폴더 (모든 Python 파일)
- ✅ configs/ 폴더
- ✅ preprocess.py
- ✅ train.py

