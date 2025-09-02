# 🧪 AI Prompt Test Bench

OpenAI GPT 모델을 이용한 AI 프롬프트 테스트 시스템입니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 환경 변수 설정 (.env 파일 생성)
cp env.example .env
# .env 파일을 열어서 API 키와 프로젝트 ID 입력
```

### 2. 의존성 설치
```bash
# uv를 사용한 의존성 설치 (권장)
uv sync

# 또는 개발 모드로 설치
uv pip install -e .
```

### 3. Google Cloud 인증
```bash
gcloud auth application-default login
gcloud config set project your-project-id
```

### 4. 초기 데이터 설정 (필요시)
```bash
uv run python insert_model_pricing.py
```

### 5. 실행
```bash
# 간편 실행
./run_ai_testbench.sh

# 또는 직접 실행
uv run streamlit run test_ai.py
```

브라우저에서 http://localhost:8800 에 접속하세요!

## 📚 자세한 문서

- **[빠른 시작](QUICKSTART.md)** - 5분 튜토리얼

## 🎯 주요 기능

- 🤖 OpenAI 모델 선택 및 테스트
- 📂 프로젝트별 프롬프트 관리 및 버전 관리 (System/User)
- 🔧 템플릿 변수 지원
- 💰 실시간 비용 추정 및 성능 모니터링
- 📊 BigQuery 로그 저장 및 분석
- 🚀 단일 테스트 및 배치 테스트 지원
- 📈 사용량 통계 및 비용 분석

## 💡 요구사항

- Python 3.10+
- OpenAI API 키
- Google Cloud Project (BigQuery 활성화)
- BigQuery 테이블: `ai_sandbox.model_pricing`, `ai_sandbox.prompt`, `ai_sandbox.run_log`

## 🆘 문제 해결

문제가 발생하면 [상세 가이드](QUICKSTART.md)의 트러블슈팅 섹션을 참조하세요.
