# 🚀 빠른 시작 가이드

AI Prompt Test Bench를 5분 안에 실행해보세요!

## ⚡ 1분 설치

### 1단계: 환경 변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=sk-your-openai-api-key-here
GCP_PROJECT=your-gcp-project-id
EOF
```

### 2단계: Google Cloud 인증
```bash
# gcloud CLI로 인증 (권장)
gcloud auth application-default login
gcloud config set project your-gcp-project-id
```

### 3단계: 의존성 설치
```bash
# uv 사용 (빠름)
uv sync

# 또는 pip 사용
pip install -e .
```

## ⚡ 1분 설정

### BigQuery 테이블 확인
기존 테이블이 이미 설정되어 있으므로 확인만 하면 됩니다:
- `ai_sandbox.model_pricing` - 모델 가격 정보
- `ai_sandbox.prompt` - 프롬프트 관리
- `ai_sandbox.run_log` - 실행 로그

테이블이 없다면 BigQuery 콘솔에서 생성해주세요.

## ⚡ 실행!

### 간편 실행 (권장)
```bash
./run_ai_testbench.sh
```

### 수동 실행
```bash
streamlit run test_ai.py
```

브라우저에서 http://localhost:8800 접속!

## 🎮 5분 튜토리얼

### 1단계: 첫 번째 테스트 (1분)
1. 사이드바에서 모델 선택 (예: `gpt-4o-mini`)
2. System 프롬프트 입력: 
   ```
   당신은 친절한 한국어 AI 어시스턴트입니다.
   ```
3. 사용자 입력에 간단한 질문: `안녕하세요!`
4. "🚀 AI에게 질문하기" 버튼 클릭

### 2단계: 프롬프트 저장 (1분)
1. 프로젝트명 선택 또는 새로 입력 (예: `AI_Assistant`)
2. 테스터명 입력 (예: `홍길동`)
3. System 프롬프트명: `korean_assistant`
4. 버전명: `v1.0.0`
5. "💾 System 프롬프트 저장" 버튼 클릭

### 3단계: 템플릿 변수 사용 (2분)
1. User 프롬프트 템플릿:
   ```
   ${city}의 ${cuisine} 맛집을 ${budget}원 예산으로 추천해주세요.
   ```
2. 템플릿 변수:
   ```json
   {
     "city": "서울",
     "cuisine": "한식",
     "budget": "50000"
   }
   ```
3. 사용자 입력: `3곳 정도 추천해주세요.`
4. 실행!

### 4단계: 배치 테스트 (1분)
1. 테스트 모드에서 "🚀 배치 테스트 (여러 데이터 동시)" 선택
2. 테스트 데이터 테이블 선택
3. 입력 컬럼 선택 후 데이터 로드
4. 체크박스로 테스트할 데이터 선택
5. 병렬 실행 후 결과 확인

### 5단계: 결과 분석 (1분)
- 응답 시간, 토큰 사용량, 예상 비용 확인
- BigQuery에서 로그 확인:
  ```sql
  SELECT * FROM ai_sandbox.run_log 
  ORDER BY ts DESC 
  LIMIT 10;
  ```

## 🔥 고급 팁

### 프롬프트 엔지니어링
```text
# 좋은 System 프롬프트 예시
당신은 전문적인 [역할]입니다.
다음 규칙을 따라주세요:
1. 정확한 정보만 제공
2. 모르는 것은 솔직히 모른다고 답변
3. 단계별로 설명
4. 한국어로 답변

답변 형식:
- 핵심 답변
- 상세 설명
- 추가 팁 (선택사항)
```

### 템플릿 변수 활용
```json
{
  "role": "데이터 분석가",
  "task": "매출 분석",
  "format": "테이블",
  "language": "한국어"
}
```

### A/B 테스트
1. 같은 내용으로 다른 버전의 프롬프트 생성
2. 각각 테스트 실행
3. 결과 비교 분석

## 🆘 문제 해결

### 자주 묻는 질문

**Q: "OpenAI API 키 오류"가 나와요.**
A: 
- API 키가 정확한지 확인
- 사용량 한도 확인 (https://platform.openai.com/usage)
- 결제 정보 업데이트

**Q: "BigQuery 권한 오류"가 나와요.**
A:
```bash
gcloud auth application-default login
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:your-email@domain.com" \
    --role="roles/bigquery.dataEditor"
```

**Q: 모델 목록이 안 보여요.**
A:
- 인터넷 연결 확인
- OpenAI API 상태 확인
- "🔄 모델 새로고침" 버튼 클릭

**Q: 느려요.**
A:
- 가벼운 모델 사용 (`gpt-4o-mini`)
- 짧은 프롬프트 테스트
- 스트리밍 모드 활용

## 📚 다음 단계

- [상세 문서](README_AI_TESTBENCH.md) 읽기
- 프롬프트 라이브러리 구축
- 팀과 프롬프트 공유
- BigQuery에서 성능 분석

## 💡 더 많은 예시

### 창작 도우미
```text
System: 당신은 창의적인 스토리텔러입니다.
User: ${genre} 장르의 ${length} 단편소설 아이디어를 제안해주세요.
Variables: {"genre": "SF", "length": "단편"}
```

### 코드 리뷰어
```text
System: 당신은 시니어 개발자입니다. 코드 리뷰를 수행해주세요.
User: 다음 ${language} 코드를 리뷰해주세요: ${code}
Variables: {"language": "Python", "code": "def hello(): print('world')"}
```

### 번역기
```text
System: 당신은 전문 번역가입니다.
User: 다음 텍스트를 ${from_lang}에서 ${to_lang}로 번역해주세요: ${text}
Variables: {"from_lang": "영어", "to_lang": "한국어", "text": "Hello World"}
```

---

🎉 **이제 시작해보세요!** 

궁금한 점이 있으면 언제든 문의해주세요.
