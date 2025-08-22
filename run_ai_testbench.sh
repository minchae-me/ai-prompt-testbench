#!/bin/bash

# AI Prompt Test Bench 실행 스크립트

set -e

echo "🧪 AI Prompt Test Bench 시작 중..."

# 환경 변수 확인
check_env_var() {
    if [ -z "${!1}" ]; then
        echo "❌ 환경 변수 $1이 설정되지 않았습니다."
        echo "💡 .env 파일을 확인하거나 다음 명령어로 설정하세요:"
        echo "   export $1=your_value_here"
        exit 1
    else
        echo "✅ $1 설정됨"
    fi
}

# .env 파일 로드 (있는 경우)
if [ -f .env ]; then
    echo "📄 .env 파일 로드 중..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# 필수 환경 변수 확인
echo "🔍 환경 변수 확인 중..."
check_env_var "OPENAI_API_KEY"
check_env_var "GCP_PROJECT"

# BigQuery 테이블 확인
echo "🗄️ BigQuery 테이블 확인 중..."
python3 -c "
import os
from google.cloud import bigquery

PROJECT_ID = os.environ.get('GCP_PROJECT')
client = bigquery.Client(project=PROJECT_ID)

try:
    # 필요한 테이블들 확인
    dataset_id = f'{PROJECT_ID}.ai_sandbox'
    client.get_dataset(dataset_id)
    
    # 핵심 테이블들 존재 확인
    required_tables = ['model_pricing', 'prompt', 'run_log']
    for table_name in required_tables:
        table_id = f'{dataset_id}.{table_name}'
        client.get_table(table_id)
    
    print('✅ BigQuery 테이블들이 모두 확인됨')
except Exception as e:
    print(f'⚠️ BigQuery 테이블 확인 실패: {e}')
    print('💡 필요한 테이블: model_pricing, prompt, run_log')
    print('💡 BigQuery 콘솔에서 테이블들을 확인해주세요.')
    print('💡 모델 가격 정보가 없다면: python insert_model_pricing.py')
    exit(1)
"

# 의존성 확인
echo "📦 의존성 확인 중..."
python3 -c "
try:
    import streamlit
    import openai
    from google.cloud import bigquery
    print('✅ 모든 의존성이 설치되어 있습니다.')
except ImportError as e:
    print(f'❌ 의존성 설치 필요: {e}')
    print('💡 다음 명령어로 설치하세요:')
    print('   pip install -e .')
    print('   # 또는')
    print('   uv sync')
    exit(1)
"

# Streamlit 실행
echo "🚀 Streamlit 서버 시작 중..."
echo "🌐 브라우저에서 http://localhost:8800 을 열어주세요."
echo "⏹️ 종료하려면 Ctrl+C를 누르세요."
echo ""

# 기본 포트 8800, 사용 중이면 8801, 8802 순으로 시도
PORT=8800
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    echo "⚠️ 포트 $PORT이 사용 중입니다. 다음 포트를 시도합니다..."
    PORT=$((PORT + 1))
    if [ $PORT -gt 8810 ]; then
        echo "❌ 사용 가능한 포트를 찾을 수 없습니다."
        exit 1
    fi
done

echo "🔌 포트 $PORT에서 실행 중..."
streamlit run test_ai.py --server.port $PORT --server.address localhost

echo "👋 AI Prompt Test Bench가 종료되었습니다."
