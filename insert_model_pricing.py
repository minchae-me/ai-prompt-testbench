#!/usr/bin/env python3
"""
모델 가격 정보를 BigQuery에 삽입하는 스크립트
OpenAI API에서 모델 목록을 가져와서 최신 가격 정보로 업데이트합니다.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery
from openai import OpenAI

# .env 파일 로드
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드됨: {env_path}")
    else:
        print(f"⚠️ .env 파일을 찾을 수 없습니다: {env_path}")
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않았습니다. 환경 변수를 직접 설정해주세요.")

PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
DATASET_ID = "ai_sandbox"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PROJECT_ID:
    print("❌ 환경 변수 GCP_PROJECT 또는 GOOGLE_CLOUD_PROJECT를 설정해주세요.")
    print("💡 .env 파일에 다음과 같이 추가하세요:")
    print("   GCP_PROJECT=your_project_id")
    exit(1)

if not OPENAI_API_KEY:
    print("❌ 환경 변수 OPENAI_API_KEY를 설정해주세요.")
    print("💡 .env 파일에 다음과 같이 추가하세요:")
    print("   OPENAI_API_KEY=sk-your_api_key_here")
    exit(1)

print(f"프로젝트: {PROJECT_ID}")
print(f"데이터셋: {DATASET_ID}")

# 클라이언트 초기화
bq_client = bigquery.Client(project=PROJECT_ID)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_openai_chat_models():
    """OpenAI API에서 채팅 모델 목록을 가져옵니다."""
    try:
        print("📡 OpenAI API에서 모델 목록 조회 중...")
        models = openai_client.models.list()

        # 채팅용이 아닌 모델들 필터링
        blocked_pattern = re.compile(
            r"(whisper|tts|embedding|dall-e|image|moderation|audio)", re.I
        )

        chat_models = []
        for model in models:
            if not blocked_pattern.search(model.id):
                chat_models.append(model.id)

        chat_models.sort()
        print(f"✅ 채팅 모델 {len(chat_models)}개 발견")

        return chat_models
    except Exception as e:
        print(f"❌ OpenAI 모델 목록 조회 실패: {e}")
        return []


def get_model_pricing_data(model_name: str):
    """모델명에 따른 추정 가격 정보를 반환합니다."""

    # OpenAI 공식 가격 정보 (2024년 12월 기준 - platform.openai.com/docs/pricing)
    known_prices = {
        # ===========================================
        # GPT-4o 계열 (Latest Generation) - 2024년 12월 기준
        # ===========================================
        "gpt-4o": {"input": 0.005, "output": 0.015},  # $5.00/$15.00 per 1M tokens
        "gpt-4o-2024-11-20": {"input": 0.005, "output": 0.015},
        "gpt-4o-2024-08-06": {"input": 0.005, "output": 0.015},
        "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {
            "input": 0.00015,
            "output": 0.0006,
        },  # $0.15/$0.60 per 1M tokens
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-audio-preview": {"input": 0.005, "output": 0.015},  # Audio-enabled
        "gpt-4o-realtime-preview": {"input": 0.005, "output": 0.015},  # Real-time API
        "gpt-4o-vision": {"input": 0.005, "output": 0.015},  # Vision-enabled
        # ===========================================
        # O1 계열 (Reasoning Models)
        # ===========================================
        "o1-preview": {"input": 0.015, "output": 0.06},
        "o1-preview-2024-09-12": {"input": 0.015, "output": 0.06},
        "o1-mini": {"input": 0.003, "output": 0.012},
        "o1-mini-2024-09-12": {"input": 0.003, "output": 0.012},
        # ===========================================
        # O3 계열 (Next Generation Reasoning - December 2024)
        # ===========================================
        "o3-mini": {"input": 0.004, "output": 0.016},  # Estimated pricing
        "o3": {"input": 0.02, "output": 0.08},  # Estimated pricing
        "o3-preview": {"input": 0.02, "output": 0.08},  # Preview version
        # ===========================================
        # GPT-5 계열 (Latest Models)
        # ===========================================
        "gpt-5": {"input": 0.00125, "output": 0.01},  # $1.25/$10.00 per 1M tokens
        "gpt-5-mini": {"input": 0.00025, "output": 0.002},  # $0.25/$2.00 per 1M tokens
        "gpt-5-nano": {"input": 0.0005, "output": 0.004},  # $0.50/$4.00 per 1M tokens
        "gpt-5-chat-latest": {
            "input": 0.00125,
            "output": 0.01,
        },  # $1.25/$10.00 per 1M tokens
        # ===========================================
        # GPT-4 Turbo 계열
        # ===========================================
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03},
        # ===========================================
        # GPT-4 Standard 계열
        # ===========================================
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-0613": {"input": 0.03, "output": 0.06},
        "gpt-4-0314": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-32k-0613": {"input": 0.06, "output": 0.12},
        "gpt-4-32k-0314": {"input": 0.06, "output": 0.12},
        # ===========================================
        # GPT-3.5 Turbo 계열
        # ===========================================
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0301": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-3.5-turbo-16k-0613": {"input": 0.003, "output": 0.004},
        "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
        # ===========================================
        # ChatGPT 계열
        # ===========================================
        "chatgpt-4o-latest": {"input": 0.005, "output": 0.015},  # Same as GPT-4o
        # ===========================================
        # Base Models (Completion/Legacy)
        # ===========================================
        "davinci-002": {"input": 0.002, "output": 0.002},
        "babbage-002": {"input": 0.0004, "output": 0.0004},
        "text-davinci-003": {"input": 0.02, "output": 0.02},
        "text-davinci-002": {"input": 0.02, "output": 0.02},
        # ===========================================
        # Fine-tuning Models
        # ===========================================
        "gpt-4o-fine-tuned": {
            "input": 0.005,
            "output": 0.015,
        },  # Training: $25/1M tokens
        "gpt-4o-mini-fine-tuned": {
            "input": 0.0003,
            "output": 0.0012,
        },  # Training: $3/1M tokens
        "gpt-3.5-turbo-fine-tuned": {
            "input": 0.003,
            "output": 0.006,
        },  # Training: $8/1M tokens
        "davinci-002-fine-tuned": {
            "input": 0.006,
            "output": 0.006,
        },  # Training: $6/1M tokens
        "babbage-002-fine-tuned": {
            "input": 0.0016,
            "output": 0.0016,
        },  # Training: $2.4/1M tokens
        # ===========================================
        # Experimental/Preview Models
        # ===========================================
        "gpt-4-with-browsing": {"input": 0.03, "output": 0.06},  # Deprecated
        "gpt-4-code-interpreter": {"input": 0.03, "output": 0.06},  # Deprecated
        "gpt-4-plugins": {"input": 0.03, "output": 0.06},  # Deprecated
    }

    # 정확한 모델명 매칭
    if model_name in known_prices:
        return known_prices[model_name]

    # 패턴 기반 추정
    model_lower = model_name.lower()

    # GPT-5 계열
    if "gpt-5" in model_lower:
        if "mini" in model_lower:
            return {"input": 0.0003, "output": 0.001}
        elif "turbo" in model_lower:
            return {"input": 0.005, "output": 0.02}
        else:
            return {"input": 0.01, "output": 0.04}

    # O3 계열
    elif "o3" in model_lower:
        if "mini" in model_lower:
            return {"input": 0.004, "output": 0.016}
        else:
            return {"input": 0.02, "output": 0.08}

    # O1 계열
    elif "o1" in model_lower:
        if "mini" in model_lower:
            return {"input": 0.003, "output": 0.012}
        else:
            return {"input": 0.015, "output": 0.06}

    # GPT-4o 계열
    elif "gpt-4o" in model_lower:
        if "mini" in model_lower:
            return {"input": 0.00015, "output": 0.0006}  # $0.15/$0.60 per 1M
        else:
            return {"input": 0.005, "output": 0.015}  # $5.00/$15.00 per 1M

    # GPT-4 계열
    elif "gpt-4" in model_lower:
        if "32k" in model_lower:
            return {"input": 0.06, "output": 0.12}
        elif "turbo" in model_lower or "preview" in model_lower:
            return {"input": 0.01, "output": 0.03}
        else:
            return {"input": 0.03, "output": 0.06}

    # GPT-3.5 계열
    elif "gpt-3.5" in model_lower:
        if "16k" in model_lower:
            return {"input": 0.003, "output": 0.004}
        elif "instruct" in model_lower:
            return {"input": 0.0015, "output": 0.002}
        else:
            return {"input": 0.0005, "output": 0.0015}

    # ChatGPT 계열
    elif "chatgpt" in model_lower:
        return {"input": 0.005, "output": 0.015}

    # Base models
    elif "davinci" in model_lower:
        return {"input": 0.002, "output": 0.002}
    elif "babbage" in model_lower:
        return {"input": 0.0004, "output": 0.0004}

    # 기본값 (GPT-3.5-turbo 수준)
    else:
        return {"input": 0.001, "output": 0.002}


def upsert_model_pricing():
    """OpenAI 모델들의 가격 정보를 BigQuery에 UPSERT (기존 데이터 업데이트 + 새 데이터 삽입)"""

    # 1. OpenAI에서 채팅 모델 목록 가져오기
    chat_models = get_openai_chat_models()

    if not chat_models:
        print("❌ 채팅 모델을 찾을 수 없습니다.")
        return False

    # 2. 기존 모델 목록 조회
    print("📋 기존 모델 가격 정보 조회 중...")
    existing_models_query = f"""
    SELECT model_name, input_per_1k, output_per_1k 
    FROM `{PROJECT_ID}.{DATASET_ID}.model_pricing` 
    WHERE effective_to IS NULL
    """

    try:
        existing_models_result = bq_client.query(existing_models_query)
        existing_models = {
            row.model_name: {
                "input": float(row.input_per_1k),
                "output": float(row.output_per_1k),
            }
            for row in existing_models_result
        }
        print(f"✅ 기존 모델 {len(existing_models)}개 확인")
    except Exception as e:
        print(f"⚠️ 기존 모델 조회 실패 (테이블이 비어있을 수 있음): {e}")
        existing_models = {}

    # 3. 각 모델에 대한 가격 정보 생성 및 비교
    new_models = []
    updated_models = []
    unchanged_models = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"💰 {len(chat_models)}개 모델의 가격 정보 분석 중...")

    for model_name in chat_models:
        pricing = get_model_pricing_data(model_name)
        new_pricing = {"input": pricing["input"], "output": pricing["output"]}

        if model_name in existing_models:
            existing_pricing = existing_models[model_name]
            if (
                existing_pricing["input"] != new_pricing["input"]
                or existing_pricing["output"] != new_pricing["output"]
            ):
                updated_models.append(
                    {
                        "model_name": model_name,
                        "old_input": existing_pricing["input"],
                        "old_output": existing_pricing["output"],
                        "new_input": new_pricing["input"],
                        "new_output": new_pricing["output"],
                    }
                )
            else:
                unchanged_models.append(model_name)
        else:
            new_models.append(
                {
                    "model_name": model_name,
                    "input_per_1k": new_pricing["input"],
                    "output_per_1k": new_pricing["output"],
                    "effective_from": current_time,
                    "effective_to": None,
                }
            )

    # 4. 결과 요약 출력
    print("\n📊 분석 결과:")
    print(f"  🆕 새로운 모델: {len(new_models)}개")
    print(f"  🔄 업데이트 필요: {len(updated_models)}개")
    print(f"  ✅ 변경사항 없음: {len(unchanged_models)}개")

    # 5. 기존 모델 업데이트 (effective_to 설정)
    if updated_models:
        print(f"\n🔄 {len(updated_models)}개 모델 가격 업데이트 중...")
        for model_data in updated_models:
            model_name = model_data["model_name"]

            # 기존 레코드에 effective_to 설정
            update_query = f"""
            UPDATE `{PROJECT_ID}.{DATASET_ID}.model_pricing` 
            SET effective_to = '{current_time}'
            WHERE model_name = '{model_name}' AND effective_to IS NULL
            """

            try:
                bq_client.query(update_query)
                print(f"  ✅ {model_name} 기존 레코드 만료 처리")
            except Exception as e:
                print(f"  ❌ {model_name} 업데이트 실패: {e}")
                return False

            # 새로운 가격으로 레코드 삽입
            new_record = {
                "model_name": model_name,
                "input_per_1k": model_data["new_input"],
                "output_per_1k": model_data["new_output"],
                "effective_from": current_time,
                "effective_to": None,
            }
            new_models.append(new_record)

    # 6. 새로운 모델들 삽입
    if new_models:
        print(f"\n➕ {len(new_models)}개 모델 데이터 삽입 중...")
        table_ref = bq_client.dataset(DATASET_ID).table("model_pricing")

        try:
            errors = bq_client.insert_rows_json(table_ref, new_models)
            if errors:
                print(f"❌ 데이터 삽입 오류: {errors}")
                return False
            else:
                print(f"✅ 모델 데이터 {len(new_models)}개 삽입 완료")
        except Exception as e:
            print(f"❌ 삽입 실패: {e}")
            return False

    # 7. 결과 상세 출력
    if updated_models:
        print("\n🔄 업데이트된 모델들:")
        for model_data in updated_models[:10]:  # 처음 10개만 표시
            print(
                f"  • {model_data['model_name']:<35} "
                f"${model_data['old_input']:.6f}→${model_data['new_input']:.6f} / "
                f"${model_data['old_output']:.6f}→${model_data['new_output']:.6f}"
            )
        if len(updated_models) > 10:
            print(f"  ... 외 {len(updated_models) - 10}개 모델")

    if new_models:
        print("\n🆕 새로 추가된 모델들:")
        for data in new_models[:10]:  # 처음 10개만 표시
            print(
                f"  • {data['model_name']:<35} 입력: ${data['input_per_1k']:<8} 출력: ${data['output_per_1k']}"
            )
        if len(new_models) > 10:
            print(f"  ... 외 {len(new_models) - 10}개 모델")

    # GPT-5, O3 등 최신 모델들 하이라이트
    latest_new_models = [
        m
        for m in new_models
        if any(keyword in m["model_name"].lower() for keyword in ["gpt-5", "o3", "o1"])
    ]
    if latest_new_models:
        print("\n🔥 새로 추가된 최신 모델들:")
        for data in latest_new_models:
            print(
                f"  • {data['model_name']:<35} 입력: ${data['input_per_1k']:<8} 출력: ${data['output_per_1k']}"
            )

    print(f"\n✅ 총 {len(chat_models)}개 모델 처리 완료!")
    return True


# 하위 호환성을 위해 기존 함수명도 유지
def insert_model_pricing():
    """기존 함수명 호환성 유지"""
    return upsert_model_pricing()


def main():
    """메인 실행 함수"""
    print("💰 모델 가격 정보 삽입을 시작합니다...\n")

    try:
        # 테이블 존재 확인
        dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
        required_tables = ["model_pricing", "run_log"]

        for table_name in required_tables:
            table_id = f"{dataset_id}.{table_name}"
            try:
                bq_client.get_table(table_id)
                print(f"✅ 테이블 확인: {table_name}")
            except Exception:
                print(f"❌ 테이블 없음: {table_name}")
                print(f"💡 BigQuery 콘솔에서 {table_name} 테이블을 생성해주세요.")
                return

        print("\n📝 데이터 삽입 중...")

        # 모델 가격 정보 삽입
        if insert_model_pricing():
            print("✅ 모델 가격 정보 삽입 성공")
            print("\n🎉 모델 가격 정보 업데이트가 완료되었습니다!")
            print("💡 이제 Streamlit 앱에서 최신 모델들을 사용할 수 있습니다!")
            print("🚀 실행: streamlit run test_ai.py")
        else:
            print("❌ 모델 가격 정보 삽입 실패")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return


if __name__ == "__main__":
    main()
