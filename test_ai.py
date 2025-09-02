# test_ai.py - AI Prompt Test Bench (Python 3.10+)
import re
import time
import hashlib
import json
import os
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from google.cloud import bigquery
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드 (현재 디렉토리에서)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
AI_DATASET = "ai_sandbox"
BQ_DATASET = "working_gyg"

# 환경 변수 확인
if not PROJECT_ID:
    st.error("❌ 환경 변수 GCP_PROJECT 또는 GOOGLE_CLOUD_PROJECT를 설정해주세요.")
    st.info("💡 .env 파일에 GCP_PROJECT=your_project_id 를 추가하세요.")
    st.stop()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ 환경 변수 OPENAI_API_KEY를 설정해주세요.")
    st.info("💡 .env 파일에 OPENAI_API_KEY=your_api_key 를 추가하세요.")
    st.code(
        """
# .env 파일 예시:
OPENAI_API_KEY=sk-your_api_key_here
GCP_PROJECT=your_project_id
    """
    )
    st.stop()

# --- GCP/BQ 클라이언트
try:
    bq = bigquery.Client(project=PROJECT_ID)
except Exception as e:
    st.error(f"❌ BigQuery 클라이언트 초기화 실패: {e}")
    st.info("💡 Google Cloud 인증을 확인하세요: gcloud auth application-default login")
    st.stop()

# --- OpenAI 클라이언트
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    st.stop()


# 1) 모델 목록 가져오기 (+캐시)
@st.cache_data(ttl=600)
def fetch_all_models_from_openai() -> list[str]:
    ids = []
    try:
        # SDK가 generator를 반환 -> for로 순회
        for m in client.models.list():
            ids.append(m.id)
    except Exception as e:
        st.warning(f"/v1/models 조회 실패: {e}")
    return sorted(set(ids))


# 2) chat/completions에 쓸 수 없는 모델만 제외 (유연한 필터링)
def filter_chat_models(model_ids: list[str]) -> list[str]:
    # 명백히 채팅용이 아닌 모델들만 제외
    blocked = re.compile(r"(whisper|tts|embedding|dall-e|image|moderation|audio)", re.I)
    # 차단된 키워드가 포함된 모델만 제외하고 나머지는 모두 허용
    keep = [m for m in model_ids if not blocked.search(m)]
    # 단순히 알파벳 순으로 정렬 (사용자가 원하는 모델을 쉽게 찾을 수 있도록)
    return sorted(keep)


# OpenAI 모델 목록을 미리 가져오기 (사이드바에서 재사용)
@st.cache_data(ttl=600)
def get_available_models():
    return filter_chat_models(fetch_all_models_from_openai())


# ---------- BigQuery helpers ----------
def bq_query(
    sql: str, params: Dict[str, Any] | None = None
) -> List[bigquery.table.Row]:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(k, "STRING", v)
            for k, v in (params or {}).items()
        ]
    )
    return list(bq.query(sql, job_config=job_config).result())


def insert_rows_json(table: str, rows: List[Dict[str, Any]]):
    table_ref = bq.dataset(AI_DATASET).table(table)
    errors = bq.insert_rows_json(table_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert error: {errors}")


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------- Load model list ----------
def load_models() -> List[str]:
    try:
        rows = bq_query(
            f"""
            SELECT DISTINCT model_name 
            FROM `{PROJECT_ID}.{AI_DATASET}.model_pricing`
            WHERE effective_to IS NULL OR effective_to > CURRENT_TIMESTAMP()
            ORDER BY model_name
        """
        )
        return [r["model_name"] for r in rows]
    except Exception as e:
        st.warning(f"⚠️ BigQuery에서 모델 목록을 불러올 수 없습니다: {e}")
        return []


# ---------- Prompt CRUD ----------
def list_project_names() -> List[str]:
    """프로젝트 목록 조회"""
    try:
        rows = bq_query(
            f"""
            SELECT DISTINCT project_name
            FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
            WHERE project_name IS NOT NULL
            ORDER BY project_name
        """
        )
        return [r["project_name"] for r in rows]
    except Exception as e:
        st.warning(f"⚠️ 프로젝트 목록을 불러올 수 없습니다: {e}")
        return []


def list_prompt_names(prompt_type: str, project_name: str = None) -> List[str]:
    """프롬프트 이름 목록 조회 (프로젝트별 필터링 가능)"""
    try:
        if project_name:
            rows = bq_query(
                f"""
                SELECT DISTINCT prompt_name
                FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
                WHERE prompt_type=@t AND project_name=@p
                ORDER BY prompt_name
            """,
                {"t": prompt_type, "p": project_name},
            )
        else:
            rows = bq_query(
                f"""
                SELECT DISTINCT prompt_name
                FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
                WHERE prompt_type=@t
                ORDER BY prompt_name
            """,
                {"t": prompt_type},
            )
        return [r["prompt_name"] for r in rows]
    except Exception as e:
        st.warning(f"⚠️ {prompt_type} 프롬프트 목록을 불러올 수 없습니다: {e}")
        return []


def list_versions(
    prompt_type: str, prompt_name: str, project_name: str = None
) -> List[str]:
    """프롬프트 버전 목록 조회 (프로젝트별 필터링 가능)"""
    if project_name:
        rows = bq_query(
            f"""
            SELECT version_name
            FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
            WHERE prompt_type=@t AND prompt_name=@n AND project_name=@p
            ORDER BY created_at DESC
        """,
            {"t": prompt_type, "n": prompt_name, "p": project_name},
        )
    else:
        rows = bq_query(
            f"""
            SELECT version_name
            FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
            WHERE prompt_type=@t AND prompt_name=@n
            ORDER BY created_at DESC
        """,
            {"t": prompt_type, "n": prompt_name},
        )
    return [r["version_name"] for r in rows]


def load_prompt_content(
    prompt_type: str, prompt_name: str, version_name: str, project_name: str = None
) -> str:
    """프롬프트 내용 로드 (프로젝트별 필터링 가능)"""
    if project_name:
        rows = bq_query(
            f"""
            SELECT content
            FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
            WHERE prompt_type=@t AND prompt_name=@n AND version_name=@v AND project_name=@p
            LIMIT 1
        """,
            {"t": prompt_type, "n": prompt_name, "v": version_name, "p": project_name},
        )
    else:
        rows = bq_query(
            f"""
            SELECT content
            FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
            WHERE prompt_type=@t AND prompt_name=@n AND version_name=@v
            LIMIT 1
        """,
            {"t": prompt_type, "n": prompt_name, "v": version_name},
        )
    return rows[0]["content"] if rows else ""


def get_prompt_project(prompt_type: str, prompt_name: str, version_name: str) -> str:
    """프롬프트의 프로젝트명 조회"""
    rows = bq_query(
        f"""
        SELECT project_name
        FROM `{PROJECT_ID}.{AI_DATASET}.prompt`
        WHERE prompt_type=@t AND prompt_name=@n AND version_name=@v
        LIMIT 1
    """,
        {"t": prompt_type, "n": prompt_name, "v": version_name},
    )
    return rows[0]["project_name"] if rows and rows[0]["project_name"] else ""


def upsert_prompt_version(
    prompt_type: str,
    prompt_name: str,
    version_name: str,
    content: str,
    created_by: str,
    project_name: str = None,
):
    insert_rows_json(
        "prompt",
        [
            {
                "prompt_type": prompt_type,
                "prompt_name": prompt_name,
                "version_name": version_name,
                "content": content,
                "project_name": project_name,
                "is_active": True,
                "created_by": created_by,
                "content_md5": md5(content),
            }
        ],
    )


def update_project_name(old_project_name: str, new_project_name: str) -> bool:
    """프로젝트명 일괄 변경"""
    try:
        # BigQuery 클라이언트 생성
        bq_client = bigquery.Client()

        # 프로젝트명 업데이트 쿼리
        query = f"""
        UPDATE `{PROJECT_ID}.{AI_DATASET}.prompt` 
        SET project_name = @new_name
        WHERE project_name = @old_name
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("new_name", "STRING", new_project_name),
                bigquery.ScalarQueryParameter("old_name", "STRING", old_project_name),
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        query_job.result()  # 결과 대기

        # 업데이트된 행 수 확인
        rows_affected = query_job.num_dml_affected_rows
        st.success(
            f"✅ 프로젝트명 변경 완료! {rows_affected}개 프롬프트가 업데이트되었습니다."
        )
        return True

    except Exception as e:
        st.error(f"❌ 프로젝트명 변경 실패: {e}")
        return False


# ---------- Cost estimator ----------
def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float | None:
    # model_pricing 테이블에서 가장 최신 가격 정보 조회 (ai_sandbox 데이터셋 사용)
    rows = bq_query(
        f"""
        SELECT input_per_1k, output_per_1k
        FROM `{PROJECT_ID}.{AI_DATASET}.model_pricing`
        WHERE model_name=@m 
        AND (effective_to IS NULL OR effective_to > CURRENT_TIMESTAMP())
        ORDER BY effective_from DESC
        LIMIT 1
    """,
        {"m": model},
    )
    if not rows:
        return None
    pin, pout = rows[0]["input_per_1k"], rows[0]["output_per_1k"]
    if pin is None or pout is None:
        return None
    # Decimal 타입을 float으로 변환하여 계산
    pin_float = float(pin)
    pout_float = float(pout)
    return (in_tokens / 1000) * pin_float + (out_tokens / 1000) * pout_float


# ---------- UI ----------
st.set_page_config(page_title="AI Prompt Test Bench", layout="wide")
st.title("🧪 AI Prompt Test Bench (GPT)")

with st.sidebar:
    st.subheader("🤖 모델 선택")

    # OpenAI API에서 실시간 모델 목록 가져오기
    st.caption("🔄 실시간 모델 목록 (10분 캐시)")
    if st.button("🔄 모델 새로고침", key="refresh_models"):
        st.cache_data.clear()
        st.rerun()

    model_name = (
        st.selectbox("모델 선택", get_available_models(), key="model_select")
        if get_available_models()
        else st.text_input("모델 직접 입력", "gpt-4o-mini", key="model_input")
    )

    if get_available_models():
        st.caption(f"총 {len(get_available_models())}개 모델 사용 가능")

    st.markdown("---")
    st.subheader("📝 프롬프트 관리")

    # 프로젝트 선택 섹션
    st.caption("📂 프로젝트 관리")
    project_names = list_project_names()
    project_options = ["(새 프로젝트)"] + project_names
    selected_project_option = st.selectbox(
        "프로젝트 선택", project_options, key="project_select"
    )

    if selected_project_option == "(새 프로젝트)":
        current_project = st.text_input(
            "새 프로젝트명", "my_project", key="new_project_name"
        )
    else:
        current_project = selected_project_option
        # 기존 프로젝트 수정 옵션
        with st.expander("프로젝트명 수정"):
            new_project_name = st.text_input(
                "수정할 프로젝트명", current_project, key="edit_project_name"
            )
            if st.button("📝 프로젝트명 변경", key="update_project"):
                if new_project_name and new_project_name != current_project:
                    if update_project_name(current_project, new_project_name):
                        st.info(
                            f"프로젝트명이 '{current_project}'에서 '{new_project_name}'으로 변경되었습니다."
                        )
                        time.sleep(2)
                        st.rerun()
                else:
                    st.warning("새로운 프로젝트명을 입력해주세요.")

    # System Prompt 섹션
    st.caption("🎭 System Prompt")
    sys_names = ["(새로 입력)"] + list_prompt_names(
        "system",
        current_project if selected_project_option != "(새 프로젝트)" else None,
    )
    sys_name = st.selectbox("System 프롬프트 선택", sys_names, key="sys_name")
    sys_version = None
    sys_content = ""
    sys_project = current_project

    if sys_name != "(새로 입력)":
        versions = list_versions(
            "system",
            sys_name,
            current_project if selected_project_option != "(새 프로젝트)" else None,
        )
        if versions:
            selected_sys_version = st.selectbox(
                "불러올 버전", versions, key="sys_version_select"
            )
            sys_content = load_prompt_content(
                "system",
                sys_name,
                selected_sys_version,
                current_project if selected_project_option != "(새 프로젝트)" else None,
            )
            # 기존 프롬프트의 프로젝트명 조회
            existing_project = get_prompt_project(
                "system", sys_name, selected_sys_version
            )
            if existing_project:
                sys_project = existing_project

            # 저장할 때 사용할 버전명 (수정 가능)
            sys_version = st.text_input(
                "저장할 버전명",
                selected_sys_version,
                key="new_sys_version",
                help="기존 버전을 수정하거나 새 버전명을 입력하세요 (예: v1.0.1)",
            )
        else:
            st.warning("저장된 버전이 없습니다.")
    else:
        sys_name = st.text_input("새 System 프롬프트명", "default", key="new_sys_name")
        sys_version = st.text_input("새 버전명", "v1.0.0", key="new_sys_version")

    # User Prompt 섹션
    st.caption("👤 User Prompt")
    usr_names = ["(새로 입력)"] + list_prompt_names(
        "user", current_project if selected_project_option != "(새 프로젝트)" else None
    )
    usr_name = st.selectbox("User 프롬프트 선택", usr_names, key="usr_name")
    usr_version = None
    usr_content = ""
    usr_project = current_project

    if usr_name != "(새로 입력)":
        versions = list_versions(
            "user",
            usr_name,
            current_project if selected_project_option != "(새 프로젝트)" else None,
        )
        if versions:
            selected_usr_version = st.selectbox(
                "불러올 버전", versions, key="usr_version_select"
            )
            usr_content = load_prompt_content(
                "user",
                usr_name,
                selected_usr_version,
                current_project if selected_project_option != "(새 프로젝트)" else None,
            )
            # 기존 프롬프트의 프로젝트명 조회
            existing_project = get_prompt_project(
                "user", usr_name, selected_usr_version
            )
            if existing_project:
                usr_project = existing_project

            # 저장할 때 사용할 버전명 (수정 가능)
            usr_version = st.text_input(
                "저장할 버전명",
                selected_usr_version,
                key="new_usr_version",
                help="기존 버전을 수정하거나 새 버전명을 입력하세요 (예: v1.0.1)",
            )
        else:
            st.warning("저장된 버전이 없습니다.")
    else:
        usr_name = st.text_input(
            "새 User 프롬프트명", "default_user", key="new_usr_name"
        )
        usr_version = st.text_input("새 버전명", "v1.0.0", key="new_usr_version")

    st.markdown("---")
    st.subheader("👨‍💻 테스터 정보")
    tester = st.text_input("테스터명", os.environ.get("USER", ""), key="tester_name")

# 메인 편집 영역
st.subheader("📝 프롬프트 편집 및 저장")

# 현재 프로젝트 정보 표시
if current_project:
    st.info(f"📂 **현재 프로젝트**: `{current_project}`")
else:
    st.warning("⚠️ 프로젝트를 선택하거나 새로 입력해주세요.")

# 2열 레이아웃으로 프롬프트 편집
col1, col2 = st.columns(2)

with col1:
    st.caption("🎭 System Prompt")

    # 프로젝트명 수정 옵션 (기존 프롬프트인 경우)
    if sys_name != "(새로 입력)" and sys_name and sys_version:
        with st.expander("프로젝트 설정"):
            sys_project = st.text_input(
                "System 프롬프트 프로젝트명",
                sys_project or current_project,
                key="sys_project_edit",
                help="이 프롬프트가 속할 프로젝트를 지정하세요",
            )
    else:
        sys_project = current_project

    sys_content = st.text_area(
        "시스템 프롬프트",
        sys_content,
        height=200,
        placeholder="AI의 역할, 성격, 규칙 등을 정의해주세요...",
        key="sys_content_editor",
    )

    # System 프롬프트 저장 버튼
    if st.button(
        "💾 System 프롬프트 저장",
        type="primary",
        use_container_width=True,
        key="save_sys",
    ):
        if sys_name and sys_version and sys_content and sys_project:
            upsert_prompt_version(
                "system",
                sys_name,
                sys_version,
                sys_content,
                tester or "unknown",
                sys_project,
            )
            st.success(f"✅ System 프롬프트 저장 완료! (프로젝트: {sys_project})")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("System 프롬프트 정보와 프로젝트명을 모두 입력해주세요.")

with col2:
    st.caption("👤 User Prompt")

    # 프로젝트명 수정 옵션 (기존 프롬프트인 경우)
    if usr_name != "(새로 입력)" and usr_name and usr_version:
        with st.expander("프로젝트 설정"):
            usr_project = st.text_input(
                "User 프롬프트 프로젝트명",
                usr_project or current_project,
                key="usr_project_edit",
                help="이 프롬프트가 속할 프로젝트를 지정하세요",
            )
    else:
        usr_project = current_project

    usr_content = st.text_area(
        "사용자 프롬프트",
        usr_content,
        height=200,
        placeholder="기본 사용자 프롬프트를 입력하세요...",
        key="usr_content_editor",
    )

    # User 프롬프트 저장 버튼
    if st.button(
        "💾 User 프롬프트 저장",
        type="primary",
        use_container_width=True,
        key="save_usr",
    ):
        if usr_name and usr_version and usr_content and usr_project:
            upsert_prompt_version(
                "user",
                usr_name,
                usr_version,
                usr_content,
                tester or "unknown",
                usr_project,
            )
            st.success(f"✅ User 프롬프트 저장 완료! (프로젝트: {usr_project})")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("User 프롬프트 정보와 프로젝트명을 모두 입력해주세요.")

# 테스트 모드 선택
test_mode = st.radio(
    "🎯 **테스트 방식을 선택하세요**",
    ["💬 단일 테스트 (즉시 실행)", "🚀 배치 테스트 (여러 데이터 동시)"],
    horizontal=True,
    help="• 단일 테스트: 위에서 입력한 사용자 질문으로 즉시 AI 테스트\n• 배치 테스트: BigQuery에서 여러 데이터를 불러와서 한번에 테스트",
)

if test_mode == "💬 단일 테스트 (즉시 실행)":
    st.markdown("---")
    st.subheader("🚀 단일 테스트 실행")

    # 사용자 입력 영역
    user_input = st.text_area(
        "사용자 입력",
        "",
        height=120,
        placeholder="실제 사용자가 입력할 질문이나 요청을 입력하세요...",
        key="user_input_text",
    )

    # 실행 버튼들
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_btn = st.button(
            "🚀 AI에게 질문하기", type="primary", use_container_width=True
        )
    with col2:
        clear_btn = st.button("🧹 초기화", use_container_width=True)
    with col3:
        show_logs = st.button("📊 로그 보기", use_container_width=True)

    if clear_btn:
        st.rerun()

    if run_btn:
        # 입력 유효성 검사
        if not model_name:
            st.error("❌ 모델을 선택해주세요.")
            st.stop()

        if not sys_content.strip():
            st.error("❌ System 프롬프트를 입력해주세요.")
            st.stop()

        # 최종 메시지 구성
        final_user_content = usr_content + ("\n\n" + user_input if user_input else "")

        if not final_user_content.strip():
            st.error("❌ 사용자 입력 또는 User 프롬프트를 입력해주세요.")
            st.stop()

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": final_user_content},
        ]

        # 요청 미리보기
        with st.expander("📋 요청 미리보기"):
            st.json({"model": model_name, "messages": messages})

        # OpenAI API 호출 (스트리밍)
        start = time.time()
        stream_area = st.empty()
        stream_text = ""

        try:
            # Chat Completions API (stream=True)
            response_stream = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.2, stream=True
            )

            response_data = None
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    stream_text += chunk.choices[0].delta.content
                    stream_area.markdown(f"**🤖 AI 응답 중...**\n\n{stream_text}")

                # 마지막 청크에서 전체 응답 정보 수집
                if chunk.choices[0].finish_reason is not None:
                    response_data = chunk

            latency_ms = int((time.time() - start) * 1000)

            # usage/응답 정보 처리
            # 스트리밍에서는 usage 정보가 제한적이므로 대략적으로 계산
            usage = getattr(response_data, "usage", None) if response_data else None
            if usage:
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            else:
                # usage 정보가 없으면 대략적으로 계산 (1토큰 ≈ 4글자)
                input_tokens = len(" ".join([msg["content"] for msg in messages])) // 4
                output_tokens = len(stream_text) // 4

            # 비용 추정 (model_catalog 단가가 있으면)
            est_cost = estimate_cost_usd(model_name, input_tokens, output_tokens)

            # 최종 응답 표시
            stream_area.markdown(f"**✅ 완료!**\n\n{stream_text}")
            st.success(f"응답 완료! (소요시간: {latency_ms}ms)")

            # 응답 상세 정보
            with st.expander("📊 응답 상세 정보"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("지연시간", f"{latency_ms}ms")
                with col2:
                    st.metric("입력 토큰", f"{input_tokens:,}")
                with col3:
                    st.metric("출력 토큰", f"{output_tokens:,}")

                if est_cost is not None:
                    st.metric("추정 비용", f"${est_cost:.6f}")

            # Raw Response 정보 (간소화)
            with st.expander("📦 요청/응답 JSON"):
                st.json(
                    {
                        "request": {"model": model_name, "messages": messages},
                        "response": {
                            "content": stream_text,
                            "usage": {
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                            },
                            "latency_ms": latency_ms,
                        },
                    }
                )

            # 로그 저장
            insert_rows_json(
                "run_log",
                [
                    {
                        "tester": tester or "unknown",
                        "model_name": model_name,
                        "system_prompt_name": sys_name,
                        "system_prompt_version": sys_version,
                        "user_prompt_name": usr_name,
                        "user_prompt_version": usr_version,
                        "user_input": user_input,
                        "request_json": json.dumps(
                            {"messages": messages, "model": model_name},
                            ensure_ascii=False,
                        ),
                        "response_json": json.dumps(
                            {
                                "content": stream_text,
                                "usage": {
                                    "prompt_tokens": input_tokens,
                                    "completion_tokens": output_tokens,
                                },
                                "latency_ms": latency_ms,
                            },
                            ensure_ascii=False,
                        ),
                        "output_text": stream_text,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "latency_ms": int(latency_ms),
                        "est_cost_usd": est_cost,
                    }
                ],
            )

        except Exception as e:
            error_msg = str(e)
            stream_area.empty()

            # 구체적인 에러 메시지 제공
            if "api_key" in error_msg.lower():
                st.error("❌ OpenAI API 키가 설정되지 않았거나 유효하지 않습니다.")
                st.info("💡 환경변수 OPENAI_API_KEY를 확인해주세요.")
            elif "rate_limit" in error_msg.lower():
                st.error("❌ API 요청 한도를 초과했습니다.")
                st.info("💡 잠시 후 다시 시도해주세요.")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                st.error(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
                st.info("💡 다른 모델을 선택하거나 모델명을 확인해주세요.")
            elif "insufficient_quota" in error_msg.lower():
                st.error("❌ OpenAI 계정의 사용량 한도를 초과했습니다.")
                st.info("💡 계정 사용량을 확인하거나 결제 정보를 업데이트해주세요.")
            else:
                st.error(f"❌ OpenAI API 호출 실패: {error_msg}")

            # 에러 로그 저장
            try:
                insert_rows_json(
                    "run_log",
                    [
                        {
                            "tester": tester or "unknown",
                            "model_name": model_name,
                            "system_prompt_name": sys_name,
                            "system_prompt_version": sys_version,
                            "user_prompt_name": usr_name,
                            "user_prompt_version": usr_version,
                            "user_input": user_input,
                            "request_json": json.dumps(
                                {"messages": messages, "model": model_name},
                                ensure_ascii=False,
                            ),
                            "response_json": json.dumps(
                                {"error": error_msg}, ensure_ascii=False
                            ),
                            "output_text": f"ERROR: {error_msg}",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "latency_ms": int((time.time() - start) * 1000),
                            "est_cost_usd": 0.0,
                        }
                    ],
                )
            except Exception as log_error:
                st.warning(f"⚠️ 에러 로그 저장 실패: {log_error}")

    # 로그 보기 기능
    if "show_logs" in locals() and show_logs:
        st.markdown("---")
        st.subheader("📊 사용 로그 및 통계")

    try:
        # 최근 10개 로그 조회
        recent_logs = bq_query(
            f"""
            SELECT 
                ts,
                tester,
                model_name,
                system_prompt_name,
                user_prompt_name,
                SUBSTR(output_text, 1, 100) as output_preview,
                input_tokens,
                output_tokens,
                latency_ms,
                est_cost_usd
            FROM `{PROJECT_ID}.{AI_DATASET}.run_log`
            WHERE tester = @tester
            ORDER BY ts DESC
            LIMIT 10
        """,
            {"tester": tester or "unknown"},
        )

        if recent_logs:
            st.caption("🕒 최근 실행 로그 (10개)")

            # 로그 테이블 표시
            import pandas as pd

            df = pd.DataFrame([dict(row) for row in recent_logs])
            df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            df["output_preview"] = df["output_preview"].apply(
                lambda x: x[:50] + "..." if len(str(x)) > 50 else x
            )

            st.dataframe(
                df,
                column_config={
                    "ts": "실행시간",
                    "model_name": "모델",
                    "system_prompt_name": "System 프롬프트",
                    "user_prompt_name": "User 프롬프트",
                    "output_preview": "응답 미리보기",
                    "input_tokens": st.column_config.NumberColumn(
                        "입력 토큰", format="%d"
                    ),
                    "output_tokens": st.column_config.NumberColumn(
                        "출력 토큰", format="%d"
                    ),
                    "latency_ms": st.column_config.NumberColumn(
                        "지연시간(ms)", format="%d"
                    ),
                    "est_cost_usd": st.column_config.NumberColumn(
                        "비용(USD)", format="$%.6f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

        # 오늘의 통계
        today_stats = bq_query(
            f"""
            SELECT 
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(est_cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency,
                COUNT(DISTINCT model_name) as unique_models
            FROM `{PROJECT_ID}.{AI_DATASET}.run_log`
            WHERE DATE(ts) = CURRENT_DATE()
            AND tester = @tester
        """,
            {"tester": tester or "unknown"},
        )

        if today_stats and today_stats[0]["total_requests"] > 0:
            st.caption("📈 오늘의 통계")
            stats = today_stats[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 요청 수", f"{stats['total_requests']:,}")
            with col2:
                st.metric(
                    "총 토큰",
                    f"{(stats['total_input_tokens'] or 0) + (stats['total_output_tokens'] or 0):,}",
                )
            with col3:
                st.metric("총 비용", f"${stats['total_cost'] or 0:.4f}")
            with col4:
                st.metric("평균 지연시간", f"{stats['avg_latency'] or 0:.0f}ms")

        # 모델별 사용량 통계
        model_stats = bq_query(
            f"""
            SELECT 
                model_name,
                COUNT(*) as requests,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(est_cost_usd) as cost,
                AVG(latency_ms) as avg_latency
            FROM `{PROJECT_ID}.{AI_DATASET}.run_log`
            WHERE DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            AND tester = @tester
            GROUP BY model_name
            ORDER BY requests DESC
        """,
            {"tester": tester or "unknown"},
        )

        if model_stats:
            st.caption("🤖 최근 7일 모델별 사용량")
            model_df = pd.DataFrame([dict(row) for row in model_stats])

            st.dataframe(
                model_df,
                column_config={
                    "model_name": "모델명",
                    "requests": st.column_config.NumberColumn("요청수", format="%d"),
                    "input_tokens": st.column_config.NumberColumn(
                        "입력 토큰", format="%d"
                    ),
                    "output_tokens": st.column_config.NumberColumn(
                        "출력 토큰", format="%d"
                    ),
                    "cost": st.column_config.NumberColumn(
                        "총 비용(USD)", format="$%.4f"
                    ),
                    "avg_latency": st.column_config.NumberColumn(
                        "평균 지연시간(ms)", format="%.0f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info("📝 아직 실행 로그가 없습니다. AI에게 질문을 해보세요!")

    except Exception as e:
        st.error(f"❌ 로그 조회 실패: {e}")
        st.info("💡 BigQuery 연결을 확인하거나 테이블이 생성되었는지 확인해주세요.")

# ---------- 배치 테스트 기능 ----------


def load_test_data(table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """BigQuery에서 테스트 데이터를 로드합니다."""
    try:
        rows = bq_query(
            f"""
            SELECT *
            FROM `{PROJECT_ID}.{BQ_DATASET}.{table_name}`
            LIMIT {limit}
        """
        )
        return [dict(row) for row in rows]
    except Exception as e:
        st.error(f"❌ 테스트 데이터 로드 실패: {e}")
        return []


def get_table_columns(table_name: str) -> List[str]:
    """테이블의 컬럼 목록을 가져옵니다."""
    try:
        table_ref = bq.get_table(f"{PROJECT_ID}.{BQ_DATASET}.{table_name}")
        return [field.name for field in table_ref.schema]
    except Exception as e:
        st.error(f"❌ 테이블 스키마 조회 실패: {e}")
        return []


def load_selected_test_data(
    table_name: str, selected_indices: List[int], input_column: str
) -> List[Dict[str, Any]]:
    """선택된 행의 데이터만 로드합니다."""
    try:
        # 모든 데이터를 로드한 후 선택된 인덱스만 필터링
        all_data = load_test_data(table_name, 1000)  # 충분한 데이터 로드

        selected_data = []
        for idx in selected_indices:
            if idx < len(all_data):
                item = all_data[idx]
                # 선택된 컬럼의 값을 user_input으로 설정
                selected_data.append(
                    {
                        "id": f"row_{idx}",
                        "user_input": str(item.get(input_column, "")),
                        "original_data": item,  # 원본 데이터도 보관
                    }
                )
        return selected_data
    except Exception as e:
        st.error(f"❌ 선택된 데이터 로드 실패: {e}")
        return []


def list_test_tables() -> List[str]:
    """사용 가능한 테스트 데이터 테이블 목록을 가져옵니다."""
    try:
        tables = bq.list_tables(BQ_DATASET)
        # 'test_' 로 시작하는 테이블들만 필터링
        test_tables = [table.table_id for table in tables if table.table_id]
        return sorted(test_tables)
    except Exception as e:
        st.warning(f"⚠️ 테스트 테이블 목록 조회 실패: {e}")
        return []


def call_openai_api(
    model_name: str, messages: List[Dict], test_id: str, user_input: str = ""
) -> Dict[str, Any]:
    """단일 OpenAI API 호출을 수행합니다."""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.2
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # 응답 처리
        content = response.choices[0].message.content
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # 비용 계산
        est_cost = estimate_cost_usd(model_name, input_tokens, output_tokens)
        # BigQuery NUMERIC 정밀도 문제 해결을 위해 소수점 6자리로 반올림
        est_cost_rounded = round(est_cost, 6) if est_cost else 0.0

        return {
            "test_id": test_id,
            "status": "success",
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "est_cost_usd": est_cost_rounded,
            "error": None,
            "user_input": user_input,
            "request_json": {"messages": messages, "model": model_name},
        }

    except Exception as e:
        return {
            "test_id": test_id,
            "status": "error",
            "content": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": int((time.time() - start_time) * 1000),
            "est_cost_usd": 0.0,
            "error": str(e),
            "user_input": user_input,
            "request_json": {"messages": messages, "model": model_name},
        }


def run_batch_test(
    model_name: str, sys_content: str, test_data: List[Dict], max_workers: int = 5
) -> List[Dict[str, Any]]:
    """배치 테스트를 병렬로 실행합니다."""

    # 프로그레스 바 설정
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    def process_single_test(item):
        test_id = item.get("id", f"test_{len(results)}")
        user_input = item.get("user_input", item.get("input", str(item)))

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_input},
        ]

        return call_openai_api(model_name, messages, test_id, user_input)

    # ThreadPoolExecutor를 사용한 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        futures = [executor.submit(process_single_test, item) for item in test_data]

        # 결과 수집 및 진행상황 업데이트
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=60)  # 60초 타임아웃
                results.append(result)

                # 진행상황 업데이트
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                status_text.text(f"진행중... {i + 1}/{len(futures)} 완료")

            except Exception as e:
                st.error(f"테스트 실행 실패: {e}")
                results.append(
                    {"test_id": f"test_{i}", "status": "error", "error": str(e)}
                )

    progress_bar.progress(1.0)
    status_text.text("✅ 배치 테스트 완료!")

    return results


def save_batch_results(
    results: List[Dict],
    model_name: str,
    sys_name: str,
    sys_version: str,
    usr_name: str,
    usr_version: str,
    tester: str,
):
    """배치 테스트 결과를 BigQuery에 저장합니다."""

    batch_logs = []
    for result in results:
        if result["status"] == "success":
            batch_logs.append(
                {
                    "tester": tester or "unknown",
                    "model_name": model_name,
                    "system_prompt_name": sys_name,
                    "system_prompt_version": sys_version,
                    "user_prompt_name": usr_name,
                    "user_prompt_version": usr_version,
                    "user_input": result.get("user_input", ""),
                    "request_json": json.dumps(
                        result.get("request_json", {"model": model_name}),
                        ensure_ascii=False,
                    ),
                    "response_json": json.dumps(
                        {
                            "content": result["content"],
                            "usage": {
                                "prompt_tokens": result["input_tokens"],
                                "completion_tokens": result["output_tokens"],
                            },
                            "latency_ms": result["latency_ms"],
                        },
                        ensure_ascii=False,
                    ),
                    "output_text": result["content"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_ms": result["latency_ms"],
                    "est_cost_usd": (
                        round(result["est_cost_usd"], 6)
                        if result.get("est_cost_usd")
                        else 0.0
                    ),
                }
            )
        else:
            # 에러 로그
            batch_logs.append(
                {
                    "tester": tester or "unknown",
                    "model_name": model_name,
                    "system_prompt_name": sys_name,
                    "system_prompt_version": sys_version,
                    "user_prompt_name": usr_name,
                    "user_prompt_version": usr_version,
                    "user_input": result.get("user_input", ""),
                    "request_json": json.dumps(
                        result.get("request_json", {"model": model_name}),
                        ensure_ascii=False,
                    ),
                    "response_json": json.dumps(
                        {"error": result["error"]}, ensure_ascii=False
                    ),
                    "output_text": f"ERROR: {result['error']}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": result.get("latency_ms", 0),
                    "est_cost_usd": 0.0,
                }
            )

    try:
        insert_rows_json("run_log", batch_logs)
        st.success(f"✅ 배치 테스트 결과 {len(batch_logs)}개 저장 완료!")
    except Exception as e:
        st.error(f"❌ 배치 결과 저장 실패: {e}")


if test_mode == "🚀 배치 테스트 (여러 데이터 동시)":
    st.subheader("🚀 배치 테스트")
    st.caption("BigQuery에서 여러 테스트 데이터를 불러와서 동시에 실행합니다")

    batch_tab1, batch_tab2, batch_tab3 = st.tabs(
        ["📋 데이터 선택", "⚙️ 실행 설정", "📊 결과"]
    )

    with batch_tab1:
        st.caption("🏗️ 테스트 데이터 설정 및 선택")

        # 한 줄로 배치된 설정 옵션들
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.caption("📊 테이블 선택")
            test_tables = list_test_tables()
            if test_tables:
                selected_table = st.selectbox(
                    "테스트 데이터 테이블",
                    test_tables,
                    key="batch_table",
                    label_visibility="collapsed",
                )
            else:
                st.warning("📝 'gyg_'로 시작하는 테이블이 없습니다.")
                selected_table = st.text_input(
                    "테이블명 직접 입력",
                    "gyg_test_data",
                    key="batch_table_manual",
                    label_visibility="collapsed",
                )

        with col2:
            st.caption("🔧 입력 컬럼 선택")
            if selected_table:
                columns = get_table_columns(selected_table)
                if columns:
                    input_column = st.selectbox(
                        "사용자 입력으로 사용할 컬럼",
                        columns,
                        key="input_column",
                        help="이 컬럼의 값이 AI에게 전달될 사용자 입력으로 사용됩니다.",
                        label_visibility="collapsed",
                    )
                else:
                    st.error("❌ 컬럼을 불러올 수 없습니다.")
                    input_column = None
            else:
                input_column = None
                st.info("← 먼저 테이블을 선택하세요")

        with col3:
            st.caption("📝 로드 개수")
            preview_count = st.number_input(
                "미리보기 데이터 개수",
                min_value=5,
                max_value=500,
                value=20,
                key="preview_count",
                help="로드할 데이터 개수 (최대 500개)",
                label_visibility="collapsed",
            )

        # 데이터 로드 버튼 (전체 너비)
        st.markdown("---")

        load_btn_disabled = not (selected_table and input_column)
        if st.button(
            "📊 데이터 로드 및 미리보기",
            key="load_preview_data",
            type="primary",
            use_container_width=True,
            disabled=load_btn_disabled,
        ):
            if selected_table and input_column:
                with st.spinner("데이터 로드 중..."):
                    preview_data = load_test_data(selected_table, preview_count)
                    st.session_state["preview_data"] = preview_data
                    st.session_state["current_table"] = selected_table
                    st.session_state["current_input_column"] = input_column

        if load_btn_disabled:
            st.caption("💡 테이블과 입력 컬럼을 모두 선택한 후 데이터를 로드하세요.")

        # 데이터가 로드되었으면 선택 인터페이스 표시
        if "preview_data" in st.session_state and st.session_state["preview_data"]:
            preview_data = st.session_state["preview_data"]
            # 현재 설정과 로드된 데이터가 일치하는지 확인
            current_table = st.session_state.get("current_table")
            current_input_column = st.session_state.get("current_input_column")

            if current_table == selected_table and current_input_column == input_column:
                st.success(f"✅ {len(preview_data)}개 데이터 로드 완료")
            else:
                st.warning("⚠️ 설정이 변경되었습니다. 데이터를 다시 로드해주세요.")
                st.session_state.pop("preview_data", None)
                st.stop()

                # 전체 선택/해제 체크박스
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                select_all = st.checkbox("전체 선택", key="select_all")
            with col2:
                if st.button("🔄 선택 초기화", key="clear_selection"):
                    st.session_state["selected_rows"] = []
                    st.rerun()

            # 선택된 행 추적
            if "selected_rows" not in st.session_state:
                st.session_state["selected_rows"] = []

            # 데이터 테이블과 체크박스
            st.markdown(
                "#### 📋 데이터 선택 (체크박스를 클릭하여 테스트할 데이터 선택)"
            )

            # 데이터를 pandas DataFrame으로 변환
            import pandas as pd

            df = pd.DataFrame(preview_data)

            # 체크박스 컬럼 추가
            selected_indices = []

            # 컨테이너로 스크롤 가능한 영역 만들기
            with st.container(height=400):
                for idx, row in enumerate(preview_data):
                    col_check, col_data = st.columns([0.1, 0.9])

                    with col_check:
                        # 전체 선택이 체크되었거나 개별적으로 선택된 경우
                        is_selected = select_all or idx in st.session_state.get(
                            "selected_rows", []
                        )

                        if st.checkbox(
                            "",
                            value=is_selected,
                            key=f"row_check_{idx}",
                            label_visibility="collapsed",
                        ):
                            if idx not in st.session_state["selected_rows"]:
                                st.session_state["selected_rows"].append(idx)
                            selected_indices.append(idx)
                        else:
                            if idx in st.session_state["selected_rows"]:
                                st.session_state["selected_rows"].remove(idx)

                    with col_data:
                        # 행 데이터 표시 (입력 컬럼 강조)
                        if input_column and input_column in row:
                            st.markdown(
                                f"**[행 {idx}]** `{input_column}`: **{row[input_column]}**"
                            )
                            # 다른 컬럼들도 작게 표시
                            other_cols = {
                                k: v for k, v in row.items() if k != input_column
                            }
                            if other_cols:
                                st.caption(f"기타: {other_cols}")
                        else:
                            st.markdown(f"**[행 {idx}]** {row}")
                        st.markdown("---")

            # 선택 요약
            if select_all:
                final_selected = list(range(len(preview_data)))
            else:
                final_selected = st.session_state.get("selected_rows", [])

            st.info(f"🎯 선택된 데이터: {len(final_selected)}개")

            # 선택된 데이터가 있으면 다음 탭으로 이동 안내
            if final_selected:
                st.success("✅ 데이터 선택 완료! '⚙️ 실행 설정' 탭으로 이동하세요.")
                # 선택된 데이터를 세션에 저장
                st.session_state["final_selected_indices"] = final_selected
                st.session_state["selected_table"] = selected_table
                st.session_state["selected_input_column"] = input_column

    with batch_tab2:
        st.caption("⚙️ 배치 테스트 실행 설정")

        # 선택된 데이터 확인
        if (
            "final_selected_indices" not in st.session_state
            or "selected_table" not in st.session_state
            or "selected_input_column" not in st.session_state
        ):
            st.warning("⚠️ 먼저 '📋 데이터 선택' 탭에서 테스트 데이터를 선택해주세요.")
        else:
            selected_count = len(st.session_state["final_selected_indices"])
            table_name = st.session_state["selected_table"]
            input_col = st.session_state["selected_input_column"]

            st.success(
                f"✅ 선택된 데이터: {selected_count}개 (테이블: {table_name}, 컬럼: {input_col})"
            )

            # 실행 설정
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔧 병렬 처리 설정")
                max_workers = st.number_input(
                    "동시 실행 수",
                    min_value=1,
                    max_value=min(20, selected_count),
                    value=min(5, selected_count),
                    key="batch_workers",
                )
                st.caption(f"⚠️ 최대 {min(20, selected_count)}개까지 가능")

            with col2:
                st.markdown("#### 📊 예상 비용")
                if model_name:
                    # 대략적인 토큰 수 추정 (1 문자 ≈ 0.75 토큰으로 가정)
                    avg_input_length = 100  # 기본값
                    if "preview_data" in st.session_state and input_col:
                        lengths = []
                        for idx in st.session_state["final_selected_indices"]:
                            if idx < len(st.session_state["preview_data"]):
                                text = str(
                                    st.session_state["preview_data"][idx].get(
                                        input_col, ""
                                    )
                                )
                                lengths.append(len(text))
                        if lengths:
                            avg_input_length = sum(lengths) / len(lengths)

                    estimated_input_tokens = (
                        int(avg_input_length * 0.75) + len(sys_content.split())
                        if sys_content
                        else 0
                    )
                    estimated_output_tokens = 150  # 추정값

                    single_cost = estimate_cost_usd(
                        model_name, estimated_input_tokens, estimated_output_tokens
                    )
                    if single_cost:
                        total_estimated_cost = single_cost * selected_count
                        st.metric("예상 총 비용", f"${total_estimated_cost:.4f}")
                        st.caption(f"테스트당 약 ${single_cost:.6f}")

            st.markdown("---")

            # 배치 테스트 실행 버튼
            if st.button(
                f"🚀 AI에게 질문하기 ({selected_count}개)",
                type="primary",
                use_container_width=True,
                key="run_batch",
            ):
                if not model_name:
                    st.error("❌ 모델을 선택해주세요.")
                elif not sys_content.strip():
                    st.error("❌ System 프롬프트를 입력해주세요.")
                else:
                    # 선택된 데이터 로드
                    st.info(f"📋 선택된 {selected_count}개 데이터 준비 중...")

                    selected_test_data = load_selected_test_data(
                        table_name,
                        st.session_state["final_selected_indices"],
                        input_col,
                    )

                    if not selected_test_data:
                        st.error("❌ 선택된 데이터를 불러올 수 없습니다.")
                    else:
                        st.success(
                            f"✅ {len(selected_test_data)}개 테스트 데이터 준비 완료"
                        )

                        # 배치 테스트 실행
                        st.info(f"⚡ 배치 테스트 시작... (동시 실행: {max_workers}개)")

                        batch_results = run_batch_test(
                            model_name=model_name,
                            sys_content=sys_content,
                            test_data=selected_test_data,
                            max_workers=max_workers,
                        )

                        # 결과 저장
                        save_batch_results(
                            results=batch_results,
                            model_name=model_name,
                            sys_name=sys_name,
                            sys_version=sys_version,
                            usr_name=usr_name,
                            usr_version=usr_version,
                            tester=tester,
                        )

                        # 세션 상태에 결과 저장 (결과 탭에서 표시)
                        st.session_state["batch_results"] = batch_results

                        st.success(
                            "🎉 배치 테스트 완료! '📊 결과' 탭에서 결과를 확인하세요."
                        )

    with batch_tab3:
        results = st.session_state.get("batch_results", [])

        if results:
            # 결과 요약
            success_count = sum(1 for r in results if r["status"] == "success")
            error_count = len(results) - success_count
            total_cost = sum(r.get("est_cost_usd", 0) for r in results)
            avg_latency = (
                sum(r.get("latency_ms", 0) for r in results) / len(results)
                if results
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("성공", success_count)
            with col2:
                st.metric("실패", error_count)
            with col3:
                st.metric("총 비용", f"${total_cost:.4f}")
            with col4:
                st.metric("평균 지연시간", f"{avg_latency:.0f}ms")

            # 상세 결과 테이블
            st.caption("📊 상세 결과")
            import pandas as pd

            df_results = pd.DataFrame(results)
            st.dataframe(
                df_results,
                column_config={
                    "test_id": "테스트 ID",
                    "status": "상태",
                    "content": st.column_config.TextColumn("응답", width="large"),
                    "input_tokens": st.column_config.NumberColumn("입력 토큰"),
                    "output_tokens": st.column_config.NumberColumn("출력 토큰"),
                    "latency_ms": st.column_config.NumberColumn("지연시간(ms)"),
                    "est_cost_usd": st.column_config.NumberColumn(
                        "비용(USD)", format="$%.6f"
                    ),
                    "error": "에러",
                },
                hide_index=True,
                use_container_width=True,
            )

            # 결과 다운로드
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 결과 CSV 다운로드",
                data=csv,
                file_name=f"batch_test_results_{int(time.time())}.csv",
                mime="text/csv",
            )
        else:
            st.info("📝 배치 테스트를 실행하면 결과가 여기에 표시됩니다.")
