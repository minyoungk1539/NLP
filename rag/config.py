"""과제 규칙 상수와 경로. 규칙에 묶인 값은 여기서만 바꾼다."""
from __future__ import annotations

import os
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # Project/
DATA_DIR = ROOT / "data"
DEV_DIR = DATA_DIR / "dev"
INDEX_DIR = ROOT / "index"
RUNS_DIR = ROOT / "runs"
EWHA_PDF = DATA_DIR / "ewha.pdf"

# ── Upstage API (과제 규칙) ───────────────────────────────────────────
BASE_URL = "https://api.upstage.ai/v1"
CHAT_MODEL = "solar-mini4-260922"          # 규칙: 이 모델만 허용 (노트북의 solar-pro4 사용 금지)
EMBED_PASSAGE_MODEL = "solar-embedding-2-passage"
EMBED_QUERY_MODEL = "solar-embedding-2-query"

MAX_CALLS_PER_QUESTION = 5                 # 규칙: 문항당 LLM 호출 최대 5회 (임베딩·검색 제외)
MAX_TOKENS_LIMIT = 4096                    # 규칙: max_tokens ≤ 4096 (reasoning 토큰 포함)

# ── 기본 생성 파라미터 (run.py 인자로 덮어쓸 수 있음) ──────────────────
DEFAULT_GEN_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 1024,
    "reasoning_effort": "none",            # 모델이 거부하면 llm.py가 자동으로 빼고 재시도
}

# ── 기본 인덱스 (rag_baseline) ───────────────────────────────────────
DEFAULT_INDEX = "ewha_fixed800"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


def get_api_key() -> str:
    """Project/.env 또는 환경변수에서 UPSTAGE_API_KEY를 읽는다."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("UPSTAGE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "UPSTAGE_API_KEY가 없습니다. Project/.env 파일을 만들고 "
            "UPSTAGE_API_KEY=... 를 넣으세요 (.env.example 참고). "
            "API 없이 배관만 시험하려면 --mock 으로 실행하세요."
        )
    return key
