"""도메인 라우팅 RAG: 이화 학칙 문항만 학칙을 검색해 붙이고, 나머지(MMLU 전문 지식)는 컨텍스트 없이 푼다.

Baseline은 MMLU 문항에도 관련 없는 학칙 구절을 붙여 MMLU 정확도가 LLM only보다 떨어졌다 (60.2% → 54.5%).

라우팅 규칙 (LLM 호출 없음, 입력 CSV의 domain 열은 쓰지 않는다: 평가 입력에는 없으므로)
  1) 질문에 한글이 있으면 → 학칙
  2) 영어라도 Ewha라는 단어가 있으면 → 학칙 ("regulation"은 MMLU 법·역사 문항에도 나와 제외)
  3) 아니면 학칙 인덱스 top-1 유사도가 ROUTE_THRESHOLD 이상일 때만 → 학칙
     dev_all 기준 top-1 유사도: 이화 0.552–0.841, MMLU 0.275–0.446 → 0.50으로 갈림

기본 인덱스는 조 단위 청킹(ewha_article). --index로 바꿀 수 있다.
"""
from __future__ import annotations

import re

from .. import prompts
from .base import Prediction, ask_once
from .rag_baseline import RAGBaseline

ROUTE_THRESHOLD = 0.50
DEFAULT_ROUTED_INDEX = "ewha_article"
_HANGUL_RE = re.compile(r"[가-힣]")
_EWHA_KEYWORD_RE = re.compile(r"\bewha\b", re.IGNORECASE)   # "regulation"은 법 문항에도 나와서 뺐다


class RoutedRAG(RAGBaseline):
    name = "routed"

    def setup(self) -> None:
        if not getattr(self.args, "index", None):
            self.args.index = DEFAULT_ROUTED_INDEX
        super().setup()

    def describe(self) -> dict:
        return {**super().describe(), "route_threshold": ROUTE_THRESHOLD}

    def route(self, q, hits) -> tuple[str, str]:
        text = q.render()
        if _HANGUL_RE.search(text):
            return "ewha", "hangul"
        if _EWHA_KEYWORD_RE.search(text):
            return "ewha", "keyword"
        if hits and hits[0].score >= ROUTE_THRESHOLD:
            return "ewha", f"score {hits[0].score:.3f}"
        return "general", f"score {hits[0].score:.3f}" if hits else "no hits"

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())          # 임베딩·검색은 LLM 호출 수에 들어가지 않는다
        route, why = self.route(q, hits)
        if route == "ewha":
            pred = ask_once(q, llm, prompts.SYSTEM_RAG, prompts.user_rag(q, hits))
            pred.info["retrieved"] = [h.brief() for h in hits]
        else:
            pred = ask_once(q, llm, prompts.SYSTEM_LLM_ONLY, prompts.user_llm_only(q))
        pred.info["route"] = route
        pred.info["route_reason"] = why
        return pred
