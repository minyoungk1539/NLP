"""Baseline RAG (기준선 2): 모든 문항에 대해 학칙 인덱스에서 dense top-k 검색 → 컨텍스트로 넣고 1회 호출.

일부러 단순하게 둔 것 (개선 실험 몫): 도메인 라우팅 없음(MMLU 문항에도 학칙을 붙임),
BM25/하이브리드 없음, 재순위화 없음, 쿼리 = 문항 전체(본문+선택지).
"""
from __future__ import annotations

from .. import config, prompts
from ..embeddings import make_embedder
from ..index import load_or_build
from .base import Pipeline, Prediction, ask_once


class RAGBaseline(Pipeline):
    name = "rag_baseline"

    def setup(self) -> None:
        mock = getattr(self.args, "mock", False)
        self.top_k = getattr(self.args, "top_k", None) or config.DEFAULT_TOP_K
        name = getattr(self.args, "index", None) or config.DEFAULT_INDEX
        if mock and not name.startswith("mock_"):
            name = "mock_" + name                      # mock 임베딩 인덱스는 실제 인덱스와 섞이지 않게
        self.index_name = name
        self.embedder = make_embedder(mock)
        self.index = load_or_build(name, self.embedder)

    def describe(self) -> dict:
        return {"index": self.index_name, "top_k": self.top_k, "index_meta": self.index.meta}

    def retrieve(self, query: str):
        qvec = self.embedder.embed_queries([query])[0]
        return self.index.search(qvec, self.top_k)

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())
        pred = ask_once(q, llm, prompts.SYSTEM_RAG, prompts.user_rag(q, hits))
        pred.info["retrieved"] = [h.brief() for h in hits]
        return pred
