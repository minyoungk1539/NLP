"""LLM only (기준선 1): 검색 없이 문항만 주고 1회 호출."""
from __future__ import annotations

from .. import prompts
from .base import Pipeline, Prediction, ask_once


class LLMOnly(Pipeline):
    name = "llm_only"

    def predict(self, q, llm) -> Prediction:
        return ask_once(q, llm, prompts.SYSTEM_LLM_ONLY, prompts.user_llm_only(q))
