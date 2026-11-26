"""ewha_prompt: gen_prompt에서 학칙(이화) 경로만 바꾼 변형.

- 전문지식(MMLU) 경로: GenPrompt와 동일 (SYSTEM_GENERAL, max_tokens 4096)
- 학칙 경로: 라우팅·검색·컨텍스트는 routed와 같고, 시스템 프롬프트와 max_tokens만 바꾼다
    ① 관련 조항을 먼저 인용  ② 숫자 계산은 단계별로  ③ 근거가 부족해도 반드시 보기 중 하나 선택 (X·빈 답 금지)
    max_tokens 4096
"""
from __future__ import annotations

from .. import prompts
from .base import Prediction, ask_once
from .mmlu_exp import GenPrompt

SYSTEM_EWHA = (
    "You are a careful expert on the Ewha Womans University regulations (이화여자대학교 학칙), "
    "answering a multiple-choice question that may be written in Korean or English. "
    "You are given passages retrieved from the regulations.\n"
    "1. First, quote the relevant article(s) from the passages word for word, with the article number "
    "(e.g. 제31조제2항). If no passage is relevant, say so and rely on your own knowledge.\n"
    "2. If the question needs numbers (credits, years, semesters, counts, sums), compute step by step: "
    "write each value with the article it comes from, then show the arithmetic.\n"
    "3. Check each option against the quoted text, then choose.\n"
    "Exactly one option is correct. Even if the evidence is insufficient, you must pick the single most "
    "likely option; never answer 'none', 'X' or leave it blank. (If an option itself states that none of "
    "the other options is correct, that option is a valid choice.)\n"
    + prompts.OUTPUT_RULE
)


class EwhaPrompt(GenPrompt):
    name = "ewha_prompt"
    ewha_params: dict = {"max_tokens": 4096}

    def describe(self) -> dict:
        return {**super().describe(), "ewha_params": self.ewha_params}

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())
        route, why = self.route(q, hits)
        if route == "ewha":
            pred = ask_once(q, llm, SYSTEM_EWHA, prompts.user_rag(q, hits), **self.ewha_params)
            pred.info["retrieved"] = [h.brief() for h in hits]
        else:                                   # GeneralExperiment.predict의 general 경로와 동일
            pred = self.solve_general(q, llm)
        pred.info.update(route=route, route_reason=why)
        return pred
