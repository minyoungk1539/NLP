"""ewha_prompt_v2: ewha_prompt 수정판.

ewha_prompt(20261127-000720)에서 학칙 지시 ①(조항을 그대로 인용)이 반복 인용(S-004, 4096토큰 소진)과
깨진 글자 복사(S-003, \\uf000…)를 일으켜 답이 안 나온 문제를 고친다.

ewha_prompt 대비 바뀐 점
  ① 학칙 지시 1번: 가장 관련 있는 한두 문장만 짧게, 한 번씩만 인용하고 깨진 글자는 무시 (단계별 계산·반드시 하나 고르기는 유지)
  ② 학칙 경로: 검색 결과를 프롬프트에 넣기 전에 사용자 정의 영역(PUA) 문자를 지운다 (pred.info["retrieved"]는 그대로)
  ③ 두 경로 모두: 답 글자를 못 뽑으면 두 번째 호출로 "[ANSWER]: (글자) 한 줄만" 다시 묻는다 → pred.info["retry"]
  ④ OUTPUT_RULE의 '(X)' 자리표시를 '(letter)'로 (이 파이프라인의 두 시스템 프롬프트에만 적용, prompts.py는 그대로)
그래서 전문지식 경로는 더 이상 gen_prompt와 완전히 같지 않다 (④의 출력 규칙 문구, ③의 재시도).
"""
from __future__ import annotations

import dataclasses
import re

from .. import prompts
from ..answer import extract_letter
from .base import Prediction
from .ewha_prompt import EwhaPrompt
from .mmlu_exp import SYSTEM_GENERAL, _user

OUTPUT_RULE_V2 = (
    "Finish your reply with exactly one final line of the form [ANSWER]: (letter), "
    "where letter is one of the valid option letters. Do not write the [ANSWER] tag anywhere else."
)

SYSTEM_EWHA_V2 = (
    "You are a careful expert on the Ewha Womans University regulations (이화여자대학교 학칙), "
    "answering a multiple-choice question that may be written in Korean or English. "
    "You are given passages retrieved from the regulations.\n"
    "1. First, quote only the one or two most relevant sentences from the passages, briefly, with the "
    "article number (e.g. 제31조제2항). Quote each sentence only once, and ignore any garbled or broken "
    "characters. If no passage is relevant, say so and rely on your own knowledge.\n"
    "2. If the question needs numbers (credits, years, semesters, counts, sums), compute step by step: "
    "write each value with the article it comes from, then show the arithmetic.\n"
    "3. Check each option against the quoted text, then choose.\n"
    "Exactly one option is correct. Even if the evidence is insufficient, you must pick the single most "
    "likely option; never answer 'none', 'X' or leave it blank. (If an option itself states that none of "
    "the other options is correct, that option is a valid choice.)\n"
    + OUTPUT_RULE_V2
)

assert prompts.OUTPUT_RULE in SYSTEM_GENERAL
SYSTEM_GENERAL_V2 = SYSTEM_GENERAL.replace(prompts.OUTPUT_RULE, OUTPUT_RULE_V2)

# 사용자 정의 영역: BMP(U+E000–U+F8FF) + 보조 평면 15·16
_PUA_RE = re.compile("[-\U000F0000-\U0010FFFF]")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n(?:[ \t]*\n)+")


def strip_pua(text: str) -> str:
    """PUA 문자를 지우고, 지우고 남은 연속 공백·빈 줄을 줄인다 (PUA가 없는 청크는 그대로)."""
    if not _PUA_RE.search(text):
        return text
    text = _SPACES_RE.sub(" ", _PUA_RE.sub("", text))
    return _BLANK_LINES_RE.sub("\n", text)


def clean_hits(hits):
    """원본 청크(인덱스 공유 객체)는 건드리지 않고 text만 정리한 사본을 만든다."""
    return [dataclasses.replace(h, chunk=dataclasses.replace(h.chunk, text=strip_pua(h.chunk.text)))
            for h in hits]


def retry_instruction(q) -> str:
    return ("Based on your reasoning above, reply with only one line: [ANSWER]: (letter), "
            f"where letter is one of: {', '.join(q.letters)}.")


class EwhaPromptV2(EwhaPrompt):
    name = "ewha_prompt_v2"
    system = SYSTEM_GENERAL_V2              # GeneralExperiment.solve_general이 쓰는 시스템 프롬프트
    ewha_params: dict = {"max_tokens": 4096}
    retry_params: dict = {"max_tokens": 256}

    def describe(self) -> dict:
        return {**super().describe(), "retry_params": self.retry_params}

    def ask_with_retry(self, q, llm, messages: list[dict], tag: str, params: dict) -> Prediction:
        out = llm.chat(messages, tag=tag, **params)
        letter, method = extract_letter(out, q.letters)
        info = {"retry": False}
        if letter is not None or llm.remaining <= 0:
            return Prediction(letter, out, method, info=info)

        # 답 글자를 못 뽑음 → 앞 풀이를 이어 붙이고 답 한 줄만 다시 묻는다.
        # 첫 출력이 비었으면 이어 붙일 풀이가 없으므로 같은 요청을 한 번 더 보낸다.
        if out and out.strip():
            retry_msgs = messages + [{"role": "assistant", "content": out},
                                     {"role": "user", "content": retry_instruction(q)}]
        else:
            retry_msgs = messages
        out2 = llm.chat(retry_msgs, tag=f"{tag}_retry", **{**params, **self.retry_params})
        letter2, method2 = extract_letter(out2, q.letters)
        info.update(retry=True, first_extract_method=method, retry_extract_method=method2,
                    retry_letter=letter2)
        return Prediction(letter2, f"{out}\n\n--- retry ---\n{out2}", f"retry_{method2}", info=info)

    def solve_general(self, q, llm) -> Prediction:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": _user(q)}]
        return self.ask_with_retry(q, llm, messages, "general", self.gen_params)

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())
        route, why = self.route(q, hits)
        if route == "ewha":
            messages = [{"role": "system", "content": SYSTEM_EWHA_V2},
                        {"role": "user", "content": prompts.user_rag(q, clean_hits(hits))}]
            pred = self.ask_with_retry(q, llm, messages, "answer", self.ewha_params)
            pred.info["retrieved"] = [h.brief() for h in hits]
        else:
            pred = self.solve_general(q, llm)
        pred.info.update(route=route, route_reason=why)
        return pred
