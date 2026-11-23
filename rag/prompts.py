"""프롬프트 템플릿. LLM only와 Baseline RAG는 '컨텍스트 유무'만 다르게 유지한다 (공정한 비교)."""
from __future__ import annotations

from .parser import Question

OUTPUT_RULE = (
    "Finish your reply with exactly one final line of the form [ANSWER]: (X), "
    "where X is one of the valid option letters. Do not write the [ANSWER] tag anywhere else."
)

SYSTEM_LLM_ONLY = (
    "You are a careful expert who answers multiple-choice questions. "
    "Questions may be written in Korean or English. Reason briefly, then give your final choice.\n"
    + OUTPUT_RULE
)

SYSTEM_RAG = (
    "You are a careful expert who answers multiple-choice questions. "
    "Questions may be written in Korean or English. "
    "You are given passages retrieved from the Ewha Womans University regulations (이화여자대학교 학칙). "
    "Use them when they are relevant to the question; if they are not relevant, rely on your own knowledge. "
    "Reason briefly, then give your final choice.\n"
    + OUTPUT_RULE
)


def valid_letters_line(q: Question) -> str:
    return "Valid letters: " + ", ".join(q.letters)


def user_llm_only(q: Question) -> str:
    return f"Question:\n{q.render()}\n\n{valid_letters_line(q)}"


def format_context(hits) -> str:
    parts = []
    for h in hits:
        c = h.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
        parts.append(f"[{h.rank}] (학칙 {pages})\n{c.text}")
    return "\n\n".join(parts)


def user_rag(q: Question, hits) -> str:
    return f"Context:\n{format_context(hits)}\n\nQuestion:\n{q.render()}\n\n{valid_letters_line(q)}"
