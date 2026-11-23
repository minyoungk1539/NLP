"""문항 파서: CSV의 prompts 텍스트 → 질문 본문(stem) + 선택지(dict).

연습 파일에서 확인한 형식 차이를 모두 처리한다.
- 선택지 구분자가 줄바꿈 대신 U+2028(LINE SEPARATOR)인 문항 (5, 16, 20번)
- "QUESTION13) QUESTION 6) ..." 처럼 번호 접두어가 겹친 문항, 끝에 붙은 " ]"
- 선택지 앞 공백, 빈 줄
- 선택지 개수가 문항마다 다름 (4개, 7개, 10개) → letters를 문항마다 계산
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ALL_LETTERS = "ABCDEFGHIJ"

_QPREFIX_RE = re.compile(r"^\s*(?:QUESTION\s*\d+\s*\)\s*)+", re.IGNORECASE)
_LINE_MARKER_RE = re.compile(r"(?:^|(?<=\n))[ \t]*\(([A-J])\)[ \t]*")
_INLINE_MARKER_RE = re.compile(r"(?:^|(?<=\s))\(([A-J])\)[ \t]*")
_HANGUL_RE = re.compile(r"[가-힣]")


@dataclass
class Question:
    qid: str
    row: int                       # 입력 파일에서의 0-based 위치 (eval.py는 위치로 채점)
    raw: str
    stem: str
    options: dict[str, str]
    parse_ok: bool = True
    meta: dict = field(default_factory=dict)   # 입력 CSV의 나머지 컬럼 (answers, domain, gold_ref ...)

    @property
    def letters(self) -> list[str]:
        return list(self.options) if self.options else list(ALL_LETTERS)

    @property
    def lang(self) -> str:
        return "ko" if _HANGUL_RE.search(self.stem) else "en"

    @property
    def domain_guess(self) -> str:
        """한글이 있으면 'ewha', 아니면 'mmlu'. 입력에 domain 컬럼이 있으면 그것을 우선 사용."""
        return self.meta.get("domain") or ("ewha" if self.lang == "ko" else "mmlu")

    @property
    def gold(self) -> str | None:
        m = re.search(r"([A-J])", str(self.meta.get("answers", "")))
        return m.group(1) if m else None

    def render(self) -> str:
        """LLM에 보낼 정리된 문항 텍스트."""
        if not self.options:
            return self.stem
        opts = "\n".join(f"({k}) {v}" for k, v in self.options.items())
        return f"{self.stem}\n{opts}"


def normalize_text(text: str) -> str:
    text = str(text)
    for a, b in (("\r\n", "\n"), ("\r", "\n"), (" ", "\n"), (" ", "\n"), ("\xa0", " ")):
        text = text.replace(a, b)
    return text


def _best_chain(markers: list[tuple[str, int, int]]) -> list[tuple[str, int, int]] | None:
    """(A)(B)(C)... 순서로 이어지는 가장 긴 마커 사슬. 같은 길이면 뒤쪽(A)를 선호."""
    best = None
    for i, m in enumerate(markers):
        if m[0] != "A":
            continue
        chain, expect = [m], "B"
        for n in markers[i + 1:]:
            if n[0] == expect:
                chain.append(n)
                expect = chr(ord(expect) + 1)
        if len(chain) >= 2 and (best is None or len(chain) >= len(best)):
            best = chain
    return best


def _clean_option(text: str) -> str:
    t = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if t.endswith("]") and "[" not in t:                    # 13번 끝의 " ]"
        t = t[:-1].rstrip()
    if len(t) >= 2 and t[0] == t[-1] == '"' and t.count('"') == 2:
        t = t[1:-1].strip()
    return t


def parse_question(raw: str, qid: str = "", row: int = 0, meta: dict | None = None) -> Question:
    text = normalize_text(raw)
    text = _QPREFIX_RE.sub("", text, count=1).strip()

    chain = None
    for regex in (_LINE_MARKER_RE, _INLINE_MARKER_RE):
        markers = [(m.group(1), m.start(), m.end()) for m in regex.finditer(text)]
        chain = _best_chain(markers)
        if chain:
            break

    if not chain:
        return Question(qid, row, raw, text.strip(), {}, parse_ok=False, meta=meta or {})

    stem = text[: chain[0][1]].strip()
    options: dict[str, str] = {}
    for i, (letter, _start, end) in enumerate(chain):
        stop = chain[i + 1][1] if i + 1 < len(chain) else len(text)
        options[letter] = _clean_option(text[end:stop])
    return Question(qid, row, raw, stem, options, parse_ok=True, meta=meta or {})


PROMPT_COLUMNS = ("prompts", "prompt", "question", "questions")


def load_questions(path: str | Path) -> list[Question]:
    """입력 CSV → Question 목록 (입력 순서 유지).

    question_id 컬럼이 있으면 그대로 쓰고, 없으면 row-0001 형식으로 부여한다.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    col = next((c for c in PROMPT_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path}: 문항 컬럼을 찾을 수 없음 (기대: {PROMPT_COLUMNS}, 실제: {list(df.columns)})")
    has_id = "question_id" in df.columns and all(str(x).strip() for x in df["question_id"])

    questions = []
    for i, rec in enumerate(df.to_dict("records")):
        qid = str(rec["question_id"]).strip() if has_id else f"row-{i + 1:04d}"
        meta = {k: v for k, v in rec.items() if k not in (col, "question_id")}
        questions.append(parse_question(rec[col], qid=qid, row=i, meta=meta))
    return questions
