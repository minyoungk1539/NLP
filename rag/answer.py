"""답 추출 · 응답 포맷 · 출력 CSV 검증기.

eval.py는 r"\\[ANSWER\\]:\\s*\\(([A-J])\\)" 의 '첫 번째' 매치만 읽고, 행 위치로 채점한다.
그래서 우리는 응답을 "[ANSWER]: (X)" 한 줄로만 내보낸다 (설명은 runs/ 로그에 남김).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .parser import Question

EVAL_RE = re.compile(r"\[ANSWER\]:\s*\(([A-J])\)")       # eval.py와 동일

# 모델이 형식을 어겼을 때 쓰는 관대한 패턴 (앞에 있을수록 우선)
_LENIENT = [
    ("tag_loose", re.compile(r"(?i:answer)\W{0,6}\(?\s*([A-J])\s*\)?(?![A-Za-z])")),
    ("korean", re.compile(r"정답\s*(?:은|는)?\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z])")),
    ("answer_is", re.compile(r"(?i:answer\s+is)\s*:?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z])")),
]
_LAST_LINE_RE = re.compile(r"^\W*\(?([A-J])\)?\W*$")
_LAST_LINE_OPTION_RE = re.compile(r"^\W{0,4}\(([A-J])\)\s+\S")


def extract_letter(text: str | None, letters: list[str]) -> tuple[str | None, str]:
    """모델 출력에서 최종 답 글자를 뽑는다. (letter, 방법). 실패하면 (None, 'none')."""
    if not text:
        return None, "empty"
    valid = set(letters)
    strict = [m for m in EVAL_RE.findall(text) if m in valid]
    if strict:
        return strict[-1], "strict"
    for name, rx in _LENIENT:
        found = [m for m in rx.findall(text) if m in valid]
        if found:
            return found[-1], name
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = _LAST_LINE_RE.match(lines[-1])
        if m and m.group(1) in valid:
            return m.group(1), "last_line"
        m = _LAST_LINE_OPTION_RE.match(lines[-1])          # 마지막 줄이 "(B) Great Hero"처럼 선택지로 시작
        if m and m.group(1) in valid:
            return m.group(1), "last_line_option"
    return None, "none"


def format_response(letter: str) -> str:
    return f"[ANSWER]: ({letter})"


def fallback_letter(q: Question) -> str:
    """추출·호출 실패 시에도 행이 비지 않도록 넣는 기본 답 (로그에 fallback=True로 표시됨)."""
    return q.letters[0]


def check_response(resp, letters: list[str]) -> list[str]:
    problems = []
    if not isinstance(resp, str) or not resp.strip():
        return ["빈 응답"]
    tags = EVAL_RE.findall(resp)
    if len(tags) != 1:
        problems.append(f"[ANSWER] 태그 {len(tags)}개 (정확히 1개여야 함)")
    if tags and tags[0] not in letters:
        problems.append(f"답 {tags[0]}가 선택지 범위({letters[0]}–{letters[-1]}) 밖")
    return problems


def validate_output_file(path: str | Path, questions: list[Question]) -> list[str]:
    """제출 형식 검증. 문제 목록을 돌려준다 (빈 리스트면 통과)."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        return ["UTF-8로 읽을 수 없음"]
    problems = []
    if list(df.columns) != ["question_id", "responses"]:
        problems.append(f"컬럼이 정확히 [question_id, responses]가 아님: {list(df.columns)}")
    if len(df) != len(questions):
        problems.append(f"행 수 {len(df)} ≠ 입력 문항 수 {len(questions)}")
    if "question_id" in df.columns:
        expected = [q.qid for q in questions]
        got = list(df["question_id"])
        bad = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        if bad:
            problems.append(f"question_id 순서 불일치 {len(bad)}건 (첫 위치 {bad[0] + 1}: {got[bad[0]]} vs {expected[bad[0]]})")
    if "responses" in df.columns:
        for i, (resp, q) in enumerate(zip(df["responses"], questions)):
            for p in check_response(resp, q.letters):
                problems.append(f"{i + 1}행 ({q.qid}): {p}")
    return problems
