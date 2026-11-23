"""API 없이 도는 기본 점검. 코드를 고친 뒤 커밋 전에 한 번씩:

    python tests/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from rag import config  # noqa: E402
from rag.answer import extract_letter, format_response, validate_output_file  # noqa: E402
from rag.llm import BudgetExceeded, MockChat, QuestionSession  # noqa: E402
from rag.parser import load_questions, parse_question  # noqa: E402

failures = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        failures.append(name)


def test_parser():
    print("[parser]")
    for f in ("test_samples.csv", "test_samples_students.csv"):
        qs = load_questions(ROOT / "data" / f)
        check(f"{f}: 전 문항 파싱", all(q.parse_ok for q in qs))
    qs = load_questions(ROOT / "data" / "test_samples.csv")
    check("ID 자동 부여 row-0001", qs[0].qid == "row-0001")
    check("U+2028 구분자(5번) 4지선다", qs[4].letters == list("ABCD") and qs[4].options["B"] == "서대문구")
    check("16·21번은 A–G", len(qs[15].letters) == 7 and len(qs[20].letters) == 7)
    check("13번 접두어·' ]' 제거", not qs[12].stem.startswith("QUESTION") and not qs[12].options["J"].endswith("]"))
    check("1번 선택지 D 보존", qs[0].options["D"] == "A,B,C 중 답 없음")
    check("도메인 추정", qs[0].domain_guess == "ewha" and qs[13].domain_guess == "mmlu")
    q = parse_question("QUESTION1) 본문에 (A) 표기가 있는 문항\n(A) 가\n(B) 나\n(C) 다")
    check("본문 속 (A)는 선택지로 오인하지 않음", q.letters == list("ABC") and "(A) 표기" in q.stem)


def test_extract():
    print("[answer extraction]")
    L = list("ABCD")
    cases = [
        ("reasoning...\n[ANSWER]: (C)", "C"),
        ("[ANSWER]: (A) ... 다시 보니\n[ANSWER]: (B)", "B"),
        ("**[ANSWER]**: (D)", "D"),
        ("Answer: C", "C"),
        ("정답은 (B)입니다", "B"),
        ("[ANSWER]: (H)", None),          # 선택지 범위 밖
        ("모르겠습니다", None),
    ]
    for text, want in cases:
        got, _ = extract_letter(text, L)
        check(f"{text[:28]!r} → {want}", got == want)
    check("format_response", format_response("E") == "[ANSWER]: (E)")


def test_budget():
    print("[LLM budget]")
    s = QuestionSession(MockChat(), "q1")
    for _ in range(config.MAX_CALLS_PER_QUESTION):
        s.chat("Valid letters: A, B")
    try:
        s.chat("Valid letters: A, B")
        check("6번째 호출 차단", False)
    except BudgetExceeded:
        check("6번째 호출 차단", True)
    s2 = QuestionSession(MockChat(), "q2")
    for bad in ({"max_tokens": 5000}, {"model": "solar-pro4-260806"}, {"n": 3}):
        try:
            s2.chat("x", **bad)
            check(f"금지 파라미터 {bad}", False)
        except ValueError:
            check(f"금지 파라미터 {bad}", True)
    check("금지 파라미터는 호출 수에 안 셈", s2.calls_used == 0)


def test_validator():
    print("[output validator]")
    qs = load_questions(ROOT / "data" / "test_samples.csv")
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.csv"
        pd.DataFrame({"question_id": [q.qid for q in qs],
                      "responses": [format_response(q.letters[0]) for q in qs]}).to_csv(good, index=False)
        check("정상 파일 통과", validate_output_file(good, qs) == [])

        bad = Path(d) / "bad.csv"
        resp = [format_response("A") for _ in qs]
        resp[0] = "[ANSWER]: (A) [ANSWER]: (B)"        # 태그 2개
        resp[15] = "[ANSWER]: (J)"                     # 16번은 A–G
        pd.DataFrame({"question_id": [q.qid for q in qs], "responses": resp}).to_csv(bad)   # 인덱스 컬럼 포함
        probs = validate_output_file(bad, qs)
        check("인덱스 컬럼 검출", any("컬럼" in p for p in probs))
        check("태그 2개 검출", any("태그 2개" in p for p in probs))
        check("선택지 범위 밖 검출", any("범위" in p for p in probs))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    test_parser()
    test_extract()
    test_budget()
    test_validator()
    print(f"\n{'모두 통과' if not failures else f'실패 {len(failures)}건: {failures}'}")
    sys.exit(1 if failures else 0)
