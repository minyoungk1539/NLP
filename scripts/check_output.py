"""출력 CSV 검증기 (제출 전 필수).

    python scripts/check_output.py responses.csv --input data/test_samples.csv

검사 항목
- UTF-8, 컬럼이 정확히 question_id,responses (인덱스 컬럼 없음)
- 행 수 = 입력 문항 수, question_id 순서 = 입력 순서
- 각 응답에 eval.py 정규식과 같은 [ANSWER]: (X) 태그가 정확히 1개, X가 그 문항의 선택지 범위 안
입력에 answers 컬럼이 있으면 eval.py와 같은 방식(행 위치)으로 정확도와 도메인별 정확도도 출력.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rag.answer import EVAL_RE, validate_output_file  # noqa: E402
from rag.parser import load_questions  # noqa: E402
from rag.scoring import format_summary, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses")
    ap.add_argument("--input", required=True, help="run.py에 넣었던 입력 CSV")
    ap.add_argument("--show-wrong", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = load_questions(a.input)
    problems = validate_output_file(a.responses, questions)
    if problems:
        print(f"✗ 형식 검증 실패 ({len(problems)}건)")
        for p in problems[:50]:
            print("  -", p)
    else:
        print(f"✓ 형식 검증 통과 ({len(questions)}행)")

    if any(q.gold for q in questions):
        df = pd.read_csv(a.responses, dtype=str, keep_default_na=False, encoding="utf-8")
        rows = []
        for q, resp in zip(questions, df.get("responses", [])):
            m = EVAL_RE.search(resp)                     # eval.py처럼 첫 매치
            pred = m.group(1) if m else None
            rows.append({"question_id": q.qid, "domain": q.domain_guess, "qtype": q.meta.get("qtype") or None,
                         "gold": q.gold, "pred": pred})
        print(format_summary(summarize(rows)))
        if a.show_wrong:
            for r, q in zip(rows, questions):
                if r["gold"] and r["pred"] != r["gold"]:
                    print(f"  ✗ {r['question_id']} [{r['domain']}] 정답 {r['gold']} / 예측 {r['pred']}: "
                          f"{q.stem[:70].replace(chr(10), ' ')}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
