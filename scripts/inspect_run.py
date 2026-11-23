"""실행 결과 분석 (실패 사례 분석·발표 자료용).

    python scripts/inspect_run.py                    # 가장 최근 run의 오답 목록
    python scripts/inspect_run.py runs/<폴더> --domain ewha --raw
    python scripts/inspect_run.py --compare runs/A runs/B     # 두 run에서 결과가 갈린 문항

출력에서 '근거쪽 미검색'이면 Missing source/Wrong passage 쪽, '근거쪽 검색됨인데 오답'이면
Missing condition/Incorrect application 쪽 실패일 가능성이 높다 (슬라이드의 실패 유형 4가지).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402


def load(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_run() -> Path:
    runs = sorted(p for p in config.RUNS_DIR.iterdir() if (p / "results.jsonl").exists())
    if not runs:
        sys.exit("runs/ 에 실행 기록이 없습니다")
    return runs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--domain")
    ap.add_argument("--all", action="store_true", help="정답 문항도 표시")
    ap.add_argument("--raw", action="store_true", help="모델 원출력 표시")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.compare:
        ra, rb = (load(Path(p)) for p in a.compare)
        mb = {r["question_id"]: r for r in rb}
        print(f"A={a.compare[0]}\nB={a.compare[1]}\n")
        for r in ra:
            s = mb.get(r["question_id"])
            if s and r["correct"] != s["correct"]:
                tag = "B만 정답" if s["correct"] else "A만 정답"
                print(f"{tag:<7} {r['question_id']:<10} [{r['domain']}] 정답 {r['gold']} A={r['pred']} B={s['pred']}"
                      f"  {r['question'][:60].replace(chr(10), ' ')}")
        return

    run_dir = Path(a.run) if a.run else latest_run()
    rows = load(run_dir)
    print(f"run: {run_dir}\n")
    for r in rows:
        if a.domain and r["domain"] != a.domain:
            continue
        if not a.all and r["correct"] is not False:
            continue
        mark = "✓" if r["correct"] else "✗"
        print(f"{mark} {r['question_id']} [{r['domain']}/{r.get('qtype') or '-'}] 정답 {r['gold']} / 예측 {r['pred']}"
              f"  calls={r['calls']} extract={r['extract_method']}{' FALLBACK' if r['fallback'] else ''}")
        print("   Q:", r["question"].split("\n")[0][:150])
        retrieved = (r.get("info") or {}).get("retrieved")
        if retrieved:
            pages = ", ".join(f"p.{x['pages'][0]}-{x['pages'][1]}({x['score']:.2f})" for x in retrieved)
            hit = r.get("gold_page_hit")
            hit_s = "" if hit is None else ("  → 근거쪽 검색됨" if hit else f"  → 근거쪽 {r['gold_page']} 미검색")
            print(f"   검색: {pages}{hit_s}")
        if r.get("error"):
            print("   오류:", r["error"].splitlines()[0])
        if a.raw:
            print("   출력:", (r.get("raw_output") or "").strip().replace("\n", "\n         ")[:1500])
        print()


if __name__ == "__main__":
    main()
