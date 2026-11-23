"""공식 실행 진입점.

    python run.py --input data/test_samples.csv --output responses.csv
    python eval.py responses.csv data/test_samples.csv

팀 실험용 옵션
    --pipeline llm_only|rag_baseline|...   (기본값 DEFAULT_PIPELINE = 제출할 최종 방법으로 바꿔둘 것)
    --mock            API 없이 가짜 LLM·임베딩으로 배관만 시험 (비용 0)
    --limit N / --ids a,b   일부 문항만 (이때 eval.py 점수는 의미 없음, 아래 자체 집계를 볼 것)
    --workers N       동시 처리 문항 수 (출력 순서는 항상 입력 순서)
    --run-name 메모    runs/ 폴더 이름과 leaderboard에 남길 메모

실행마다 runs/<시각>_<파이프라인>/ 에 config.json, results.jsonl(문항별), calls.jsonl(호출별),
summary.json이 저장되고 runs/leaderboard.csv에 한 줄이 추가된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import config  # noqa: E402
from rag.answer import fallback_letter, format_response, validate_output_file  # noqa: E402
from rag.llm import BudgetExceeded, QuestionSession, make_backend  # noqa: E402
from rag.parser import load_questions  # noqa: E402
from rag.pipelines import PIPELINES, Prediction  # noqa: E402
from rag.scoring import format_summary, parse_pages, summarize  # noqa: E402

DEFAULT_PIPELINE = "rag_baseline"      # ← 최종 제출 전에 팀의 최종 파이프라인 이름으로 바꾸기

for stream in (sys.stdout, sys.stderr):          # Windows 콘솔(cp949)에서 한글·기호 깨짐 방지
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAG multiple-choice QA runner")
    p.add_argument("--input", default="data/test_samples.csv")
    p.add_argument("--output", default="responses.csv")
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE, choices=sorted(PIPELINES))
    p.add_argument("--index", default=None, help=f"인덱스 이름 (기본 {config.DEFAULT_INDEX})")
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None, help="쉼표로 구분한 question_id만 실행")
    p.add_argument("--mock", action="store_true", help="가짜 LLM·임베딩 (API 키 불필요)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--reasoning-effort", default=None, help='예: none, low, medium, high / "omit"이면 안 보냄')
    p.add_argument("--run-name", default="")
    p.add_argument("--no-log", action="store_true", help="runs/에 로그를 남기지 않음")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def gen_params(args) -> dict:
    params = dict(config.DEFAULT_GEN_PARAMS)
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        params["reasoning_effort"] = None if args.reasoning_effort == "omit" else args.reasoning_effort
    if params.get("max_tokens", 0) > config.MAX_TOKENS_LIMIT:
        raise SystemExit(f"max_tokens는 {config.MAX_TOKENS_LIMIT} 이하여야 합니다 (규칙).")
    return params


def solve_one(q, pipeline, backend, params) -> tuple[dict, list[dict]]:
    session = QuestionSession(backend, q.qid, defaults=params)
    t0 = time.perf_counter()
    error = None
    try:
        pred = pipeline.predict(q, session)
    except BudgetExceeded as e:
        pred, error = Prediction(None), f"BudgetExceeded: {e}"
    except Exception as e:  # 한 문항 실패가 전체 실행을 멈추지 않도록
        pred, error = Prediction(None), f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    letter = pred.letter if pred.letter in q.letters else None
    fallback = letter is None
    if fallback:
        letter = fallback_letter(q)

    gold_pages = parse_pages(q.meta.get("gold_page"))
    retrieved = pred.info.get("retrieved")
    hit = None
    if gold_pages and retrieved is not None:
        got = {pg for r in retrieved for pg in range(r["pages"][0], r["pages"][1] + 1)}
        hit = bool(got & gold_pages)

    result = {
        "question_id": q.qid,
        "row": q.row,
        "domain": q.domain_guess,
        "qtype": q.meta.get("qtype") or None,
        "gold": q.gold,
        "pred": letter,
        "correct": (letter == q.gold) if q.gold else None,
        "fallback": fallback,
        "extract_method": pred.extract_method,
        "calls": session.calls_used,
        "latency_s": round(time.perf_counter() - t0, 2),
        "parse_ok": q.parse_ok,
        "n_options": len(q.letters),
        "gold_page": sorted(gold_pages) or None,
        "gold_page_hit": hit,
        "error": error,
        "info": pred.info,
        "question": q.render(),
        "raw_output": pred.raw_output,
    }
    return result, [r.to_dict() for r in session.records]


def append_leaderboard(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    questions = load_questions(args.input)
    subset = False
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        questions = [q for q in questions if q.qid in wanted]
        subset = True
    if args.limit:
        questions = questions[: args.limit]
        subset = True
    bad_parse = [q.qid for q in questions if not q.parse_ok]
    if bad_parse:
        print(f"⚠ 선택지 파싱 실패 {len(bad_parse)}건: {bad_parse[:10]} (원문 그대로 보내고 A–J 허용)")

    params = gen_params(args)
    pipeline = PIPELINES[args.pipeline](args)
    pipeline.setup()
    backend = make_backend(args.mock)

    print(f"▶ {args.pipeline} | 문항 {len(questions)}개 | workers={args.workers}"
          f"{' | MOCK' if args.mock else ''} | params={params}")

    results: list[dict | None] = [None] * len(questions)
    calls: list[list[dict]] = [[] for _ in questions]
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(solve_one, q, pipeline, backend, params): i for i, q in enumerate(questions)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i], calls[i] = fut.result()
            done += 1
            if not args.quiet:
                r = results[i]
                mark = "" if r["gold"] is None else (" ✓" if r["correct"] else f" ✗ (정답 {r['gold']})")
                warn = " [fallback]" if r["fallback"] else ""
                print(f"[{done:>3}/{len(questions)}] {r['question_id']:<10} {r['domain']:<5} "
                      f"pred={r['pred']}{mark} calls={r['calls']}{warn}")

    # ── 제출 형식으로 저장 (입력 순서 유지, 인덱스 컬럼 없음) ──
    out = pd.DataFrame({
        "question_id": [r["question_id"] for r in results],
        "responses": [format_response(r["pred"]) for r in results],
    })
    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    problems = validate_output_file(out_path, questions)
    summary = summarize(results)
    summary["format_ok"] = not problems

    print("\n" + "─" * 60)
    print(f"저장: {out_path}  ({len(out)}행)")
    print("형식 검증: 통과" if not problems else "형식 검증: 실패\n  " + "\n  ".join(problems[:20]))
    print(format_summary(summary))
    if subset:
        print("※ 일부 문항만 실행했으므로 eval.py 점수는 의미 없음 (위 집계를 볼 것)")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"⚠ 오류 {len(errors)}건, 첫 오류 ({errors[0]['question_id']}): {errors[0]['error'].splitlines()[0]}")

    # ── 실행 로그 ──
    if not args.no_log:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = "_".join(x for x in (stamp, args.pipeline, args.run_name.replace(" ", "-"), "mock" if args.mock else "") if x)
        run_dir = config.RUNS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"argv": sys.argv, "args": vars(args), "gen_params": params, "chat_model": config.CHAT_MODEL,
               "pipeline": pipeline.describe(), "n_questions": len(questions), "subset": subset}
        (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        with open(run_dir / "calls.jsonl", "w", encoding="utf-8") as f:
            for recs in calls:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        out.to_csv(run_dir / "responses.csv", index=False, encoding="utf-8")

        bd = summary.get("by_domain", {})
        append_leaderboard(config.RUNS_DIR / "leaderboard.csv", {
            "run": tag,
            "pipeline": args.pipeline,
            "input": Path(args.input).name,
            "n": len(questions),
            "acc": summary["overall"]["acc"],
            "acc_ewha": bd.get("ewha", {}).get("acc"),
            "acc_mmlu": bd.get("mmlu", {}).get("acc"),
            "retr_hit": summary.get("retrieval_gold_page_hit", {}).get("rate"),
            "avg_calls": summary.get("calls", {}).get("avg"),
            "max_calls": summary.get("calls", {}).get("max"),
            "fallbacks": summary["fallbacks"],
            "format_ok": summary["format_ok"],
            "mock": args.mock,
            "subset": subset,
            "sec": round(time.time() - t_start, 1),
            "note": args.run_name,
        })
        print(f"로그: {run_dir.relative_to(ROOT)}  (leaderboard: runs/leaderboard.csv)")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
