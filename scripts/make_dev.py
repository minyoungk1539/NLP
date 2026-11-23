"""개발셋 만들기: MMLU-Pro 표본(dev_mmlu.csv) 다운로드 + 전체 합본(dev_all.csv).

    python scripts/make_dev.py                    # 과목당 15문항, seed 42
    python scripts/make_dev.py --per-subject 30 --seed 7

- MMLU-Pro test split(TIGER-Lab/MMLU-Pro, Hugging Face)에서 과제 5개 과목(law, psychology,
  business, philosophy, history)을 층화 추출한다. 연습 문항(test_samples*, 18개)과 같은 문제는 제외.
- 개발셋은 '평가용'이다. 검색 소스(인덱스)에 넣으면 규칙 위반(0점)이므로 절대 넣지 말 것.
- dev_official.csv(공식 25문항+근거 주석)와 dev_ewha.csv(학칙 기반 자작 문항)는 손으로 관리하는 파일.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rag import config  # noqa: E402
from rag.parser import load_questions  # noqa: E402

MMLU_PRO_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
SUBJECTS = ["law", "psychology", "business", "philosophy", "history"]
COLUMNS = ["question_id", "prompts", "answers", "domain", "subject", "qtype", "gold_ref", "gold_page", "source", "note"]


def _key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:120]


def fetch_mmlu_pro(cache: Path) -> pd.DataFrame:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"다운로드: {MMLU_PRO_URL}")
        urllib.request.urlretrieve(MMLU_PRO_URL, cache)
    return pd.read_parquet(cache)


def build_mmlu(per_subject: int, seed: int) -> pd.DataFrame:
    df = fetch_mmlu_pro(config.DEV_DIR / "_cache" / "mmlu_pro_test.parquet")
    practice = []
    for f in ("test_samples.csv", "test_samples_students.csv"):
        practice += [q for q in load_questions(config.DATA_DIR / f) if q.lang == "en"]
    practice_keys = {_key(q.stem) for q in practice}

    df = df[df["category"].isin(SUBJECTS)].copy()
    df["_k"] = df["question"].map(_key)
    overlap = df[df["_k"].isin(practice_keys)]
    print(f"연습 문항과 겹치는 MMLU-Pro 문항 {len(overlap)}개 제외 "
          f"(연습 영어 문항 {len(practice)}개 중 {overlap['_k'].nunique()}개가 MMLU-Pro test에 있음)")
    df = df[~df["_k"].isin(practice_keys)]

    rows = []
    for subj in SUBJECTS:
        part = df[df["category"] == subj].sample(n=per_subject, random_state=seed)
        for _, r in part.iterrows():
            opts = list(r["options"])
            letters = "ABCDEFGHIJ"[: len(opts)]
            body = "\n".join(f"({l}) {o}" for l, o in zip(letters, opts))
            rows.append({
                "question_id": f"M-{subj[:3].upper()}-{int(r['question_id']):05d}",
                "prompts": f"QUESTION{len(rows) + 1}) {r['question'].strip()}\n{body}",
                "answers": f"({r['answer']})",
                "domain": "mmlu", "subject": subj,
                "qtype": "knowledge", "gold_ref": "", "gold_page": "",
                "source": f"MMLU-Pro test qid={int(r['question_id'])} ({r['src']})",
                "note": "",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true", help="dev_mmlu.csv를 다시 만들지 않고 합본만")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    mmlu_path = config.DEV_DIR / "dev_mmlu.csv"
    if not a.skip_download:
        mmlu = build_mmlu(a.per_subject, a.seed)
        mmlu.to_csv(mmlu_path, index=False, encoding="utf-8")
        print(f"저장: {mmlu_path} ({len(mmlu)}문항)")

    parts = []
    for name in ("dev_official.csv", "dev_ewha.csv", "dev_mmlu.csv"):
        p = config.DEV_DIR / name
        if p.exists():
            parts.append(pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8"))
    merged = pd.concat(parts, ignore_index=True)[COLUMNS]
    dup = merged["question_id"].duplicated()
    if dup.any():
        sys.exit(f"question_id 중복: {merged.loc[dup, 'question_id'].tolist()[:5]}")
    merged.to_csv(config.DEV_DIR / "dev_all.csv", index=False, encoding="utf-8")
    print(f"저장: {config.DEV_DIR / 'dev_all.csv'} ({len(merged)}문항) "
          + " / ".join(f"{k} {v}" for k, v in merged["domain"].value_counts().items()))


if __name__ == "__main__":
    main()
