"""검색만 따로 확인하기 (LLM 호출 없음).

    python scripts/search_index.py "휴학 신청은 언제까지?" -k 5
    python scripts/search_index.py --qid E-015 --input data/dev/dev_ewha.csv   # 개발 문항으로 검색 + 근거쪽 적중 여부
    python scripts/search_index.py "..." --mock --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.embeddings import make_embedder  # noqa: E402
from rag.index import load_or_build  # noqa: E402
from rag.parser import load_questions  # noqa: E402
from rag.scoring import parse_pages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--qid")
    ap.add_argument("--input", default="data/dev/dev_all.csv")
    ap.add_argument("--index", default=config.DEFAULT_INDEX)
    ap.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--full", action="store_true", help="청크 전문 출력")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold_pages = set()
    if a.qid:
        q = next((q for q in load_questions(a.input) if q.qid == a.qid), None)
        if q is None:
            sys.exit(f"{a.input}에 {a.qid} 없음")
        query = q.render()
        gold_pages = parse_pages(q.meta.get("gold_page"))
        print(f"[{q.qid}] 정답 {q.gold} | 근거 {q.meta.get('gold_ref', '-')} | 근거쪽 {sorted(gold_pages) or '-'}")
    elif a.query:
        query = a.query
    else:
        sys.exit("검색어 또는 --qid 를 주세요")

    name = ("mock_" + a.index) if a.mock and not a.index.startswith("mock_") else a.index
    emb = make_embedder(a.mock)
    idx = load_or_build(name, emb)
    hits = idx.search(emb.embed_queries([query])[0], a.k)
    print(f"질의: {query[:200]}\n")
    for h in hits:
        c = h.chunk
        mark = " ★근거쪽" if gold_pages & set(c.pages) else ""
        body = c.text if a.full else c.text[:220].replace("\n", " ") + ("…" if len(c.text) > 220 else "")
        print(f"#{h.rank} {h.score:.3f}  {c.chunk_id} p.{c.page_start}-{c.page_end}{mark}\n   {body}\n")


if __name__ == "__main__":
    main()
