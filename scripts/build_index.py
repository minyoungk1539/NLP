"""학칙 인덱스 만들기.

    python scripts/build_index.py                       # index/ewha_fixed800 (Upstage 임베딩)
    python scripts/build_index.py --size 500 --overlap 100 --name ewha_fixed500
    python scripts/build_index.py --chunker article --size 1200 --overlap 200 --name ewha_article
    python scripts/build_index.py --mock                # index/mock_ewha_fixed800 (API 없이)
    python scripts/build_index.py --dump                # 청크를 index/<name>/chunks_preview.txt 로도 저장

인덱스는 한 명이 만들어 공유하면 된다 (임베딩 비용은 LLM 호출 수에 포함되지 않음).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.embeddings import make_embedder  # noqa: E402
from rag.index import build_ewha_index, index_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=config.DEFAULT_INDEX)
    ap.add_argument("--size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--chunker", choices=["fixed", "article"], default="fixed",
                    help="fixed: 고정 길이 / article: 조·부칙·별표 단위 (긴 조는 --size 창으로 나눔)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 만들기")
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    name = ("mock_" + a.name) if a.mock and not a.name.startswith("mock_") else a.name
    path = index_path(name)
    if (path / "meta.json").exists() and not a.force:
        print(f"이미 있음: {path} (다시 만들려면 --force)")
        return
    idx = build_ewha_index(name, make_embedder(a.mock), size=a.size, overlap=a.overlap, chunker=a.chunker)
    print(f"저장: {path}  청크 {idx.meta['n_chunks']}개, dim {idx.meta['dim']}, 임베더 {idx.meta['embedder']}")
    lens = [len(c.text) for c in idx.chunks]
    print(f"청크 길이(글자) 평균 {sum(lens) / len(lens):.0f}, 최소 {min(lens)}, 최대 {max(lens)}")
    if a.dump:
        with open(path / "chunks_preview.txt", "w", encoding="utf-8") as f:
            for c in idx.chunks:
                f.write(f"===== {c.chunk_id}  p.{c.page_start}-{c.page_end}  ({len(c.text)}자)\n{c.text}\n\n")
        print(f"미리보기: {path / 'chunks_preview.txt'}")


if __name__ == "__main__":
    main()
