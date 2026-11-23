"""벡터 인덱스: 청크 + 정규화된 임베딩 행렬. numpy 내적으로 검색 (청크 수백 개라 FAISS 불필요).

저장 형식 (index/<name>/)
- chunks.jsonl  청크 텍스트·쪽 범위·메타
- vectors.npy   float32 (N, d), L2 정규화
- meta.json     임베딩 모델, 청킹 설정, 원본 PDF sha256, 생성 시각
제출 ZIP에 이 폴더를 넣어야 채점 환경에서 재현된다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .corpus import Chunk, chunk_article, chunk_fixed, file_sha256, load_pages


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int

    def brief(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id, "pages": [self.chunk.page_start, self.chunk.page_end],
                "score": round(self.score, 4)}


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict):
        if len(chunks) != len(vectors):
            raise ValueError("청크 수와 벡터 수가 다릅니다")
        self.chunks, self.vectors, self.meta = chunks, vectors.astype(np.float32), meta

    def search(self, qvec: np.ndarray, k: int) -> list[Hit]:
        scores = self.vectors @ qvec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(top)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(path / "vectors.npy", self.vectors)
        (path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(path / "vectors.npy"), meta)


def index_path(name: str) -> Path:
    return config.INDEX_DIR / name


def build_ewha_index(name: str, embedder, size: int = config.DEFAULT_CHUNK_SIZE,
                     overlap: int = config.DEFAULT_CHUNK_OVERLAP, chunker: str = "fixed") -> VectorIndex:
    """ewha.pdf → 청크 (fixed: 고정 길이 / article: 조 단위) → 임베딩 → index/<name>/ 저장."""
    pages = load_pages(config.EWHA_PDF)
    if chunker == "article":
        chunks = chunk_article(pages, max_chars=size, overlap=overlap)
    elif chunker == "fixed":
        chunks = chunk_fixed(pages, size=size, overlap=overlap)
    else:
        raise ValueError(f"알 수 없는 chunker: {chunker}")
    vectors = embedder.embed_passages([c.text for c in chunks])
    meta = {
        "name": name,
        "source": "data/ewha.pdf",
        "source_sha256": file_sha256(config.EWHA_PDF),
        "chunker": {"type": chunker, "size": size, "overlap": overlap, "strip_headers": True},
        "embedder": embedder.model_id,
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    idx = VectorIndex(chunks, vectors, meta)
    idx.save(index_path(name))
    return idx


def load_or_build(name: str, embedder, size: int = config.DEFAULT_CHUNK_SIZE,
                  overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> VectorIndex:
    path = index_path(name)
    if (path / "meta.json").exists():
        idx = VectorIndex.load(path)
        if idx.meta.get("embedder") != embedder.model_id:
            raise SystemExit(
                f"인덱스 '{name}'은 {idx.meta.get('embedder')}로 만들어졌는데 지금 임베더는 {embedder.model_id}입니다. "
                "--mock 여부나 --index 이름을 확인하세요."
            )
        return idx
    print(f"[index] '{name}' 인덱스가 없어 새로 만듭니다 (임베딩 API 호출, LLM 호출 수에는 포함 안 됨)...")
    return build_ewha_index(name, embedder, size=size, overlap=overlap)
