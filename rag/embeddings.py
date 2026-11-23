"""임베딩 래퍼. 문서용(passage)과 질문용(query) 모델이 다르다는 점에 주의.

임베딩 호출은 LLM 호출 5회 제한에 포함되지 않는다.
"""
from __future__ import annotations

import threading
import zlib

import numpy as np

from . import config


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class UpstageEmbedder:
    model_id = "solar-embedding-2"

    def __init__(self, api_key: str | None = None, batch_size: int = 32):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or config.get_api_key(), base_url=config.BASE_URL,
                             timeout=120, max_retries=5)
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str], model: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=model, input=batch, encoding_format="float")
            out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return _normalize(np.array(out))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, config.EMBED_PASSAGE_MODEL)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._embed(missing, config.EMBED_QUERY_MODEL)
            with self._lock:
                self._cache.update(zip(missing, vecs))
        return np.stack([self._cache[t] for t in texts])


class MockEmbedder:
    """API 없이 돌려보는 해시 기반 글자 n-gram 임베딩 (오프라인 테스트용, 성능 비교에 쓰지 말 것)."""

    model_id = "mock-hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = "".join(text.split())
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                v[zlib.crc32(t[i:i + n].encode("utf-8")) % self.dim] += 1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    embed_queries = embed_passages


def make_embedder(mock: bool = False):
    return MockEmbedder() if mock else UpstageEmbedder()
