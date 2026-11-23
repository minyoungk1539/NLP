"""ewha.pdf(이화여자대학교 학칙, 54쪽) → 페이지 텍스트 → 청크.

Baseline은 의도적으로 단순하다: 쪽 머리말만 지우고, 고정 길이로 자른다.
아래는 '개선 실험' 몫으로 남겨 둔 것 (실험 보드 참고):
- 줄 중간에서 끊긴 문장 이어 붙이기, 조(條) 단위 청킹
- 개정 날짜 "(개정 1999.2.9)" 메타데이터화
- p.9·p.20 성적 표처럼 열 단위로 추출되는 표 복원 (Document Parse 등)
- 별표 1의 여러 개정 버전 중 최신 버전 선택
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config

# 매 쪽 머리말: "이화여자대학교 학칙" / "2 - 2 - N"
_HEADER_LINE_RE = re.compile(r"^\s*(?:이화여자대학교 학칙|\d+\s*-\s*\d+\s*-\s*\d+)\s*$")


@dataclass
class Page:
    page_no: int          # 1-based
    text: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    meta: dict = field(default_factory=dict)

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pages(pdf_path: str | Path = config.EWHA_PDF, strip_headers: bool = True) -> list[Page]:
    try:
        import pymupdf
    except ImportError:  # 구버전 PyMuPDF
        import fitz as pymupdf

    pages = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            lines = page.get_text().splitlines()
            if strip_headers:
                lines = [ln for ln in lines if not _HEADER_LINE_RE.match(ln)]
            pages.append(Page(i + 1, "\n".join(ln.rstrip() for ln in lines).strip()))
    return pages


def chunk_fixed(pages: list[Page], size: int = config.DEFAULT_CHUNK_SIZE,
                overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> list[Chunk]:
    """전체 텍스트를 size 글자 안팎으로 자른다 (가능하면 줄 경계에서). 각 청크에 쪽 범위를 기록."""
    if overlap >= size:
        raise ValueError("overlap은 size보다 작아야 합니다")
    full, spans = "", []
    for p in pages:
        start = len(full)
        full += p.text + "\n"
        spans.append((start, len(full), p.page_no))

    def pages_of(a: int, b: int) -> list[int]:
        return [no for s, e, no in spans if s < b and e > a] or [spans[-1][2]]

    chunks, pos, n = [], 0, len(full)
    while pos < n:
        end = min(pos + size, n)
        if end < n:
            nl = full.rfind("\n", pos + int(size * 0.6), end)
            if nl != -1:
                end = nl + 1
        text = full[pos:end].strip()
        if text:
            pg = pages_of(pos, end)
            chunks.append(Chunk(f"c{len(chunks):04d}", text, min(pg), max(pg)))
        if end >= n:
            break
        nxt = max(end - overlap, pos + 1)
        nl = full.find("\n", nxt, end)                 # 다음 청크는 줄 시작에서
        pos = nl + 1 if nl != -1 else nxt
    return chunks
