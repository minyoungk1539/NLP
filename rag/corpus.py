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


# ── 조(條) 단위 청킹 ────────────────────────────────────────────────
_CHAPTER_RE = re.compile(r"^제\s*\d+\s*장\s*\S")                       # 제3장 부설기관
_SECTION_RE = re.compile(r"^제\s*\d+\s*절\s*\S")                       # 제1절 ...
_ARTICLE_RE = re.compile(r"^(제\d+조(?:의\d+)?)\s*[（(]([^)）]*)[)）]")   # 제31조(재입학)
_ADDENDA_RE = re.compile(r"^부\s*칙")                                   # 부칙(2014. 6. 17. 개정)
_TABLE_RE = re.compile(r"^\[별표\s*\d+\]")                              # [별표 1]
# 새 줄로 시작해야 하는 구조 표시: ①항, 1.호, 가.목, 장·절·조·부칙·별표
_BLOCK_START_RE = re.compile(r"^(?:[①-⑳]|\d+\.\s|[가-하]\.\s|제\s*\d+\s*[장절조]|부\s*칙|\[별표)")
_HANGUL_END_RE = re.compile(r"[가-힣]$")
_HANGUL_START_RE = re.compile(r"^[가-힣]")


def _join_lines(lines: list[str]) -> str:
    """PDF 줄바꿈으로 끊긴 문장을 잇는다. 항·호·목 같은 구조 줄은 새 줄로 둔다.
    한글-한글 사이 줄바꿈은 붙인다 ("편입학\n을" → "편입학을"). 줄 끝이 어절 경계일 때
    띄어쓰기가 사라질 수 있지만 검색에는 영향이 작다."""
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if not out or _BLOCK_START_RE.match(s):
            out.append(s)
        elif _HANGUL_END_RE.search(out[-1]) and _HANGUL_START_RE.match(s):
            out[-1] += s
        else:
            out[-1] += " " + s
    return "\n".join(out)


def chunk_article(pages: list[Page], max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """조·부칙·별표 단위로 자른다. 긴 조는 max_chars 창으로 나누되 머리말(장 > 조 제목)을 매 창 앞에 붙인다.

    - 본문 조: 머리말 "제3장 부설기관 > 제7조(부설기관)"
    - 부칙 안의 조: 머리말 "부칙(2014. 6. 17. 개정) > 제1조(시행일)"  (본문 조와 헷갈리지 않게)
    - meta: article(제31조), title(재입학), chapter, addenda, kind(article/addenda/table/preamble)
    """
    sections: list[dict] = []
    chapter, addenda = "", ""
    cur = {"kind": "preamble", "head": "학칙 서두", "article": "", "title": "",
           "chapter": "", "addenda": "", "lines": []}

    def flush():
        if any(t for t, _ in cur["lines"]):
            sections.append(dict(cur))

    for p in pages:
        for raw in p.text.splitlines():
            s = raw.strip()
            if not s:
                continue
            new = None
            if _CHAPTER_RE.match(s) and len(s) < 70:
                chapter = re.sub(r"\s*[（(]\s*(?:개정|신설).*$", "", s).strip()   # 장 제목의 개정 날짜는 뺀다
                addenda = ""
                continue                                  # 장 제목은 머리말로만 쓴다
            if _SECTION_RE.match(s) and len(s) < 70:
                continue
            if _TABLE_RE.match(s):
                addenda = ""
                new = {"kind": "table", "head": s.split("(")[0].strip() + " " + s[len(s.split("(")[0]):].strip(),
                       "article": s.split("]")[0] + "]", "title": "", "chapter": "", "addenda": ""}
            elif _ADDENDA_RE.match(s) and len(s) < 40:
                addenda = re.sub(r"\s+", " ", s)
                new = {"kind": "addenda", "head": addenda, "article": "", "title": "",
                       "chapter": "", "addenda": addenda}
            else:
                m = _ARTICLE_RE.match(s)
                if m:
                    art, title = m.group(1), m.group(2).strip()
                    if addenda:
                        head = f"{addenda} > {art}({title})"
                    else:
                        head = f"{chapter} > {art}({title})" if chapter else f"{art}({title})"
                    new = {"kind": "addenda" if addenda else "article", "head": head, "article": art,
                           "title": title, "chapter": "" if addenda else chapter, "addenda": addenda}
            if new:
                flush()
                cur = {**new, "lines": []}
            cur["lines"].append((s, p.page_no))
    flush()

    chunks: list[Chunk] = []
    for sec in sections:
        body_lines = [t for t, _ in sec["lines"]]
        # 줄별 쪽 번호를 잇기 전 텍스트 위치로 추적하기 어려우므로, 창 단위 쪽 범위는 줄 비율로 근사
        text = _join_lines(body_lines)
        pages_all = [pg for _, pg in sec["lines"]]
        windows = []
        if len(text) <= max_chars:
            windows.append((0, len(text)))
        else:
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                if end < len(text):
                    nl = text.rfind("\n", start + int(max_chars * 0.5), end)
                    if nl != -1:
                        end = nl
                windows.append((start, end))
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
        for a, b in windows:
            lo = pages_all[min(len(pages_all) - 1, int(len(pages_all) * a / max(1, len(text))))]
            hi = pages_all[min(len(pages_all) - 1, int(len(pages_all) * b / max(1, len(text))))]
            body = text[a:b].strip()
            if not body:
                continue
            prefix = sec["head"]
            full = body if body.startswith(prefix) else f"[{prefix}]\n{body}"
            meta = {k: sec[k] for k in ("kind", "article", "title", "chapter", "addenda") if sec[k]}
            chunks.append(Chunk(f"a{len(chunks):04d}", full, min(lo, hi), max(lo, hi), meta))
    return chunks
