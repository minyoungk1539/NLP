"""ewha_xref_v2: ewha_xref의 참조 수집 규칙만 바꾼 판.

ewha_xref(runs/20261127-004740_ewha_xref_ewha47_xref_1)에서 다른 법령의 조가 학칙 조로 잘못 붙었다.
  예) 「고등교육법」 제2조 → 학칙 제2조(명칭) 14회, 「고등교육법 시행령」 제29조 → 인덱스에 없음 13회
또 부칙 청크의 참조(시행일 예외 "다만, 제8조 제2항 제26호는 …부터 시행" 등)가 본문 참조보다 먼저 자리를 차지했다.

바뀐 점
  ① 참조 바로 앞에 다른 법령 이름이 있으면 건너뛴다: 「…」, ~법, ~령, ~규정, ~세칙, ~규칙
     (사이의 괄호 "대학학생정원령(대통령령 제13,875호, 1993.4.2) 제2조"와 공백은 무시).
     "「고등교육법」 제2조 및 제3조"처럼 접속어(및 · , 또는 와 과 부터 까지 ~)로만 이어진 뒤 참조도 같은 법령으로 보고 건너뛴다.
  ② 우선순위: 보기 → 질문 → 본문 청크(순위순) → 부칙 청크(순위순). 부칙 참조는 버리지 않고 맨 뒤로.
나머지(본문 조만 찾기, 이미 검색된 청크 제외, 최대 4개, PUA 제거, LLM 호출 방식, 기록 형식)는 ewha_xref와 같다.
pred.info["xref"]["external_skipped"]에 건너뛴 다른 법령 참조를 남긴다.
"""
from __future__ import annotations

import re

from .ewha_xref import _REF_RE, EwhaXref

# 참조 앞 텍스트의 끝이 법령 이름이면 다른 법령의 조로 본다
_EXTERNAL_TAIL_RE = re.compile(r"(?:」|법|령|규정|세칙|규칙)\s*$")
_TRAILING_PAREN_RE = re.compile(r"[(（][^()（）]*[)）]\s*$")
# 직전 참조와 이 참조 사이가 접속어뿐이면 같은 법령의 연속 참조
_CHAIN_GAP_RE = re.compile(r"^[\s,·ㆍ및또는와과부터까지~\-]*$")
_LOOKBACK = 60


def _is_external(prefix: str) -> bool:
    tail = prefix[-_LOOKBACK:].rstrip()
    for _ in range(3):                                  # "령(대통령령 제…호, 1993.4.2)"처럼 괄호가 끼어도 본다
        stripped = _TRAILING_PAREN_RE.sub("", tail).rstrip()
        if stripped == tail:
            break
        tail = stripped
    return bool(_EXTERNAL_TAIL_RE.search(tail))


def find_refs_v2(text: str) -> tuple[list[str], list[str]]:
    """(학칙 조 참조, 건너뛴 다른 법령 참조). 둘 다 나온 순서대로, 중복 제거, 조 번호로 정규화."""
    refs: list[str] = []
    skipped: list[str] = []
    text = text or ""
    prev_end, prev_external = None, False
    for m in _REF_RE.finditer(text):
        art = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        chained = prev_external and prev_end is not None and _CHAIN_GAP_RE.match(text[prev_end:m.start()])
        external = bool(chained) or _is_external(text[:m.start()])
        if external:
            lo = max(0, m.start() - _LOOKBACK)
            start = text.rfind("「", lo, m.start())              # 「법령 이름」부터, 없으면 앞 30자
            if start == -1:
                start = max(0, m.start() - 30)
                while start > lo and not text[start - 1].isspace():   # 단어 중간에서 자르지 않게
                    start -= 1
            label = re.sub(r"\s+", " ", text[start:m.end()].split("\n")[-1]).strip()
            if label not in skipped:
                skipped.append(label)
        elif art not in refs:
            refs.append(art)
        prev_end, prev_external = m.end(), external
    return refs, skipped


class EwhaXrefV2(EwhaXref):
    name = "ewha_xref_v2"

    def collect_refs(self, q, hits):
        cands: list[tuple[str, str]] = []
        seen: set[str] = set()
        skipped_all: list[str] = []

        def add_from(text: str, src: str, own: str | None = None) -> None:
            refs, skipped = find_refs_v2(text)
            for s in skipped:
                if s not in skipped_all:
                    skipped_all.append(s)
            for art in refs:
                if art != own and art not in seen:
                    seen.add(art)
                    cands.append((art, src))

        for letter, opt in q.options.items():                       # 1) 보기
            add_from(opt, f"option {letter}")
        add_from(q.stem, "question")                                 # 2) 질문
        body = [h for h in hits if h.chunk.meta.get("kind") != "addenda"]
        addenda = [h for h in hits if h.chunk.meta.get("kind") == "addenda"]
        for h in body:                                               # 3) 본문 청크 (순위순)
            add_from(h.chunk.text, f"passage {h.chunk.chunk_id}", h.chunk.meta.get("article"))
        for h in addenda:                                            # 4) 부칙 청크 (순위순)
            add_from(h.chunk.text, f"addenda {h.chunk.chunk_id}", h.chunk.meta.get("article"))
        return cands, {"external_skipped": skipped_all}
