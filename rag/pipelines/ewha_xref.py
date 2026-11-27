"""ewha_xref: ewha_prompt_v2 + 조항 상호참조 보강.

학칙 경로에서
  ① 질문 본문·보기·검색된 청크에서 조항 참조(제N조, 제N조의M, 제N조제M항, 제N조제M호 …)를 정규식으로 찾고
  ② 그 조의 본문 청크(ewha_article 인덱스의 meta.article, kind == "article")를 context 뒤에 덧붙인다.
     이미 검색된 청크는 빼고, 질문·보기에서 찾은 참조를 먼저, 그다음 검색 청크(순위순)에서 찾은 참조 순으로 최대 4개.
  ③ 덧붙인 청크에도 ewha_prompt_v2와 같은 깨진 문자(PUA) 제거를 적용한다.
전문지식 경로와 LLM 호출 방식(첫 호출 + 답을 못 뽑으면 재시도 1회, 최대 2회)은 ewha_prompt_v2와 같다.

기록
  pred.info["xref"]      : 찾은 참조, 덧붙인 조·청크 ID와 출처, 인덱스에 없는 참조
  pred.info["retrieved"] : 검색 top-k 뒤에 덧붙인 청크도 넣는다 ("xref": true). 검색 근거쪽 적중률이 덧붙인 청크까지 센다.

참고
  - 부칙 안의 조(예: 부칙의 제1조(시행일))는 본문 조와 번호가 겹치므로 찾는 대상에서 뺀다 (kind == "article"만).
  - 검색 청크 안의 참조에서는 그 청크 자신의 조(머리말 "[… > 제31조(재입학)]" 등)는 뺀다.
  - 부칙 청크 안의 "제2조" 같은 참조가 부칙의 조를 가리켜도 본문 조로 해석된다 (구분할 정보가 없음).
  - 덧붙인 청크의 score는 같은 질의 벡터와의 실제 유사도다.
"""
from __future__ import annotations

import re

from .. import prompts
from ..index import Hit
from .ewha_prompt_v2 import SYSTEM_EWHA_V2, EwhaPromptV2, clean_hits

MAX_XREF = 4

# 제N조 / 제N조의M (뒤의 제M항·제M호는 조를 특정하는 데 필요 없으므로 매치만 하고 버린다)
_REF_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*제\s*\d+\s*[항호])*")


def find_refs(text: str) -> list[str]:
    """텍스트에 나온 순서대로 조 번호를 정규화해 돌려준다 (중복 제거). 예: '제 28조제4호' → '제28조'."""
    out: list[str] = []
    for m in _REF_RE.finditer(text or ""):
        art = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        if art not in out:
            out.append(art)
    return out


def user_rag_xref(q, hits, xref_hits) -> str:
    """prompts.user_rag와 같은 형식 + context 뒤에 상호참조 조항 블록."""
    context = prompts.format_context(hits)
    if xref_hits:
        context += ("\n\nReferenced articles (cited in the question or in the passages above):\n"
                    + prompts.format_context(xref_hits))
    return f"Context:\n{context}\n\nQuestion:\n{q.render()}\n\n{prompts.valid_letters_line(q)}"


class EwhaXref(EwhaPromptV2):
    name = "ewha_xref"
    max_xref = MAX_XREF

    def setup(self) -> None:
        super().setup()
        self.article_pos: dict[str, list[int]] = {}       # "제31조" → 인덱스 내 위치 (본문 조만)
        for i, c in enumerate(self.index.chunks):
            if c.meta.get("kind") == "article" and c.meta.get("article"):
                self.article_pos.setdefault(c.meta["article"], []).append(i)
        if not self.article_pos:
            print(f"[ewha_xref] 경고: 인덱스 '{self.index_name}'에 조 메타데이터(meta.article)가 없어 "
                  "상호참조를 덧붙이지 않습니다 (ewha_article 인덱스를 쓰세요).")

    def describe(self) -> dict:
        return {**super().describe(), "max_xref": self.max_xref,
                "xref_articles_in_index": len(self.article_pos)}

    def collect_refs(self, q, hits) -> tuple[list[tuple[str, str]], dict]:
        """참조 후보 [(조, 출처)]를 우선순위 순으로. 질문·보기 먼저, 그다음 검색 청크(순위순, 자기 조 제외).
        두 번째 값은 info["xref"]에 더할 추가 정보 (v2에서 씀)."""
        cands: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(art: str, src: str) -> None:
            if art not in seen:
                seen.add(art)
                cands.append((art, src))

        for art in find_refs(q.stem):
            add(art, "question")
        for letter, opt in q.options.items():
            for art in find_refs(opt):
                add(art, f"option {letter}")
        for h in hits:
            own = h.chunk.meta.get("article")
            for art in find_refs(h.chunk.text):
                if art != own:
                    add(art, f"passage {h.chunk.chunk_id}")
        return cands, {}

    def xref_hits(self, q, hits, qvec) -> tuple[list[Hit], dict]:
        cands, extra = self.collect_refs(q, hits)
        retrieved_ids = {h.chunk.chunk_id for h in hits}
        added, missing, already = [], [], []
        out: list[Hit] = []
        for art, src in cands:
            if len(out) >= self.max_xref:
                break
            pos = self.article_pos.get(art)
            if not pos:
                missing.append(art)
                continue
            new = [i for i in pos if self.index.chunks[i].chunk_id not in retrieved_ids]
            if not new:
                already.append(art)
                continue
            for i in new:
                if len(out) >= self.max_xref:
                    break
                c = self.index.chunks[i]
                score = float(self.index.vectors[i] @ qvec)
                out.append(Hit(c, score, len(hits) + len(out) + 1))
                added.append({"article": art, "chunk_id": c.chunk_id, "from": src})
        info = {"refs_found": [a for a, _ in cands], "added": added,
                "already_retrieved": already, "not_in_index": missing, **extra}
        return out, info

    def predict(self, q, llm):
        qvec = self.embedder.embed_queries([q.render()])[0]
        hits = self.index.search(qvec, self.top_k)
        route, why = self.route(q, hits)
        if route != "ewha":
            pred = self.solve_general(q, llm)
            pred.info.update(route=route, route_reason=why)
            return pred

        xhits, xinfo = self.xref_hits(q, hits, qvec)
        messages = [{"role": "system", "content": SYSTEM_EWHA_V2},
                    {"role": "user", "content": user_rag_xref(q, clean_hits(hits), clean_hits(xhits))}]
        pred = self.ask_with_retry(q, llm, messages, "answer", self.ewha_params)
        pred.info["retrieved"] = [h.brief() for h in hits] + [{**h.brief(), "xref": True} for h in xhits]
        pred.info["xref"] = xinfo
        pred.info.update(route=route, route_reason=why)
        return pred
