"""결과 집계: 전체/도메인별/유형별 정확도, 호출 수, 검색 쪽 적중률(개발셋에 gold_page가 있을 때)."""
from __future__ import annotations

from collections import defaultdict


def _acc(rows) -> dict:
    graded = [r for r in rows if r.get("gold")]
    n = len(graded)
    c = sum(1 for r in graded if r["pred"] == r["gold"])
    return {"n": n, "correct": c, "acc": round(100 * c / n, 2) if n else None}


def parse_pages(value) -> set[int]:
    out = set()
    for tok in str(value or "").replace(",", ";").split(";"):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


def summarize(results: list[dict]) -> dict:
    s = {"n": len(results), "overall": _acc(results)}
    for key in ("domain", "qtype"):
        groups = defaultdict(list)
        for r in results:
            if r.get(key):
                groups[r[key]].append(r)
        if groups:
            s[f"by_{key}"] = {k: _acc(v) for k, v in sorted(groups.items())}
    calls = [r["calls"] for r in results if "calls" in r]
    if calls:
        s["calls"] = {"avg": round(sum(calls) / len(calls), 2), "max": max(calls)}
    s["fallbacks"] = sum(1 for r in results if r.get("fallback"))
    hits = [r["gold_page_hit"] for r in results if r.get("gold_page_hit") is not None]
    if hits:
        s["retrieval_gold_page_hit"] = {"n": len(hits), "rate": round(100 * sum(hits) / len(hits), 2)}
    return s


def format_summary(s: dict) -> str:
    o = s["overall"]
    lines = []
    if o["n"]:
        lines.append(f"정확도  {o['acc']}%  ({o['correct']}/{o['n']})")
    else:
        lines.append("정답 컬럼이 없어 정확도는 계산하지 않음")
    for key, title in (("by_domain", "도메인별"), ("by_qtype", "유형별")):
        if key in s:
            parts = [f"{k} {v['acc']}% ({v['correct']}/{v['n']})" for k, v in s[key].items() if v["n"]]
            if parts:
                lines.append(f"{title}  " + " | ".join(parts))
    if "calls" in s:
        lines.append(f"LLM 호출  평균 {s['calls']['avg']} / 최대 {s['calls']['max']} (한도 5)")
    if "retrieval_gold_page_hit" in s:
        h = s["retrieval_gold_page_hit"]
        lines.append(f"검색 근거쪽 적중률  {h['rate']}% (gold_page 있는 {h['n']}문항)")
    if s.get("fallbacks"):
        lines.append(f"⚠ fallback 답 {s['fallbacks']}건 (추출 실패 또는 API 오류 → 로그 확인)")
    return "\n".join(lines)
