"""MMLU(전문 지식) 문항 개선 실험. 이화 문항은 routed와 똑같이 학칙 RAG로 푼다.

변형 (run.py --pipeline <이름>)
  gen_prompt   : 프롬프트 개선 (단계별 추론, 애매해도 반드시 하나 고르기) + max_tokens 4096
  gen_r_low    : gen_prompt + reasoning_effort=low
  gen_r_high   : gen_prompt + reasoning_effort=high
  gen_sc5      : 같은 문항을 temperature 0.7로 5번 풀고 다수결 (Self-Consistency, LLM 5회)
  gen_wiki     : 위키백과 API 검색 구절을 컨텍스트로 넣기 (LLM 1회)
  gen_wiki_q   : LLM이 검색어를 만든 뒤 위키백과 검색 (LLM 2회)
  gen_hybrid   : low 추론(간결 지시) → 답이 비면 추론 없이 재시도 (LLM 1–2회)
  gen_hybrid_sc3: low 추론 3번 다수결 → 모두 비면 추론 없이 1번 (LLM 3–4회)
  gen_wiki_hybrid: 위키백과 구절 + gen_hybrid (LLM 1–2회)

모든 변형은 입력 CSV의 domain/subject 열을 쓰지 않는다 (평가 입력에는 없음).
"""
from __future__ import annotations

from collections import Counter

from .. import prompts
from ..answer import extract_letter
from .base import Prediction
from .routed import RoutedRAG

SYSTEM_GENERAL = (
    "You are a world-class expert in law, psychology, business, philosophy and history, "
    "answering a difficult multiple-choice exam question.\n"
    "Work through the question step by step: identify what is being asked, recall the relevant "
    "principle, rule, fact or formula, and test each option against it. For calculations, show the arithmetic.\n"
    "Exactly one option is correct. Even if you are unsure, you must pick the single most likely option; "
    "never answer 'none', 'X' or leave it blank.\n"
    + prompts.OUTPUT_RULE
)

SYSTEM_GENERAL_CONTEXT = (
    SYSTEM_GENERAL
    + "\nYou are also given encyclopedia excerpts that may or may not be relevant. "
    "Use them only when they actually help; otherwise rely on your own knowledge."
)

SYSTEM_QUERY = (
    "You write search queries for English Wikipedia. Given an exam question, output 1 to 3 short "
    "search queries (each a few words naming the key concept, doctrine, person, event or theory needed to "
    "answer it), one per line, with no numbering or other text."
)


def _user(q, context: str | None = None) -> str:
    base = prompts.user_llm_only(q)
    return f"Encyclopedia excerpts:\n{context}\n\n{base}" if context else base


class GeneralExperiment(RoutedRAG):
    """routed의 '일반(MMLU)' 경로만 바꾼다."""
    name = "gen_base"
    gen_params: dict = {"max_tokens": 4096}
    system = SYSTEM_GENERAL

    def describe(self) -> dict:
        return {**super().describe(), "gen_params": self.gen_params, "variant": self.name}

    def solve_general(self, q, llm) -> Prediction:
        out = llm.chat([{"role": "system", "content": self.system}, {"role": "user", "content": _user(q)}],
                       tag="general", **self.gen_params)
        letter, method = extract_letter(out, q.letters)
        return Prediction(letter, out, method)

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())
        route, why = self.route(q, hits)
        if route == "ewha":
            pred = super().predict(q, llm)       # routed와 동일 (검색은 한 번 더 하지만 LLM 호출 수는 같음)
            return pred
        pred = self.solve_general(q, llm)
        pred.info.update(route=route, route_reason=why)
        return pred


class GenPrompt(GeneralExperiment):
    name = "gen_prompt"


class GenReasonLow(GeneralExperiment):
    name = "gen_r_low"
    gen_params = {"max_tokens": 4096, "reasoning_effort": "low"}


class GenReasonHigh(GeneralExperiment):
    name = "gen_r_high"
    gen_params = {"max_tokens": 4096, "reasoning_effort": "high"}


class GenSelfConsistency(GeneralExperiment):
    """Self-Consistency (Wang et al., 2023): 독립 풀이 n개 → 다수결. 동률이면 먼저 나온 답."""
    name = "gen_sc5"
    n_samples = 5
    sample_params = {"max_tokens": 4096, "temperature": 0.7}

    def solve_general(self, q, llm) -> Prediction:
        votes, outs = [], []
        for i in range(self.n_samples):
            out = llm.chat([{"role": "system", "content": self.system}, {"role": "user", "content": _user(q)}],
                           tag=f"sample{i + 1}", **self.sample_params)
            outs.append(out)
            letter, _ = extract_letter(out, q.letters)
            if letter:
                votes.append(letter)
        if not votes:
            return Prediction(None, outs[-1], "none", info={"votes": []})
        counts = Counter(votes)
        top = max(counts.values())
        winner = next(v for v in votes if counts[v] == top)
        return Prediction(winner, outs[votes.index(winner)] if winner in votes else outs[-1], "vote",
                          info={"votes": votes, "vote_share": round(top / len(votes), 2)})


class GenWiki(GeneralExperiment):
    """위키백과 Action API 검색 → 상위 문서 요약 구절을 컨텍스트로."""
    name = "gen_wiki"
    use_llm_query = False
    n_pages = 3

    def setup(self) -> None:
        super().setup()
        from ..wiki import WikiClient
        self.wiki = WikiClient()

    def queries(self, q, llm) -> list[str]:
        if not self.use_llm_query:
            return [q.stem[:300]]
        out = llm.chat([{"role": "system", "content": SYSTEM_QUERY}, {"role": "user", "content": q.render()}],
                       tag="query", max_tokens=200)
        qs = [ln.strip(" -*\"'") for ln in out.splitlines() if ln.strip()]
        return qs[:3] or [q.stem[:300]]

    def solve_general(self, q, llm) -> Prediction:
        queries = self.queries(q, llm)
        passages = self.wiki.passages(queries, n_pages=self.n_pages)
        context = "\n\n".join(f"[{i + 1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)) or None
        out = llm.chat([{"role": "system", "content": SYSTEM_GENERAL_CONTEXT},
                        {"role": "user", "content": _user(q, context)}], tag="general", **self.gen_params)
        letter, method = extract_letter(out, q.letters)
        return Prediction(letter, out, method,
                          info={"wiki_queries": queries, "wiki_titles": [p["title"] for p in passages]})


class GenWikiQuery(GenWiki):
    name = "gen_wiki_q"
    use_llm_query = True


BRIEF_THINKING = ("\nThink efficiently: keep your private reasoning under about 800 words and do not "
                  "re-check the same option repeatedly.")


class GenHybrid(GeneralExperiment):
    """reasoning_effort=low(+간결 지시)로 먼저 풀고, 추론이 4096 토큰을 다 써서 답이 비면 추론 없이 다시 푼다.
    dev에서 low 추론은 3문항 중 1개꼴로 토큰을 다 써 버렸다 (gen_r_low fallback 34/88)."""
    name = "gen_hybrid"
    n_reason = 1                       # 추론 풀이 횟수 (>1이면 다수결)
    reason_params = {"max_tokens": 4096, "reasoning_effort": "low", "temperature": 0.0}
    plain_params = {"max_tokens": 4096, "reasoning_effort": "none", "temperature": 0.0}

    def solve_general(self, q, llm) -> Prediction:
        votes, outs, used_plain = [], [], False
        sys_reason = SYSTEM_GENERAL + BRIEF_THINKING
        for i in range(self.n_reason):
            params = dict(self.reason_params)
            if i > 0:
                params["temperature"] = 0.7
            out = llm.chat([{"role": "system", "content": sys_reason}, {"role": "user", "content": _user(q)}],
                           tag=f"reason{i + 1}", **params)
            outs.append(out)
            letter, _ = extract_letter(out, q.letters)
            if letter:
                votes.append(letter)
        if not votes and llm.remaining > 0:                # 추론이 모두 실패하면 추론 없이 1번
            used_plain = True
            out = llm.chat([{"role": "system", "content": SYSTEM_GENERAL}, {"role": "user", "content": _user(q)}],
                           tag="plain", **self.plain_params)
            outs.append(out)
            letter, _ = extract_letter(out, q.letters)
            if letter:
                votes.append(letter)
        if not votes:
            return Prediction(None, outs[-1], "none", info={"votes": [], "used_plain": used_plain})
        counts = Counter(votes)
        top = max(counts.values())
        winner = next(v for v in votes if counts[v] == top)
        return Prediction(winner, outs[-1], "vote" if len(votes) > 1 else "single",
                          info={"votes": votes, "used_plain": used_plain})


class GenHybridSC3(GenHybrid):
    """추론 풀이 3번 다수결 (+ 전부 실패 시 추론 없이 1번). LLM 최대 4회."""
    name = "gen_hybrid_sc3"
    n_reason = 3


class GenWikiHybrid(GenWiki):
    """gen_wiki + gen_hybrid: 위키백과 구절을 붙이고 low 추론(간결 지시)으로 풀고,
    추론이 토큰을 다 써서 답이 비면 같은 컨텍스트로 추론 없이 다시 푼다 (LLM 1–2회)."""
    name = "gen_wiki_hybrid"
    reason_params = {"max_tokens": 4096, "reasoning_effort": "low", "temperature": 0.0}
    plain_params = {"max_tokens": 4096, "reasoning_effort": "none", "temperature": 0.0}

    def solve_general(self, q, llm) -> Prediction:
        queries = self.queries(q, llm)
        passages = self.wiki.passages(queries, n_pages=self.n_pages)
        context = "\n\n".join(f"[{i + 1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)) or None
        user = _user(q, context)
        info = {"wiki_queries": queries, "wiki_titles": [p["title"] for p in passages], "used_plain": False}
        out = llm.chat([{"role": "system", "content": SYSTEM_GENERAL_CONTEXT + BRIEF_THINKING},
                        {"role": "user", "content": user}], tag="reason", **self.reason_params)
        letter, method = extract_letter(out, q.letters)
        if letter is None and llm.remaining > 0:
            info["used_plain"] = True
            out = llm.chat([{"role": "system", "content": SYSTEM_GENERAL_CONTEXT},
                            {"role": "user", "content": user}], tag="plain", **self.plain_params)
            letter, method = extract_letter(out, q.letters)
        return Prediction(letter, out, method, info=info)


EXPERIMENTS = (GenPrompt, GenReasonLow, GenReasonHigh, GenSelfConsistency, GenWiki, GenWikiQuery,
               GenHybrid, GenHybridSC3, GenWikiHybrid)
