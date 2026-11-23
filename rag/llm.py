"""LLM 호출 래퍼.

- 모델은 config.CHAT_MODEL(solar-mini4-260922)로 고정. 다른 모델을 넘기면 에러.
- 문항마다 QuestionSession을 하나 만들고, 그 세션으로만 호출한다.
  세션이 호출 수를 세고 5회를 넘기려 하면 BudgetExceeded를 던진다.
  (보수적으로 집계: API 오류로 실패한 호출도 1회로 센다.)
- max_tokens ≤ 4096 강제, n>1 금지 (한 번에 여러 샘플 = 여러 호출로 볼 소지가 있음).
- 모든 호출은 CallRecord로 남아 runs/<run>/calls.jsonl에 저장된다.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import asdict, dataclass

from . import config


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallRecord:
    question_id: str
    call_index: int
    tag: str
    params: dict
    messages: list
    output: str | None
    usage: dict | None
    latency_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── 외부 호출 감사 (HTTP 계층) ─────────────────────────────────────────
# QuestionSession의 집계는 파이프라인이 쓰는 창구 안에서 센다. 이것과 별개로, Upstage로 실제 나간
# /chat/completions HTTP 응답을 httpx 이벤트 훅에서 센다. 파이프라인이 세션을 우회해도 여기서 잡힌다.
# run.py가 문항마다 audit_scope(qid)를 열고, 끝난 뒤 http_audit(qid)로 검사한다 (스레드별 문항 1개).
_audit_local = threading.local()
_audit_lock = threading.Lock()
_AUDIT: dict[str, dict[str, int]] = {}


class audit_scope:
    def __init__(self, qid: str):
        self.qid = qid

    def __enter__(self):
        _audit_local.qid = self.qid
        with _audit_lock:
            _AUDIT[self.qid] = {"ok": 0, "failed": 0}
        return self

    def __exit__(self, *exc):
        _audit_local.qid = None
        return False


def http_audit(qid: str) -> dict[str, int]:
    with _audit_lock:
        return dict(_AUDIT.get(qid, {"ok": 0, "failed": 0}))


def _on_http_response(response) -> None:
    if not response.request.url.path.endswith("/chat/completions"):
        return                                          # 임베딩 등은 LLM 호출이 아님
    qid = getattr(_audit_local, "qid", None) or "<no-question>"
    key = "ok" if response.status_code == 200 else "failed"   # 429·5xx 뒤 SDK 재시도는 failed로 따로 센다
    with _audit_lock:
        _AUDIT.setdefault(qid, {"ok": 0, "failed": 0})[key] += 1


class UpstageChat:
    """실제 Upstage 호출 백엔드. 여러 스레드에서 같이 써도 된다."""

    def __init__(self, api_key: str | None = None, timeout: float = 120, max_retries: int = 5):
        import httpx
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or config.get_api_key(),
            base_url=config.BASE_URL,
            timeout=timeout,
            max_retries=max_retries,            # 429·5xx·연결 오류는 SDK가 지수 백오프로 재시도
            http_client=httpx.Client(timeout=timeout, event_hooks={"response": [_on_http_response]}),
        )
        self._drop_reasoning = False
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], **params) -> tuple[str, dict | None]:
        from openai import BadRequestError

        params = dict(params)
        if self._drop_reasoning:
            params.pop("reasoning_effort", None)
        try:
            resp = self.client.chat.completions.create(model=config.CHAT_MODEL, messages=messages, **params)
        except BadRequestError as e:
            if "reasoning_effort" in params and "reasoning" in str(e).lower():
                with self._lock:
                    if not self._drop_reasoning:
                        print(f"[llm] 모델이 reasoning_effort를 거부해서 이후 호출에서는 뺍니다: {e}")
                    self._drop_reasoning = True
                params.pop("reasoning_effort", None)
                resp = self.client.chat.completions.create(model=config.CHAT_MODEL, messages=messages, **params)
            else:
                raise
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return text, usage


class MockChat:
    """API 없이 배관을 시험하는 가짜 백엔드. 'Valid letters:' 줄에서 고른 결정적 답을 돌려준다."""

    def complete(self, messages: list[dict], **params) -> tuple[str, dict | None]:
        last = messages[-1]["content"] if messages else ""
        m = re.search(r"Valid letters:\s*([A-J](?:\s*,\s*[A-J])*)", last)
        letters = re.findall(r"[A-J]", m.group(1)) if m else list("ABCD")
        h = int(hashlib.md5(last.encode("utf-8")).hexdigest(), 16)
        letter = letters[h % len(letters)]
        return f"(mock) placeholder reasoning.\n[ANSWER]: ({letter})", {"prompt_tokens": 0, "completion_tokens": 0}


class QuestionSession:
    """문항 1개를 푸는 동안 쓰는 호출 창구. 파이프라인은 이것만 받는다."""

    def __init__(self, backend, question_id: str, max_calls: int = config.MAX_CALLS_PER_QUESTION,
                 defaults: dict | None = None):
        self.backend = backend
        self.question_id = question_id
        self.max_calls = max_calls
        self.defaults = dict(config.DEFAULT_GEN_PARAMS if defaults is None else defaults)
        self.records: list[CallRecord] = []
        self.calls_used = 0

    @property
    def remaining(self) -> int:
        return self.max_calls - self.calls_used

    def chat(self, messages: list[dict] | str, tag: str = "answer", **params) -> str:
        """LLM 1회 호출. messages가 문자열이면 user 메시지 하나로 감싼다."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.calls_used >= self.max_calls:
            raise BudgetExceeded(f"{self.question_id}: 문항당 LLM 호출 한도 {self.max_calls}회 초과 (tag={tag})")

        p = {**self.defaults, **params}
        p = {k: v for k, v in p.items() if v is not None}
        if "model" in p:
            raise ValueError("model은 config.CHAT_MODEL로 고정입니다. 인자로 넘기지 마세요.")
        for key in ("max_tokens", "max_completion_tokens"):
            if key in p and int(p[key]) > config.MAX_TOKENS_LIMIT:
                raise ValueError(f"{key}={p[key]} > {config.MAX_TOKENS_LIMIT} (규칙 위반)")
        if int(p.get("n", 1)) != 1:
            raise ValueError("n>1 금지: 여러 샘플이 필요하면 chat()을 여러 번 호출하세요 (호출 수에 포함).")

        self.calls_used += 1
        t0 = time.perf_counter()
        try:
            text, usage = self.backend.complete(messages, **p)
        except Exception as e:  # 기록만 남기고 위로 전달
            self.records.append(CallRecord(self.question_id, self.calls_used, tag, p, messages, None, None,
                                           round(time.perf_counter() - t0, 3), error=repr(e)))
            raise
        self.records.append(CallRecord(self.question_id, self.calls_used, tag, p, messages, text, usage,
                                       round(time.perf_counter() - t0, 3)))
        return text


def make_backend(mock: bool = False):
    return MockChat() if mock else UpstageChat()
