"""파이프라인 공통 인터페이스.

새 방법을 실험하려면 Pipeline을 상속해서 predict()만 구현하고 pipelines/__init__.py에 등록한다.
- setup()은 실행 시작 시 1번 (인덱스 로드 등)
- predict(q, llm)은 문항마다 1번. LLM 호출은 반드시 llm.chat(...)으로만 (호출 수가 자동 집계·제한됨)
- 답 글자와 함께 info에 검색 결과 등 분석용 정보를 넣어두면 runs/ 로그에 남는다
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..answer import extract_letter
from ..llm import QuestionSession
from ..parser import Question


@dataclass
class Prediction:
    letter: str | None
    raw_output: str = ""
    extract_method: str = ""
    info: dict = field(default_factory=dict)


class Pipeline:
    name = "base"

    def __init__(self, args):
        self.args = args

    def setup(self) -> None:
        pass

    def describe(self) -> dict:
        """runs/<run>/config.json에 남길 설정."""
        return {}

    def predict(self, q: Question, llm: QuestionSession) -> Prediction:
        raise NotImplementedError


def ask_once(q: Question, llm: QuestionSession, system: str, user: str, tag: str = "answer", **params) -> Prediction:
    """시스템/유저 프롬프트로 1회 호출하고 답 글자를 추출한다."""
    out = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], tag=tag, **params)
    letter, method = extract_letter(out, q.letters)
    return Prediction(letter, out, method)
