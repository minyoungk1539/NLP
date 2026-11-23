"""새 파이프라인 템플릿 (복사해서 쓰세요. 이 파일 자체는 등록되지 않음).

1) 이 파일을 rag/pipelines/my_method.py 로 복사하고 클래스 이름·name을 바꾼다
2) rag/pipelines/__init__.py 의 PIPELINES 에 추가한다
3) python run.py --pipeline my_method --input data/dev/dev_all.csv --output out.csv

규칙 체크
- LLM 호출은 llm.chat(...)만 사용 (직접 OpenAI 클라이언트 호출 금지 → 호출 수 집계가 깨짐)
- 한 문항에서 llm.chat은 최대 5번. 남은 횟수는 llm.remaining
- 검색 결과를 pred.info["retrieved"] = [hit.brief() ...] 로 남기면 검색 적중률이 자동 집계됨
"""
from __future__ import annotations

from .. import prompts
from ..answer import extract_letter
from .base import Pipeline, Prediction
from .rag_baseline import RAGBaseline


class MyMethod(RAGBaseline):          # 검색이 필요 없으면 Pipeline을 상속
    name = "my_method"

    def predict(self, q, llm) -> Prediction:
        hits = self.retrieve(q.render())

        # 예: 1차 답변 → 자기검증 (LLM 2회)
        draft = llm.chat([
            {"role": "system", "content": prompts.SYSTEM_RAG},
            {"role": "user", "content": prompts.user_rag(q, hits)},
        ], tag="draft")
        final = llm.chat([
            {"role": "system", "content": prompts.SYSTEM_RAG},
            {"role": "user", "content": prompts.user_rag(q, hits)},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": "Check your reasoning against the context and give the final answer."},
        ], tag="verify")

        letter, method = extract_letter(final, q.letters)
        if letter is None:                                  # 검증 단계 추출 실패 시 1차 답 사용
            letter, method = extract_letter(draft, q.letters)
        return Prediction(letter, final, method, info={"retrieved": [h.brief() for h in hits]})
