"""파이프라인 등록부. 새 파이프라인은 여기에 한 줄 추가."""
from .base import Pipeline, Prediction, ask_once
from .llm_only import LLMOnly
from .rag_baseline import RAGBaseline
from .routed import RoutedRAG

PIPELINES: dict[str, type[Pipeline]] = {
    cls.name: cls
    for cls in (
        LLMOnly,
        RAGBaseline,
        RoutedRAG,
        # MyMethod,   ← 새 파이프라인 등록
    )
}

__all__ = ["PIPELINES", "Pipeline", "Prediction", "ask_once"]
