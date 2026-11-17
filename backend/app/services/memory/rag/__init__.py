"""RAG Question Answering Subsystem for Multilingual Meeting Intelligence."""

from app.services.memory.rag.types import (
    AbstentionStatus,
    Citation,
    EvidenceItem,
    RetrievalDiagnostics,
    RawAnswerResult,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.memory.rag.context import (
    ContextBuilder,
    DEFAULT_MAX_CONTEXT_CHARS,
)
from app.services.memory.rag.providers import (
    AnswerProvider,
    DeterministicAnswerProvider,
    OllamaAnswerProvider,
)
from app.services.memory.rag.verification import (
    CitationVerifier,
)
from app.services.memory.rag.engine import (
    RAGEngine,
)

__all__ = [
    "AbstentionStatus",
    "Citation",
    "EvidenceItem",
    "RetrievalDiagnostics",
    "RawAnswerResult",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "ContextBuilder",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "AnswerProvider",
    "DeterministicAnswerProvider",
    "OllamaAnswerProvider",
    "CitationVerifier",
    "RAGEngine",
]
