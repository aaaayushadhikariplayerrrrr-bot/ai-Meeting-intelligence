"""Exceptions for Organizational Memory, Semantic Chunking, and Multilingual Embeddings."""


class MemoryError(Exception):
    """Base exception for all organizational memory and RAG subsystem errors."""
    pass


class EmbeddingError(MemoryError):
    """Base exception for embedding operations."""
    pass


class ModelNotCachedError(EmbeddingError):
    """Raised when the requested embedding model is not present in local cache and auto-download is disabled."""
    pass


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when embedding model loading or initialization fails."""
    pass


class ChunkingError(MemoryError):
    """Base exception for transcript chunking errors."""
    pass


class UnsafeTranscriptError(ChunkingError):
    """Raised when an unredacted transcript (ORIGINAL or CLEANED) is supplied for chunking or indexing."""
    pass


class TenantSecurityError(MemoryError):
    """Raised when a transcript or meeting does not belong to the authorized tenant organization."""
    pass


class VectorStorageError(MemoryError):
    """Base exception for vector storage and chunk persistence errors."""
    pass


class VectorValidationError(VectorStorageError):
    """Raised when a vector violates shape (must be 384-dim), finite values, or unit normalization."""
    pass


class FTS5UnavailableError(MemoryError):
    """Raised when SQLite FTS5 extension is unavailable or disabled."""
    pass


class RetrievalError(MemoryError):
    """Base exception for search and hybrid retrieval errors."""
    pass


class PrivacyEnforcementError(UnsafeTranscriptError):
    """Raised when an operation attempts to store or index unredacted (ORIGINAL/CLEANED) content."""
    pass
