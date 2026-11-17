"""Data structures and Pydantic DTOs for Semantic Chunking and Multilingual Embeddings."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class SemanticChunk(BaseModel):
    """Represents a discrete semantic chunk constructed from one or more privacy-safe transcript segments."""
    chunk_index: int = Field(..., ge=0, description="Sequential zero-based index of this chunk within the transcript")
    meeting_id: str = Field(..., min_length=1, description="UUID of the parent meeting")
    organization_id: str = Field(..., min_length=1, description="Tenant organization boundary UUID")
    transcript_id: str = Field(..., min_length=1, description="UUID of the source privacy-safe transcript")
    segment_ids: List[str] = Field(..., min_length=1, description="Ordered UUIDs of transcript segments included in this chunk")
    text: str = Field(..., min_length=1, description="Aggregated text from the source segments")
    start_time: float = Field(..., ge=0.0, description="Earliest start timestamp in seconds from source segments")
    end_time: float = Field(..., ge=0.0, description="Latest end timestamp in seconds from source segments")
    language_codes: List[str] = Field(default_factory=list, description="Unique language codes represented in this chunk")
    dominant_language: str = Field(default="en", description="Dominant language code for this chunk")
    token_count: int = Field(..., ge=0, description="Estimated or exact token count of the chunk text")
    is_split_utterance: bool = Field(default=False, description="True if this chunk represents a sub-slice of a single long utterance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional indexing metadata")

    model_config = ConfigDict(from_attributes=True)


class ChunkEmbeddingResult(BaseModel):
    """Container holding a SemanticChunk and its normalized dense embedding vector."""
    chunk: SemanticChunk = Field(..., description="The semantic chunk payload")
    embedding: List[float] = Field(..., description="Normalized dense float vector representation")
    dimension: int = Field(default=384, ge=1, description="Vector dimension of the embedding")
    model_name: str = Field(..., description="Identifier of the model used to produce the embedding")

    model_config = ConfigDict(from_attributes=True)


class KeywordSearchResult(BaseModel):
    """Result from SQLite FTS5 keyword full-text search with BM25 ranking."""
    chunk_id: str = Field(..., description="UUID of the matched chunk record")
    bm25_score: float = Field(..., description="Raw BM25 score from SQLite FTS5")
    bm25_rank: int = Field(..., ge=1, description="1-based ranking within keyword search results")
    text: str = Field(..., description="Privacy-safe chunk text")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    organization_id: str = Field(..., description="Tenant organization boundary UUID")
    transcript_id: str = Field(..., description="UUID of the source privacy-safe transcript")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    segment_ids: List[str] = Field(default_factory=list, description="Source transcript segment UUIDs")
    language_codes: List[str] = Field(default_factory=list, description="Language codes represented in chunk")
    dominant_language: str = Field(default="en", description="Dominant language code")
    token_count: int = Field(..., ge=0, description="Token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(from_attributes=True)


class SemanticSearchResult(BaseModel):
    """Result from dense vector cosine similarity search."""
    chunk_id: str = Field(..., description="UUID of the matched chunk record")
    cosine_similarity: float = Field(..., description="Cosine similarity score between query and stored vector")
    semantic_rank: int = Field(..., ge=1, description="1-based ranking within semantic search results")
    text: str = Field(..., description="Privacy-safe chunk text")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    organization_id: str = Field(..., description="Tenant organization boundary UUID")
    transcript_id: str = Field(..., description="UUID of the source privacy-safe transcript")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    segment_ids: List[str] = Field(default_factory=list, description="Source transcript segment UUIDs")
    language_codes: List[str] = Field(default_factory=list, description="Language codes represented in chunk")
    dominant_language: str = Field(default="en", description="Dominant language code")
    token_count: int = Field(..., ge=0, description="Token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(from_attributes=True)


class HybridSearchResult(BaseModel):
    """Result combining keyword and semantic rankings via Reciprocal Rank Fusion (RRF)."""
    chunk_id: str = Field(..., description="UUID of the matched chunk record")
    rrf_score: float = Field(..., ge=0.0, description="Reciprocal Rank Fusion score: sum(1 / (k + rank))")
    keyword_rank: Optional[int] = Field(default=None, description="1-based keyword rank if matched in FTS5")
    keyword_score: Optional[float] = Field(default=None, description="Raw BM25 score from FTS5 if matched")
    semantic_rank: Optional[int] = Field(default=None, description="1-based semantic rank if matched in vector search")
    semantic_score: Optional[float] = Field(default=None, description="Cosine similarity score if matched")
    text: str = Field(..., description="Privacy-safe chunk text")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    organization_id: str = Field(..., description="Tenant organization boundary UUID")
    transcript_id: str = Field(..., description="UUID of the source privacy-safe transcript")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk index")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    segment_ids: List[str] = Field(default_factory=list, description="Source transcript segment UUIDs")
    language_codes: List[str] = Field(default_factory=list, description="Language codes represented in chunk")
    dominant_language: str = Field(default="en", description="Dominant language code")
    token_count: int = Field(..., ge=0, description="Token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(from_attributes=True)
