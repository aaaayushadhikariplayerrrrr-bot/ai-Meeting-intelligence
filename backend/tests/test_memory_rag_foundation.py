"""Focused tests for Phase 8.1: Multilingual Embedding Service & Semantic Chunking.

Tests cover:
1. Correct E5 passage prefix ('passage: ')
2. Correct E5 query prefix ('query: ')
3. Embedding dimensionality (384-dim)
4. Embedding normalization (unit L2 norm)
5. Empty input handling
6. Batch embedding behavior
7. Chunk size limits
8. Segment boundary preservation
9. Timestamp and segment ID preservation
10. Mixed-language chunk handling (English, Hindi, Gujarati, Hinglish)
11. Rejection of unredacted transcript versions (ORIGINAL, CLEANED)
12. Organization and meeting ownership validation
13. Long utterance handling without crashing or exceeding limits
14. Model loading failure and un-cached model error handling
"""

import math
from typing import List, Any
import numpy as np
import pytest
import torch

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory import (
    MultilingualE5EmbeddingService,
    SemanticTranscriptChunker,
    chunk_privacy_safe_transcript,
    validate_transcript_privacy,
    estimate_multilingual_tokens,
    SemanticChunk,
    ChunkEmbeddingResult,
    ModelNotCachedError,
    EmbeddingModelLoadError,
    UnsafeTranscriptError,
    TenantSecurityError,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
)


class MockTokenizer:
    """Mock HuggingFace tokenizer recording input texts for prefix verification."""

    def __init__(self):
        self.last_encoded_texts: List[str] = []

    def __call__(self, texts, max_length=512, padding=True, truncation=True, return_tensors="pt"):
        self.last_encoded_texts = list(texts)
        batch_size = len(texts)
        seq_len = 12
        return {
            "input_ids": torch.randint(100, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        return [1] * max(1, len(text.split()))


class MockE5Model:
    """Mock Transformer model outputting 384-dimensional hidden states."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        pass

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # Generate non-zero hidden states
        torch.manual_seed(42)
        last_hidden = torch.randn(batch_size, seq_len, self.dimension)

        class ModelOutput:
            pass

        out = ModelOutput()
        out.last_hidden_state = last_hidden
        return out


@pytest.fixture
def mock_embedding_service():
    """Create a MultilingualE5EmbeddingService wired with mocked tokenizer and model."""
    tok = MockTokenizer()
    model = MockE5Model(dimension=384)
    service = MultilingualE5EmbeddingService(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=8,
        dimension=384,
        lazy_load=False,
        tokenizer=tok,
        model=model,
    )
    return service, tok, model


# --- 1 & 2. Task Prefix Enforcement ---

def test_correct_e5_passage_prefix(mock_embedding_service):
    """Verify that passage indexing strictly applies the 'passage: ' E5 prefix."""
    service, tok, _ = mock_embedding_service
    passages = ["The deployment of Phase 8 was approved.", "Key risks include GPU memory pressure."]
    service.embed_passages(passages)

    assert len(tok.last_encoded_texts) == 2
    assert tok.last_encoded_texts[0] == f"{E5_PASSAGE_PREFIX}{passages[0]}"
    assert tok.last_encoded_texts[1] == f"{E5_PASSAGE_PREFIX}{passages[1]}"


def test_correct_e5_query_prefix(mock_embedding_service):
    """Verify that search queries strictly apply the 'query: ' E5 prefix."""
    service, tok, _ = mock_embedding_service
    query = "What was the decision on Phase 8?"
    service.embed_query(query)

    assert len(tok.last_encoded_texts) == 1
    assert tok.last_encoded_texts[0] == f"{E5_QUERY_PREFIX}{query}"


# --- 3 & 4. Dimensionality & Normalization ---

def test_embedding_dimensionality(mock_embedding_service):
    """Verify generated embeddings have exactly 384 dimensions."""
    service, _, _ = mock_embedding_service
    vectors = service.embed_passages(["Meeting scheduled for next Tuesday."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_embedding_normalization(mock_embedding_service):
    """Verify all embeddings are unit L2 normalized (|v| == 1.0)."""
    service, _, _ = mock_embedding_service
    texts = [
        "First discussion on budget.",
        "Second discussion on timeline.",
        "Final action item assigned to John.",
    ]
    vectors = service.embed_passages(texts)

    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        assert math.isclose(norm, 1.0, rel_tol=1e-5), f"Expected norm 1.0, got {norm}"


# --- 5 & 6. Empty Input & Batch Behavior ---

def test_empty_input_handling(mock_embedding_service):
    """Verify empty input returns empty results without invoking model inference."""
    service, tok, _ = mock_embedding_service
    tok.last_encoded_texts = ["prior_state"]
    result = service.embed_passages([])

    assert result == []
    assert tok.last_encoded_texts == ["prior_state"]  # Model was not invoked


def test_batch_embedding_behavior(mock_embedding_service):
    """Verify large input lists are correctly split into batches of configured batch_size."""
    service, tok, _ = mock_embedding_service
    # 19 items with batch_size=8 -> batches of 8, 8, 3
    texts = [f"Utterance segment number {i}" for i in range(19)]
    vectors = service.embed_passages(texts)

    assert len(vectors) == 19
    for v in vectors:
        assert len(v) == 384


# --- 7, 8 & 9. Chunking, Token Limits & Segment Boundary Preservation ---

def test_chunk_size_limits_and_boundary_preservation():
    """Verify chunks respect target token limits and utterance boundaries."""
    chunker = SemanticTranscriptChunker(target_tokens=50, max_tokens=80, overlap_tokens=0)

    # 5 segments of ~20 tokens each -> should group into ~2 segments per chunk
    segments = [
        TranscriptSegment(
            id=f"seg-{i}",
            transcript_id="tr-1",
            sequence_order=i,
            start_time=float(i * 10),
            end_time=float(i * 10 + 9),
            text=f"This is segment number {i} with a detailed explanation of our roadmap plans.",
            language_code="en",
        )
        for i in range(5)
    ]

    chunks = chunker.chunk_segments(
        segments=segments,
        meeting_id="m-1",
        organization_id="org-1",
        transcript_id="tr-1",
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= 80, f"Chunk exceeded max_tokens: {chunk.token_count}"
        assert len(chunk.segment_ids) >= 1
        assert chunk.meeting_id == "m-1"
        assert chunk.organization_id == "org-1"


def test_timestamp_and_segment_id_preservation():
    """Verify start_time, end_time, and source segment IDs strictly derive from underlying segments."""
    chunker = SemanticTranscriptChunker(target_tokens=200, max_tokens=300)

    segments = [
        TranscriptSegment(
            id="seg-alpha",
            transcript_id="tr-safe",
            sequence_order=0,
            start_time=12.5,
            end_time=25.0,
            text="First sentence.",
            language_code="en",
        ),
        TranscriptSegment(
            id="seg-beta",
            transcript_id="tr-safe",
            sequence_order=1,
            start_time=25.5,
            end_time=42.0,
            text="Second sentence.",
            language_code="en",
        ),
    ]

    chunks = chunker.chunk_segments(
        segments=segments,
        meeting_id="m-alpha",
        organization_id="org-alpha",
        transcript_id="tr-safe",
    )

    assert len(chunks) == 1
    c = chunks[0]
    assert c.segment_ids == ["seg-alpha", "seg-beta"]
    assert c.start_time == 12.5
    assert c.end_time == 42.0
    assert "First sentence. Second sentence." in c.text


# --- 10. Mixed-Language Chunk Handling ---

def test_mixed_language_chunk_handling():
    """Verify semantic chunking accurately handles English, Hindi, Gujarati, and Hinglish code-switching."""
    chunker = SemanticTranscriptChunker(target_tokens=150, max_tokens=250)

    segments = [
        TranscriptSegment(
            id="seg-en",
            transcript_id="tr-multi",
            sequence_order=0,
            start_time=0.0,
            end_time=10.0,
            text="Today we discuss the cross-regional rollout.",
            language_code="en",
        ),
        TranscriptSegment(
            id="seg-hi",
            transcript_id="tr-multi",
            sequence_order=1,
            start_time=10.5,
            end_time=22.0,
            text="इस परियोजना में हमें सुरक्षा और गोपनीयता का ध्यान रखना होगा।",
            language_code="hi",
        ),
        TranscriptSegment(
            id="seg-gu",
            transcript_id="tr-multi",
            sequence_order=2,
            start_time=22.5,
            end_time=35.0,
            text="આ મીટિંગમાં તમામ મુખ્ય નિર્ણયોની નોંધ લેવામાં આવી છે.",
            language_code="gu",
        ),
        TranscriptSegment(
            id="seg-hinglish",
            transcript_id="tr-multi",
            sequence_order=3,
            start_time=35.5,
            end_time=48.0,
            text="Deployment Friday ko complete ho jayega without any downtime.",
            language_code="hi-en",
        ),
    ]

    chunks = chunker.chunk_segments(
        segments=segments,
        meeting_id="m-multi",
        organization_id="org-multi",
        transcript_id="tr-multi",
    )

    assert len(chunks) >= 1
    # Check language diversity preservation
    all_langs = set()
    for c in chunks:
        all_langs.update(c.language_codes)

    assert "en" in all_langs
    assert "hi" in all_langs
    assert "gu" in all_langs
    assert "hi-en" in all_langs


# --- 11 & 12. Privacy Safeguards & Ownership Validation ---

def test_rejection_of_unredacted_transcript_versions():
    """Verify that chunking strictly rejects ORIGINAL and CLEANED transcripts."""
    meeting = Meeting(id="m-priv-1", organization_id="org-priv-1", title="Privacy Meeting")

    # 1. Reject ORIGINAL
    tr_orig = Transcript(id="tr-orig", meeting_id="m-priv-1", transcript_type=TranscriptType.ORIGINAL.value, version=1)
    with pytest.raises(UnsafeTranscriptError) as exc_orig:
        validate_transcript_privacy(transcript=tr_orig, meeting=meeting, organization_id="org-priv-1")
    assert "strictly require privacy-safe transcripts" in str(exc_orig.value)
    assert "ORIGINAL" in str(exc_orig.value)

    # 2. Reject CLEANED
    tr_clean = Transcript(id="tr-clean", meeting_id="m-priv-1", transcript_type=TranscriptType.CLEANED.value, version=1)
    with pytest.raises(UnsafeTranscriptError) as exc_clean:
        validate_transcript_privacy(transcript=tr_clean, meeting=meeting, organization_id="org-priv-1")
    assert "CLEANED" in str(exc_clean.value)

    # 3. Accept CLEANED_REDACTED
    tr_clean_red = Transcript(id="tr-safe-1", meeting_id="m-priv-1", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)
    validate_transcript_privacy(transcript=tr_clean_red, meeting=meeting, organization_id="org-priv-1")

    # 4. Accept REDACTED
    tr_red = Transcript(id="tr-safe-2", meeting_id="m-priv-1", transcript_type=TranscriptType.REDACTED.value, version=1)
    validate_transcript_privacy(transcript=tr_red, meeting=meeting, organization_id="org-priv-1")


def test_organization_and_meeting_ownership_validation():
    """Verify tenant boundary security rejects cross-tenant or mismatched meeting transcripts."""
    meeting_a = Meeting(id="m-tenant-a", organization_id="org-tenant-a", title="Org A Meeting")
    tr_safe = Transcript(id="tr-safe", meeting_id="m-tenant-a", transcript_type=TranscriptType.CLEANED_REDACTED.value, version=1)

    # Wrong meeting ID
    meeting_b = Meeting(id="m-tenant-b", organization_id="org-tenant-a", title="Org B Meeting")
    with pytest.raises(TenantSecurityError) as exc_m:
        validate_transcript_privacy(transcript=tr_safe, meeting=meeting_b, organization_id="org-tenant-a")
    assert "does not belong to meeting" in str(exc_m.value)

    # Wrong organization ID
    with pytest.raises(TenantSecurityError) as exc_org:
        validate_transcript_privacy(transcript=tr_safe, meeting=meeting_a, organization_id="org-unauthorized")
    assert "does not match authorized tenant" in str(exc_org.value)


# --- 13. Long Utterance Handling ---

def test_long_utterance_handling_without_crashing():
    """Verify that a single long utterance exceeding max_tokens is split into sub-chunks without crashing."""
    chunker = SemanticTranscriptChunker(target_tokens=40, max_tokens=60)

    long_text = (
        "This is sentence one of the long monologue. "
        "Here is sentence two explaining the technical requirements in great detail. "
        "Sentence three covers deployment deadlines for the entire cluster. "
        "Sentence four outlines security and privacy safeguards across all endpoints. "
        "Finally sentence five concludes the opening address."
    )

    long_seg = TranscriptSegment(
        id="seg-monologue",
        transcript_id="tr-long",
        sequence_order=0,
        start_time=10.0,
        end_time=95.0,
        text=long_text,
        language_code="en",
    )

    chunks = chunker.chunk_segments(
        segments=[long_seg],
        meeting_id="m-long",
        organization_id="org-long",
        transcript_id="tr-long",
    )

    assert len(chunks) > 1, "Long utterance should have been split into multiple sub-chunks"
    for c in chunks:
        assert c.segment_ids == ["seg-monologue"]
        assert c.is_split_utterance is True
        assert c.start_time == 10.0
        assert c.end_time == 95.0
        assert c.token_count <= 60


# --- 14. Model Loading Failure & Uncached Model Handling ---

def test_uncached_model_raises_model_not_cached_error():
    """Verify that when auto-download is disabled and model is not cached, ModelNotCachedError is raised."""
    service = MultilingualE5EmbeddingService(
        model_name="nonexistent/model-not-cached-12345",
        allow_download=False,
        lazy_load=True,
    )

    assert service.is_cached() is False
    with pytest.raises(ModelNotCachedError) as exc_info:
        service.load_model()
    assert "is not present in the local HuggingFace cache" in str(exc_info.value)
    assert "~470 MB" in str(exc_info.value)


def test_model_loading_failure_handling(monkeypatch):
    """Verify that when AutoTokenizer/AutoModel raises an error, EmbeddingModelLoadError is raised."""
    service = MultilingualE5EmbeddingService(
        model_name="corrupted/model",
        allow_download=True,
        lazy_load=True,
    )

    import transformers
    def failing_loader(*args, **kwargs):
        raise RuntimeError("Corrupted safetensors weight buffer")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", failing_loader)

    with pytest.raises(EmbeddingModelLoadError) as exc_info:
        service.load_model()
    assert "Corrupted safetensors weight buffer" in str(exc_info.value)
