"""Semantic Transcript Chunking with Utterance Boundary Preservation and Privacy Validation.

Provides:
- Strict rejection of unredacted transcript versions (ORIGINAL, CLEANED).
- Organization-level tenant security checks.
- Utterance-boundary preservation preventing clipped phrases or detached timestamps.
- Target ~250 token passages with configurable overlap and maximum bounds.
- Graceful handling of individual long utterances via sentence-level sub-chunking.
- Exact preservation of original text, segment UUIDs, and start/end timestamps.
- Support for multilingual speech (English, Hindi, Gujarati, Hinglish code-switching).
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter

from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory.exceptions import (
    UnsafeTranscriptError,
    TenantSecurityError,
    ChunkingError,
)
from app.services.memory.types import SemanticChunk
from app.core.config import settings

logger = logging.getLogger(__name__)

SAFE_TRANSCRIPT_TYPES = {
    TranscriptType.CLEANED_REDACTED.value,
    TranscriptType.REDACTED.value,
}

# Delimiters for splitting long utterances across English and Indic scripts
SENTENCE_SPLIT_REGEX = re.compile(r'([.?!।॥\n]+)')


def estimate_multilingual_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Accurately estimate token count across Latin and Indic (Devanagari, Gujarati) scripts.

    If a HuggingFace tokenizer is supplied, delegates to its encode method.
    Otherwise uses an intelligent script-aware subword estimator.
    """
    if not text or not text.strip():
        return 0

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    words = text.strip().split()
    total_tokens = 0
    for w in words:
        # Check for Devanagari (U+0900 to U+097F) or Gujarati (U+0A80 to U+0AFF)
        has_indic = any(('\u0900' <= ch <= '\u097f') or ('\u0a80' <= ch <= '\u0aff') for ch in w)
        if has_indic:
            # Indic words decompose into 2-3 subwords in SentencePiece / BPE
            total_tokens += max(2, int(len(w) / 2.5) + 1)
        else:
            # Latin words average ~1.3 tokens
            total_tokens += max(1, int(len(w) / 3.8) + 1)
    return total_tokens


def validate_transcript_privacy(
    transcript: Transcript,
    meeting: Optional[Meeting] = None,
    organization_id: Optional[str] = None,
) -> None:
    """Enforce privacy-safe transcript constraints and tenant isolation.

    Raises:
        UnsafeTranscriptError: If transcript is ORIGINAL or CLEANED.
        TenantSecurityError: If meeting or organization boundaries do not match.
    """
    if not transcript:
        raise UnsafeTranscriptError("Transcript is required for chunking.")

    # Strict privacy requirement: only REDACTED or CLEANED_REDACTED versions
    if transcript.transcript_type not in SAFE_TRANSCRIPT_TYPES:
        raise UnsafeTranscriptError(
            f"Transcript '{transcript.id}' is of type '{transcript.transcript_type}'. "
            "Memory indexing and semantic chunking strictly require privacy-safe transcripts "
            "(REDACTED or CLEANED_REDACTED). Unredacted transcripts (ORIGINAL, CLEANED) cannot be indexed."
        )

    # Tenant boundary validation
    if meeting:
        if transcript.meeting_id != meeting.id:
            raise TenantSecurityError(
                f"Transcript '{transcript.id}' does not belong to meeting '{meeting.id}'."
            )
        if organization_id and meeting.organization_id != organization_id:
            raise TenantSecurityError(
                f"Meeting '{meeting.id}' belongs to organization '{meeting.organization_id}', "
                f"which does not match authorized tenant '{organization_id}'."
            )


class SemanticTranscriptChunker:
    """Chunks transcript segments into coherent semantic passages bounded by token limits."""

    def __init__(
        self,
        target_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.target_tokens = target_tokens or getattr(settings, "MEMORY_CHUNK_TARGET_TOKENS", 250)
        self.max_tokens = max_tokens or getattr(settings, "MEMORY_CHUNK_MAX_TOKENS", 350)
        self.overlap_tokens = overlap_tokens or getattr(settings, "MEMORY_CHUNK_OVERLAP_TOKENS", 30)
        self.tokenizer = tokenizer

        if self.target_tokens > self.max_tokens:
            raise ValueError(f"target_tokens ({self.target_tokens}) cannot exceed max_tokens ({self.max_tokens}).")

    def _split_long_utterance_text(self, text: str) -> List[str]:
        """Split a single utterance that exceeds max_tokens into clean sentence-level units."""
        raw_parts = SENTENCE_SPLIT_REGEX.split(text)
        sentences: List[str] = []
        current = ""
        for part in raw_parts:
            current += part
            if SENTENCE_SPLIT_REGEX.match(part):
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        # If regex did not split (e.g. no punctuation), split by words
        if len(sentences) <= 1:
            words = text.split()
            sub_chunks = []
            cur_words: List[str] = []
            cur_tok = 0
            for w in words:
                w_tok = estimate_multilingual_tokens(w, tokenizer=self.tokenizer)
                if cur_tok + w_tok > self.target_tokens and cur_words:
                    sub_chunks.append(" ".join(cur_words))
                    cur_words = [w]
                    cur_tok = w_tok
                else:
                    cur_words.append(w)
                    cur_tok += w_tok
            if cur_words:
                sub_chunks.append(" ".join(cur_words))
            return sub_chunks

        # Group sentences into units up to target_tokens
        grouped: List[str] = []
        curr_group: List[str] = []
        curr_tokens = 0
        for sent in sentences:
            s_tok = estimate_multilingual_tokens(sent, tokenizer=self.tokenizer)
            if curr_tokens + s_tok > self.target_tokens and curr_group:
                grouped.append(" ".join(curr_group))
                curr_group = [sent]
                curr_tokens = s_tok
            else:
                curr_group.append(sent)
                curr_tokens += s_tok
        if curr_group:
            grouped.append(" ".join(curr_group))
        return grouped

    def chunk_segments(
        self,
        segments: List[TranscriptSegment],
        meeting_id: str,
        organization_id: str,
        transcript_id: str,
    ) -> List[SemanticChunk]:
        """Group transcript segments into semantic chunks respecting utterance boundaries.

        Returns an ordered list of SemanticChunk instances.
        """
        if not segments:
            return []

        # Ensure segments are sorted in sequence order
        sorted_segments = sorted(segments, key=lambda s: (s.sequence_order, s.start_time))
        chunks: List[SemanticChunk] = []

        current_segments: List[TranscriptSegment] = []
        current_token_count = 0

        def emit_chunk(segs: List[TranscriptSegment], is_split: bool = False, text_override: Optional[str] = None):
            nonlocal chunks
            if not segs and not text_override:
                return

            chunk_text = text_override if text_override is not None else " ".join(s.text.strip() for s in segs if s.text and s.text.strip())
            if not chunk_text.strip():
                return

            tok_count = estimate_multilingual_tokens(chunk_text, tokenizer=self.tokenizer)
            min_start = min(s.start_time for s in segs)
            max_end = max(s.end_time for s in segs)
            seg_ids = [s.id for s in segs]

            # Language breakdown
            lang_counts = Counter(s.language_code for s in segs)
            unique_langs = sorted(list(lang_counts.keys()))
            dominant = lang_counts.most_common(1)[0][0] if lang_counts else "en"

            chunk = SemanticChunk(
                chunk_index=len(chunks),
                meeting_id=meeting_id,
                organization_id=organization_id,
                transcript_id=transcript_id,
                segment_ids=seg_ids,
                text=chunk_text,
                start_time=min_start,
                end_time=max_end,
                language_codes=unique_langs,
                dominant_language=dominant,
                token_count=tok_count,
                is_split_utterance=is_split,
            )
            chunks.append(chunk)

        for seg in sorted_segments:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            seg_tokens = estimate_multilingual_tokens(seg_text, tokenizer=self.tokenizer)

            # Case 1: Single segment by itself exceeds max_tokens -> split into sub-chunks
            if seg_tokens > self.max_tokens:
                # Flush current accumulator first
                if current_segments:
                    emit_chunk(current_segments)
                    current_segments = []
                    current_token_count = 0

                # Split long utterance
                sub_texts = self._split_long_utterance_text(seg_text)
                for sub_t in sub_texts:
                    emit_chunk([seg], is_split=True, text_override=sub_t)
                continue

            # Case 2: Adding this segment exceeds target tokens -> emit current and start new chunk
            if current_segments and (current_token_count + seg_tokens > self.target_tokens):
                emit_chunk(current_segments)

                # Overlap logic: carry over the last segment if its token count <= overlap_tokens
                last_seg = current_segments[-1]
                last_tokens = estimate_multilingual_tokens(last_seg.text, tokenizer=self.tokenizer)

                if self.overlap_tokens > 0 and last_tokens <= self.overlap_tokens and len(current_segments) > 1:
                    current_segments = [last_seg, seg]
                    current_token_count = last_tokens + seg_tokens
                else:
                    current_segments = [seg]
                    current_token_count = seg_tokens
            else:
                current_segments.append(seg)
                current_token_count += seg_tokens

        # Flush final remaining accumulator
        if current_segments:
            emit_chunk(current_segments)

        return chunks


def chunk_privacy_safe_transcript(
    transcript: Transcript,
    segments: List[TranscriptSegment],
    meeting: Meeting,
    target_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
    tokenizer: Optional[Any] = None,
) -> List[SemanticChunk]:
    """Helper orchestrating privacy validation and semantic chunking for a transcript."""
    validate_transcript_privacy(
        transcript=transcript,
        meeting=meeting,
        organization_id=meeting.organization_id,
    )

    chunker = SemanticTranscriptChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer=tokenizer,
    )

    return chunker.chunk_segments(
        segments=segments,
        meeting_id=meeting.id,
        organization_id=meeting.organization_id,
        transcript_id=transcript.id,
    )
