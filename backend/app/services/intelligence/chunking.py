"""Segment chunking, resource budget enforcement, and multi-chunk result merging."""

from typing import List, Tuple, Optional, Dict, Set
from app.models.transcription import TranscriptSegment
from app.schemas.intelligence import (
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
    StructuredExtractionPayload,
)


def chunk_transcript_segments(
    segments: List[TranscriptSegment],
    max_chars: int = 40000,
    max_chunks: int = 50,
) -> Tuple[List[List[TranscriptSegment]], bool, Optional[str]]:
    """Group segments into character-bounded chunks without splitting individual utterances.

    Returns:
        (chunks, is_partial, partial_reason)
    """
    if not segments:
        return [], False, None

    chunks: List[List[TranscriptSegment]] = []
    current_chunk: List[TranscriptSegment] = []
    current_char_count = 0

    is_partial = False
    partial_reason: Optional[str] = None

    for seg in segments:
        seg_len = len(seg.text or "") + 50  # Add approximate overhead for ID/metadata formatting

        # If adding this segment exceeds max_chars and current chunk is not empty, start a new chunk
        if current_chunk and (current_char_count + seg_len > max_chars):
            chunks.append(current_chunk)
            current_chunk = []
            current_char_count = 0

            # Check max_chunks limit
            if len(chunks) >= max_chunks:
                is_partial = True
                processed_count = sum(len(c) for c in chunks)
                partial_reason = (
                    f"Reached configured limit of {max_chunks} chunks. "
                    f"Processed {processed_count} of {len(segments)} segments."
                )
                break

        current_chunk.append(seg)
        current_char_count += seg_len

    # Add final remaining chunk if within chunk limit
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(current_chunk)
    elif current_chunk and len(chunks) >= max_chunks:
        is_partial = True
        processed_count = sum(len(c) for c in chunks)
        partial_reason = (
            f"Reached configured limit of {max_chunks} chunks. "
            f"Processed {processed_count} of {len(segments)} segments."
        )

    return chunks, is_partial, partial_reason


def merge_extracted_payloads(
    payloads: List[StructuredExtractionPayload],
) -> StructuredExtractionPayload:
    """Merge extracted payloads from multiple chunks, deduplicating decisions, actions, and risks."""
    if not payloads:
        return StructuredExtractionPayload(
            executive_summary="No content available.",
            detailed_summary=None,
            key_takeaways=[],
            topics=[],
            decisions=[],
            action_items=[],
            risks=[],
            unresolved_questions=[],
        )

    if len(payloads) == 1:
        return payloads[0]

    # Combine summaries
    summary_parts = [p.executive_summary.strip() for p in payloads if p.executive_summary and p.executive_summary.strip()]
    merged_executive = " ".join(summary_parts)

    detailed_parts = [p.detailed_summary.strip() for p in payloads if p.detailed_summary and p.detailed_summary.strip()]
    merged_detailed = "\n\n".join(detailed_parts) if detailed_parts else None

    # Merge key takeaways with deduplication
    merged_takeaways: List[str] = []
    seen_takeaways: Set[str] = set()
    for p in payloads:
        if p.key_takeaways:
            for item in p.key_takeaways:
                clean = item.strip()
                norm = clean.lower()
                if norm not in seen_takeaways:
                    seen_takeaways.add(norm)
                    merged_takeaways.append(clean)

    # Merge topics
    merged_topics: List[TopicCreate] = []
    seen_topics: Set[str] = set()
    for p in payloads:
        for t in p.topics:
            norm = t.title.strip().lower()
            if norm not in seen_topics:
                seen_topics.add(norm)
                merged_topics.append(t)

    # Merge decisions with deduplication
    merged_decisions: List[DecisionCreate] = []
    seen_decisions: Set[str] = set()
    for p in payloads:
        for d in p.decisions:
            norm = d.description.strip().lower()
            if norm not in seen_decisions:
                seen_decisions.add(norm)
                merged_decisions.append(d)

    # Merge action items with deduplication by (description, owner)
    merged_actions: List[ActionItemCreate] = []
    seen_actions: Set[Tuple[str, str]] = set()
    for p in payloads:
        for a in p.action_items:
            norm_desc = a.description.strip().lower()
            norm_owner = (a.owner_name or "").strip().lower()
            key = (norm_desc, norm_owner)
            if key not in seen_actions:
                seen_actions.add(key)
                merged_actions.append(a)

    # Merge risks with deduplication
    merged_risks: List[RiskCreate] = []
    seen_risks: Set[str] = set()
    for p in payloads:
        for r in p.risks:
            norm = r.description.strip().lower()
            if norm not in seen_risks:
                seen_risks.add(norm)
                merged_risks.append(r)

    # Merge questions with deduplication
    merged_questions: List[UnresolvedQuestionCreate] = []
    seen_questions: Set[str] = set()
    for p in payloads:
        for q in p.unresolved_questions:
            norm = q.question_text.strip().lower()
            if norm not in seen_questions:
                seen_questions.add(norm)
                merged_questions.append(q)

    return StructuredExtractionPayload(
        executive_summary=merged_executive,
        detailed_summary=merged_detailed,
        key_takeaways=merged_takeaways,
        topics=merged_topics,
        decisions=merged_decisions,
        action_items=merged_actions,
        risks=merged_risks,
        unresolved_questions=merged_questions,
    )
