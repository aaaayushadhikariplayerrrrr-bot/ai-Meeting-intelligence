"""Evidence grounding and transcript segment validation."""

import logging
from typing import Dict, List, Optional, Tuple, Any
from app.models.transcription import TranscriptSegment
from app.schemas.intelligence import (
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
    StructuredExtractionPayload,
)

logger = logging.getLogger(__name__)


def ground_and_verify_evidence(
    payload: StructuredExtractionPayload,
    segments: List[TranscriptSegment],
) -> StructuredExtractionPayload:
    """Validate all referenced segment IDs against verified transcript segments and derive exact timestamps.

    STRICT EVIDENCE ATTRIBUTION RULES:
    1. Every evidence_segment_id MUST match an actual segment from the supplied transcript.
    2. Unknown, malformed, or hallucinated segment IDs are stripped (set to None) to prevent bogus evidence links.
    3. Timestamps are derived strictly from the verified segment's start_time in the database, never from LLM output.
    4. If no supporting segment can be verified, the entity is retained only as an ungrounded item (evidence_segment_id=None, timestamp_seconds=None).
    """
    valid_segments_map: Dict[str, TranscriptSegment] = {s.id: s for s in segments}

    # 1. Ground Decisions
    grounded_decisions: List[DecisionCreate] = []
    for d in payload.decisions:
        seg_id = d.evidence_segment_id
        if seg_id and seg_id in valid_segments_map:
            verified_seg = valid_segments_map[seg_id]
            d.evidence_segment_id = verified_seg.id
            d.timestamp_seconds = verified_seg.start_time
        else:
            if seg_id:
                logger.debug(f"Stripped unverified/fabricated evidence segment ID '{seg_id}' on decision.")
            d.evidence_segment_id = None
            d.timestamp_seconds = None
        grounded_decisions.append(d)

    # 2. Ground Action Items
    grounded_actions: List[ActionItemCreate] = []
    for a in payload.action_items:
        seg_id = a.evidence_segment_id
        if seg_id and seg_id in valid_segments_map:
            verified_seg = valid_segments_map[seg_id]
            a.evidence_segment_id = verified_seg.id
            a.timestamp_seconds = verified_seg.start_time
        else:
            if seg_id:
                logger.debug(f"Stripped unverified/fabricated evidence segment ID '{seg_id}' on action item.")
            a.evidence_segment_id = None
            a.timestamp_seconds = None
        grounded_actions.append(a)

    # 3. Ground Risks
    grounded_risks: List[RiskCreate] = []
    for r in payload.risks:
        seg_id = r.evidence_segment_id
        if seg_id and seg_id in valid_segments_map:
            verified_seg = valid_segments_map[seg_id]
            r.evidence_segment_id = verified_seg.id
        else:
            if seg_id:
                logger.debug(f"Stripped unverified/fabricated evidence segment ID '{seg_id}' on risk.")
            r.evidence_segment_id = None
        grounded_risks.append(r)

    # 4. Ground Unresolved Questions
    grounded_questions: List[UnresolvedQuestionCreate] = []
    for q in payload.unresolved_questions:
        seg_id = q.evidence_segment_id
        if seg_id and seg_id in valid_segments_map:
            verified_seg = valid_segments_map[seg_id]
            q.evidence_segment_id = verified_seg.id
        else:
            if seg_id:
                logger.debug(f"Stripped unverified/fabricated evidence segment ID '{seg_id}' on unresolved question.")
            q.evidence_segment_id = None
        grounded_questions.append(q)

    # 5. Sanitize Topic Boundaries
    if segments:
        min_start = segments[0].start_time
        max_end = segments[-1].end_time
        for t in payload.topics:
            if t.start_time is not None:
                t.start_time = max(min_start, t.start_time)
            if t.end_time is not None:
                t.end_time = min(max_end, t.end_time)

    return StructuredExtractionPayload(
        executive_summary=payload.executive_summary,
        detailed_summary=payload.detailed_summary,
        key_takeaways=payload.key_takeaways,
        topics=payload.topics,
        decisions=grounded_decisions,
        action_items=grounded_actions,
        risks=grounded_risks,
        unresolved_questions=grounded_questions,
    )
