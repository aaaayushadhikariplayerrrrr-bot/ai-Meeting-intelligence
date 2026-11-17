"""Conservative deterministic rule-based extractor for multilingual meeting intelligence."""

import re
from typing import List, Optional, Tuple
from app.models.transcription import TranscriptSegment
from app.models.intelligence import ActionItemStatus, RiskSeverity, QuestionStatus
from app.schemas.intelligence import (
    TopicCreate,
    DecisionCreate,
    ActionItemCreate,
    RiskCreate,
    UnresolvedQuestionCreate,
    StructuredExtractionPayload,
)


class DeterministicExtractor:
    """Heuristic rule-based extractor providing reliable offline fallback intelligence extraction."""

    # Decision markers (English, Hindi, Gujarati, Hinglish)
    DECISION_PATTERNS = [
        re.compile(r"\b(?:we\s+(?:have\s+)?decided\s+to|decision\s+is\s+to|agreed\s+(?:to|on|that)|approved\s+the|consensus\s+is\s+to|finalized\s+(?:to|on)|we\s+will\s+go\s+with)\b", re.IGNORECASE),
        re.compile(r"\b(?:decide\s+kiya|faisla\s+liya|tay\s+hua|final\s+karyu|nakki\s+karyu|pakka\s+kiya)\b", re.IGNORECASE),
    ]

    # Action item commitment & assignment markers
    ACTION_PATTERNS = [
        re.compile(r"\b(?:i\s+will|i'll|i\s+am\s+going\s+to|action\s+item\s+for|assigned\s+to|please\s+ensure|will\s+take\s+care\s+of|will\s+handle|responsible\s+for)\b", re.IGNORECASE),
        re.compile(r"\b(?:main\s+karunga|hum\s+karenge|tame\s+karjo|aa\s+kaam\s+karjo)\b", re.IGNORECASE),
    ]

    # Risk & blocker markers
    RISK_PATTERNS = [
        re.compile(r"\b(?:risk\s+is|risk\s+of|major\s+concern|blocker\s+is|blocked\s+by|potential\s+issue|impediment|bottleneck|vram\s+might\s+be\s+tight|overheating|out\s+of\s+memory)\b", re.IGNORECASE),
        re.compile(r"\b(?:khatra\s+hai|problem\s+aa\s+sakti|takleef\s+chhe|blocker\s+chhe)\b", re.IGNORECASE),
    ]

    # Unresolved question markers
    QUESTION_PATTERNS = [
        re.compile(r"\b(?:can\s+we|how\s+(?:will|do|can|should)|should\s+we|who\s+will|what\s+if|is\s+there\s+any|open\s+question|unresolved)\b", re.IGNORECASE),
        re.compile(r"\b(?:kaise\s+hoga|kya\s+karenge|kem\s+thashe|su\s+karsu)\b", re.IGNORECASE),
    ]

    # Simple deadline detection
    DEADLINE_PATTERN = re.compile(
        r"\b(?:by\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|eod|end\s+of\s+day|next\s+week|next\s+sprint))\b",
        re.IGNORECASE,
    )

    def extract(
        self,
        segments: List[TranscriptSegment],
        meeting_title: str = "Meeting",
        output_language: str = "en",
    ) -> StructuredExtractionPayload:
        """Extract structured intelligence deterministically from transcript segments."""
        if not segments:
            return StructuredExtractionPayload(
                executive_summary="Empty transcript provided; no intelligence could be extracted.",
                detailed_summary=None,
                key_takeaways=[],
                topics=[],
                decisions=[],
                action_items=[],
                risks=[],
                unresolved_questions=[],
            )

        decisions: List[DecisionCreate] = []
        action_items: List[ActionItemCreate] = []
        risks: List[RiskCreate] = []
        questions: List[UnresolvedQuestionCreate] = []

        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue

            # 1. Detect Decisions
            if any(p.search(text) for p in self.DECISION_PATTERNS):
                clean_desc = self._clean_sentence(text)
                speaker_name = self._resolve_speaker_name(seg)
                decisions.append(
                    DecisionCreate(
                        description=clean_desc,
                        rationale_or_context=f"Agreed during discussion around {seg.start_time:.1f}s",
                        decided_by=speaker_name,
                        evidence_segment_id=seg.id,
                        timestamp_seconds=seg.start_time,
                    )
                )

            # 2. Detect Action Items
            if any(p.search(text) for p in self.ACTION_PATTERNS):
                clean_desc = self._clean_sentence(text)
                owner = self._extract_owner(text, seg)
                action_items.append(
                    ActionItemCreate(
                        description=clean_desc,
                        owner_name=owner,
                        status=ActionItemStatus.PENDING,
                        evidence_segment_id=seg.id,
                        timestamp_seconds=seg.start_time,
                    )
                )

            # 3. Detect Risks
            if any(p.search(text) for p in self.RISK_PATTERNS):
                clean_desc = self._clean_sentence(text)
                severity = RiskSeverity.HIGH if any(w in text.lower() for w in ["critical", "blocker", "tight", "out of memory"]) else RiskSeverity.MEDIUM
                risks.append(
                    RiskCreate(
                        description=clean_desc,
                        severity=severity,
                        mitigation_suggestion="Review during implementation sync.",
                        evidence_segment_id=seg.id,
                    )
                )

            # 4. Detect Unresolved Questions
            if text.endswith("?") and any(p.search(text) for p in self.QUESTION_PATTERNS):
                # Filter out short rhetorical questions like "Right?", "Okay?"
                if len(text.split()) > 3:
                    clean_desc = self._clean_sentence(text)
                    speaker_name = self._resolve_speaker_name(seg)
                    questions.append(
                        UnresolvedQuestionCreate(
                            question_text=clean_desc,
                            raised_by=speaker_name,
                            status=QuestionStatus.OPEN,
                            evidence_segment_id=seg.id,
                        )
                    )

        # 5. Formulate Topics (partition into 1-4 chronological blocks)
        topics = self._partition_topics(segments, meeting_title)

        # 6. Formulate Executive & Detailed Summaries
        exec_summary, detailed_summary, takeaways = self._synthesize_summaries(
            meeting_title=meeting_title,
            segments=segments,
            decisions=decisions,
            action_items=action_items,
            risks=risks,
            topics=topics,
        )

        return StructuredExtractionPayload(
            executive_summary=exec_summary,
            detailed_summary=detailed_summary,
            key_takeaways=takeaways,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            risks=risks,
            unresolved_questions=questions,
        )

    def _resolve_speaker_name(self, seg: TranscriptSegment) -> str:
        """Derive readable speaker name from speaker relation or label."""
        if seg.speaker:
            return seg.speaker.display_name or seg.speaker.speaker_label
        return "Participant"

    def _extract_owner(self, text: str, seg: TranscriptSegment) -> Optional[str]:
        """Extract owner name if explicitly assigned, or derive from speaker if self-committing."""
        # Check explicit "assigned to X"
        match = re.search(r"\bassigned\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Check self-commitment ("I will...", "I'll...")
        if re.search(r"\b(?:i\s+will|i'll|i\s+am\s+going\s+to|main\s+karunga)\b", text, re.IGNORECASE):
            return self._resolve_speaker_name(seg)

        return None

    def _clean_sentence(self, text: str) -> str:
        """Clean leading/trailing whitespace and punctuation."""
        s = text.strip()
        if s and not s[-1] in ".?!":
            s += "."
        return s

    def _partition_topics(self, segments: List[TranscriptSegment], meeting_title: str) -> List[TopicCreate]:
        """Group segments into 1-4 structured chronological topic segments."""
        if not segments:
            return []

        total_segments = len(segments)
        num_topics = 1 if total_segments <= 3 else (2 if total_segments <= 8 else 3)
        chunk_size = max(1, total_segments // num_topics)

        topics: List[TopicCreate] = []
        for i in range(num_topics):
            start_idx = i * chunk_size
            end_idx = total_segments if i == num_topics - 1 else (i + 1) * chunk_size
            sub_segs = segments[start_idx:end_idx]
            if not sub_segs:
                continue

            t_start = sub_segs[0].start_time
            t_end = sub_segs[-1].end_time

            # Generate concise descriptive title
            if i == 0:
                t_title = f"{meeting_title} - Opening & Context"
            elif i == num_topics - 1:
                t_title = f"{meeting_title} - Action Items & Wrap-Up"
            else:
                t_title = f"{meeting_title} - Core Discussion (Part {i+1})"

            t_desc = f"Discussion spanning {t_start:.1f}s to {t_end:.1f}s covering {len(sub_segs)} utterances."
            topics.append(TopicCreate(title=t_title, description=t_desc, start_time=t_start, end_time=t_end))

        return topics

    def _synthesize_summaries(
        self,
        meeting_title: str,
        segments: List[TranscriptSegment],
        decisions: List[DecisionCreate],
        action_items: List[ActionItemCreate],
        risks: List[RiskCreate],
        topics: List[TopicCreate],
    ) -> Tuple[str, Optional[str], List[str]]:
        """Synthesize executive summary, topic breakdown, and key takeaways."""
        duration = segments[-1].end_time - segments[0].start_time if segments else 0.0
        unique_langs = sorted(list({s.language_code for s in segments if s.language_code}))
        lang_str = ", ".join(unique_langs) if unique_langs else "en"

        exec_summary = (
            f"The meeting '{meeting_title}' spanned {duration:.1f} seconds across {len(segments)} utterances "
            f"in {lang_str}. Key discussions centered on {len(topics)} main topic blocks. "
            f"The team recorded {len(decisions)} explicit decisions and identified {len(action_items)} action items."
        )

        detailed_parts = [f"### Meeting Overview\n{exec_summary}"]
        for t in topics:
            detailed_parts.append(f"#### {t.title} ({t.start_time:.1f}s - {t.end_time:.1f}s)\n{t.description}")

        detailed_summary = "\n\n".join(detailed_parts)

        takeaways = []
        if decisions:
            takeaways.append(f"Decided: {decisions[0].description}")
        if action_items:
            takeaways.append(f"Action: {action_items[0].description}")
        if risks:
            takeaways.append(f"Risk Noted: {risks[0].description}")
        if not takeaways and segments:
            takeaways.append(f"Discussion concluded with {len(segments)} segments reviewed.")

        return exec_summary, detailed_summary, takeaways
