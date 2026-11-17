"""Multilingual prompt templates and context formatting for LLM intelligence extraction."""

from typing import List, Dict, Any
from app.models.transcription import TranscriptSegment


SYSTEM_PROMPT_TEMPLATE = """You are a precise, privacy-aware Enterprise Meeting Intelligence Analyst.
Your task is to analyze the provided meeting transcript and extract structured organizational intelligence.

CRITICAL INSTRUCTIONS:
1. MULTILINGUAL PROCESSING:
   The meeting transcript may contain utterances in English, Hindi (हिन्दी), Gujarati (ગુજરાતી), or Hinglish (Hindi-English code-switching).
   You must understand all languages present in the transcript and produce your output in {output_language}.
   Preserve factual names, technical terms, and proper nouns faithfully.

2. EVIDENCE GROUNDING (STRICTEST REQUIREMENT):
   Every decision, action item, risk, and unresolved question MUST reference the exact `evidence_segment_id` of the transcript segment from which it was extracted.
   Do NOT fabricate segment IDs. You must copy the exact segment ID from the provided transcript lines.

3. ANTI-FABRICATION & TRUTH PRESERVATION:
   - DECISIONS: Only include explicit, agreed-upon decisions. Do not record exploratory suggestions or tentative brainstorming as decisions.
   - ACTION ITEMS: Only include commitments and tasks explicitly assigned or volunteered. If an owner is not stated, leave `owner_name` as null. If no deadline is stated, leave `deadline` as null. Never invent owners or deadlines.
   - RISKS & BLOCKERS: Include explicit impediments, technical risks, or major concerns stated by participants.
   - UNRESOLVED QUESTIONS: Include clear open inquiries, unknowns, or follow-up questions that were not answered in the meeting.
   - If no items exist in a category, provide an empty array [].

4. REQUIRED JSON SCHEMA:
You must output a single valid JSON object with the following exact keys:
{{
  "executive_summary": "Concise executive overview (1-2 paragraphs)",
  "detailed_summary": "In-depth summary organized by the main discussion topics",
  "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
  "topics": [
    {{
      "title": "Topic title",
      "description": "Brief description of the discussion on this topic",
      "start_time": 0.0,
      "end_time": 60.0
    }}
  ],
  "decisions": [
    {{
      "description": "Explicit statement of decision reached",
      "rationale_or_context": "Why it was decided or context",
      "decided_by": "Person or team responsible",
      "evidence_segment_id": "exact-segment-id-from-transcript"
    }}
  ],
  "action_items": [
    {{
      "description": "Task to be executed",
      "owner_name": "Name of owner or null",
      "deadline": null,
      "status": "PENDING",
      "evidence_segment_id": "exact-segment-id-from-transcript"
    }}
  ],
  "risks": [
    {{
      "description": "Description of the risk or blocker",
      "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
      "mitigation_suggestion": "Proposed mitigation or null",
      "evidence_segment_id": "exact-segment-id-from-transcript"
    }}
  ],
  "unresolved_questions": [
    {{
      "question_text": "The open question requiring follow-up",
      "raised_by": "Speaker name or null",
      "status": "OPEN",
      "evidence_segment_id": "exact-segment-id-from-transcript"
    }}
  ]
}}
"""


def format_transcript_context(segments: List[TranscriptSegment]) -> str:
    """Format a sequence of transcript segments into a clean numbered prompt context."""
    lines = []
    for s in segments:
        speaker = s.speaker.speaker_label if s.speaker else f"Speaker_{s.speaker_id or 'unknown'}"
        time_tag = f"{s.start_time:.1f}s - {s.end_time:.1f}s"
        lines.append(f"[{s.id} | {time_tag} | {speaker} | lang:{s.language_code}]: {s.text.strip()}")
    return "\n".join(lines)


def build_intelligence_prompt(
    segments: List[TranscriptSegment],
    meeting_title: str = "Meeting",
    output_language: str = "English",
) -> Dict[str, str]:
    """Construct system prompt and user prompt for LLM intelligence extraction."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(output_language=output_language)
    transcript_text = format_transcript_context(segments)
    user_prompt = (
        f"Meeting Title: {meeting_title}\n"
        f"Transcript Segment Count: {len(segments)}\n\n"
        f"--- PRIVACY-SAFE TRANSCRIPT BEGIN ---\n"
        f"{transcript_text}\n"
        f"--- PRIVACY-SAFE TRANSCRIPT END ---\n\n"
        "Extract the structured intelligence now. Respond with valid JSON only."
    )
    return {
        "system": system_prompt,
        "prompt": user_prompt,
    }
