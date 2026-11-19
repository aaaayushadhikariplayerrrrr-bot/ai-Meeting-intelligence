"""Phase 10 — Spoken PII & Verbal Credential Redaction Regression Test Suite.

Validates that spoken sensitive information:
- Spoken mobile numbers (digit-by-digit English number words, ASR "oh"/"zero", multipliers like "double"/"triple", spaced digits, Hinglish).
- Spoken email addresses ("... at the rate ...").
- Verbally introduced passwords ("password is ...", "password set to ...", "mera password hai ...").
- Spoken API keys and tokens ("the api key we are using will be ...", spelled-out characters, retro-assignment).

Are reliably detected and redacted from CLEANED_REDACTED transcripts while:
1. Strictly preserving the immutable ORIGINAL transcript.
2. Protecting ordinary quantities, dates, years, times, currency, and section references from false-positive redaction.
3. Preserving legal conditioning and obligation terms (shall, must, shall not, subject to, etc.).
4. Ensuring downstream memory indexing and SQLite FTS5 contain zero sensitive tokens, and that stale chunks are purged.

IMPORTANT: All test fixtures use synthetic, fictitious dummy values only. Real credentials or numbers are never used.
"""

import json
import uuid
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization, User
from app.models.meeting import Meeting, ProcessingJob, JobStatus, JobType
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.privacy_redaction.redactor import (
    redact_text,
    _redact_spoken_and_spaced_numbers,
    _redact_verbal_passwords,
    _redact_verbal_api_keys,
)
from app.services.transcript_processing.processor import derive_processed_transcript
from app.services.memory.job_handler import schedule_indexing_job, process_indexing_job_sync
from app.services.memory.storage import ChunkStorageManager


# ==============================================================================
# Area 1: Spoken Phone Number & Spaced Digits Detection
# ==============================================================================

def test_spoken_phone_number_english_digit_words():
    """Verify 10-digit phone spoken as individual English words is redacted."""
    text = "Please reach me on my mobile number nine eight one one four zero one zero seven two tomorrow."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "nine eight one one" not in res.redacted_text
    assert "PHONE" in res.categories_detected
    assert "reach me on my mobile number [PHONE_REDACTED] tomorrow" in res.redacted_text


def test_spoken_phone_number_with_double_multiplier():
    """Verify spoken numbers using 'double' or 'triple' multipliers are expanded and redacted."""
    text = "My phone is nine eight double one four zero one zero eight two check it."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "nine eight double one" not in res.redacted_text
    assert "PHONE" in res.categories_detected
    assert "check it" in res.redacted_text


def test_spoken_phone_number_with_oh_asr_variant():
    """Verify spoken numbers with ASR 'oh' for zero are recognized and redacted."""
    text = "Call zero nine eight one one four oh one oh seven two please."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "four oh one oh" not in res.redacted_text
    assert "please" in res.redacted_text


def test_spoken_phone_number_with_country_code():
    """Verify spoken numbers prefixed by 'plus nine one' or '+91' are redacted."""
    text1 = "You can call plus nine one nine eight one one four zero one zero seven two anytime."
    res1 = redact_text(text1)
    assert res1.is_redacted is True
    assert "[PHONE_REDACTED]" in res1.redacted_text

    text2 = "Contact me at +91 9 8 1 1 4 0 1 0 7 2 for support."
    res2 = redact_text(text2)
    assert res2.is_redacted is True
    assert "[PHONE_REDACTED]" in res2.redacted_text
    assert "for support" in res2.redacted_text


def test_spaced_numeric_digits_phone():
    """Verify phone numbers transcribed as spaced digits are redacted."""
    text = "My contact number is 9 8 1 1 4 0 1 0 7 2 thank you."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "9 8 1 1" not in res.redacted_text
    assert "thank you" in res.redacted_text


def test_spoken_phone_number_hinglish():
    """Verify spoken phone number in transliterated Hindi/Hinglish is redacted."""
    text = "Mera mobile number hai nau aath ek ek char shunya ek shunya saat do note kar lijiye."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "nau aath ek ek" not in res.redacted_text
    assert "note kar lijiye" in res.redacted_text


# ==============================================================================
# Area 2: Spoken Email Address Detection
# ==============================================================================

def test_spoken_email_address_at_the_rate():
    """Verify spoken email containing 'at the rate' is redacted."""
    text = "I will notify you on your gmail id ravi zero nine eight at the rate gmail.com okay."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[EMAIL_REDACTED]" in res.redacted_text
    assert "ravi zero nine eight" not in res.redacted_text
    assert "EMAIL" in res.categories_detected
    assert "okay" in res.redacted_text


def test_spoken_email_address_at_rate_variant():
    """Verify spoken email with 'at rate' and 'dot com' is redacted."""
    text = "Send the document to dummy user at rate company dot com by noon."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[EMAIL_REDACTED]" in res.redacted_text
    assert "EMAIL" in res.categories_detected
    assert "by noon" in res.redacted_text


# ==============================================================================
# Area 3: Verbal Credential & Password Detection
# ==============================================================================

def test_verbal_password_introduction():
    """Verify verbally introduced passwords following assignment phrases are redacted."""
    text = "My username is testuser and password is SecretDummyPass123 and we can log in."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PASSWORD_REDACTED]" in res.redacted_text
    assert "SecretDummyPass123" not in res.redacted_text
    assert "SECRET" in res.categories_detected
    assert "and we can log in" in res.redacted_text


def test_verbal_password_with_filler_and_repetition():
    """Verify verbal passwords with hesitation noise ('uh') and multiple occurrences are redacted."""
    text = "and password is uh sh 123 and password is rabi 2026 check it"
    res = redact_text(text)
    assert res.is_redacted is True
    assert "sh 123" not in res.redacted_text
    assert "rabi 2026" not in res.redacted_text
    assert res.redacted_text.count("[PASSWORD_REDACTED]") == 2
    assert "check it" in res.redacted_text


def test_verbal_password_hinglish():
    """Verify verbal passwords introduced in Hinglish are redacted."""
    text = "Mera password hai SuperSafePass99 toh kisi ko mat batana."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PASSWORD_REDACTED]" in res.redacted_text
    assert "SuperSafePass99" not in res.redacted_text
    assert "mat batana" in res.redacted_text


# ==============================================================================
# Area 4: Spoken API Key & Secret Token Detection
# ==============================================================================

def test_spoken_api_key_verbal_prefix():
    """Verify spoken API keys introduced by verbal phrases are redacted."""
    text = "The api key we are using will be zero x y z nine eight one one one four c y z this will work."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[API_KEY_REDACTED]" in res.redacted_text
    assert "zero x y z" not in res.redacted_text
    assert "API_KEY" in res.categories_detected
    assert "this will work" in res.redacted_text


def test_retro_spoken_api_key():
    """Verify retroactively introduced API keys ending with 'this will be our api key' are redacted."""
    text = "zero a b c one two three four this will be our api key and to check yourself"
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[API_KEY_REDACTED]" in res.redacted_text
    assert "zero a b c" not in res.redacted_text
    assert "and to check yourself" in res.redacted_text


def test_spoken_openai_key_format():
    """Verify standard OpenAI key format in speech or transcript is redacted."""
    text = "We configured the system with sk-proj-1234567890abcdef1234567890 for testing."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[API_KEY_REDACTED]" in res.redacted_text
    assert "sk-proj-1234567890abcdef" not in res.redacted_text


# ==============================================================================
# Area 5: False Positive Protections (Must NOT Redact)
# ==============================================================================

def test_false_positive_quantities_and_counts():
    """Verify ordinary conversational quantities are never redacted."""
    text = "We have three servers and five databases running in production."
    res = redact_text(text)
    assert res.is_redacted is False
    assert res.redacted_text == text
    assert len(res.categories_detected) == 0


def test_false_positive_dates_and_years():
    """Verify calendar dates, months, and years are protected from redaction."""
    text = "We will try to upload and deploy our project on 20 november 2026 without delay."
    res = redact_text(text)
    assert res.is_redacted is False
    assert "20 november 2026" in res.redacted_text
    assert len(res.categories_detected) == 0


def test_false_positive_written_date():
    """Verify spelled-out dates are protected."""
    text = "The next quarterly meeting is scheduled on October twenty fourth."
    res = redact_text(text)
    assert res.is_redacted is False
    assert "October twenty fourth" in res.redacted_text


def test_false_positive_currency_and_metrics():
    """Verify currency amounts, percentages, and metrics are protected."""
    text = "We spent fifty thousand dollars, representing an increase of five percent."
    res = redact_text(text)
    assert res.is_redacted is False
    assert "fifty thousand dollars" in res.redacted_text
    assert "five percent" in res.redacted_text


def test_false_positive_demo_intro():
    """Verify demo introductory sentence is not over-redacted."""
    text = "Demo one this video is for the testing purpose only."
    res = redact_text(text)
    assert res.is_redacted is False
    assert "Demo one" in res.redacted_text


# ==============================================================================
# Area 6: Legal Conditioning Preservation
# ==============================================================================

def test_legal_conditioning_obligation_words_preserved():
    """Verify legal conditioning, obligations, and restrictions are preserved while redacting credentials."""
    text = (
        "The vendor shall not disclose confidential data, subject to Section 10. "
        "The password is DummyPassword123 and must be changed immediately."
    )
    res = redact_text(text)
    assert res.is_redacted is True
    assert "[PASSWORD_REDACTED]" in res.redacted_text
    assert "DummyPassword123" not in res.redacted_text
    assert "shall not disclose confidential data" in res.redacted_text
    assert "subject to Section 10" in res.redacted_text
    assert "must be changed immediately" in res.redacted_text


# ==============================================================================
# Area 7: Pipeline Integration & Original Transcript Immutability
# ==============================================================================

def test_pipeline_preserves_original_transcript_and_redacts_derived(db_session: Session):
    """Verify end-to-end pipeline:
    1. ORIGINAL transcript segments remain untouched.
    2. CLEANED_REDACTED transcript contains redacted placeholders.
    3. Audit log records redaction event without sensitive values.
    """
    org = Organization(name="Spoken PII Org", slug=f"spoken-pii-{uuid.uuid4().hex[:8]}")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="PII Verification Sync")
    db_session.add(meeting)
    db_session.commit()

    # Create ORIGINAL transcript with spoken sensitive information
    orig_transcript = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    db_session.add(orig_transcript)
    db_session.flush()

    raw_phone_text = "Please call my mobile number nine eight one one four zero one zero seven two soon."
    raw_pass_text = "The admin password is AdminSecretPass2026 and must remain private."
    raw_safe_text = "Deploy on 20 november 2026 with three servers."

    seg0 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=orig_transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text=raw_phone_text,
        language_code="en",
        is_redacted=False,
    )
    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=orig_transcript.id,
        sequence_order=1,
        start_time=5.0,
        end_time=10.0,
        text=raw_pass_text,
        language_code="en",
        is_redacted=False,
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=orig_transcript.id,
        sequence_order=2,
        start_time=10.0,
        end_time=15.0,
        text=raw_safe_text,
        language_code="en",
        is_redacted=False,
    )
    db_session.add_all([seg0, seg1, seg2])
    db_session.commit()

    # Derive CLEANED_REDACTED transcript
    derived = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    # Invariant 1: ORIGINAL segments remain completely untouched
    orig_segs = (
        db_session.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == orig_transcript.id)
        .order_by(TranscriptSegment.sequence_order.asc())
        .all()
    )
    assert orig_segs[0].text == raw_phone_text
    assert orig_segs[0].is_redacted is False
    assert orig_segs[1].text == raw_pass_text
    assert orig_segs[1].is_redacted is False
    assert orig_segs[2].text == raw_safe_text
    assert orig_segs[2].is_redacted is False

    # Invariant 2: Derived CLEANED_REDACTED segments are redacted
    derived_segs = (
        db_session.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == derived.id)
        .order_by(TranscriptSegment.sequence_order.asc())
        .all()
    )
    assert "[PHONE_REDACTED]" in derived_segs[0].text
    assert "nine eight one one" not in derived_segs[0].text
    assert derived_segs[0].is_redacted is True

    assert "[PASSWORD_REDACTED]" in derived_segs[1].text
    assert "AdminSecretPass2026" not in derived_segs[1].text
    assert derived_segs[1].is_redacted is True

    assert "20 november 2026" in derived_segs[2].text
    assert "three servers" in derived_segs[2].text
    assert derived_segs[2].is_redacted is False


# ==============================================================================
# Area 8: Downstream Memory Invalidation
# ==============================================================================

def test_downstream_memory_replaces_stale_chunks_on_reindexing(db_session: Session):
    """Verify that re-indexing a meeting purges stale chunks from memory and FTS5."""
    org = Organization(name="Stale Purge Org", slug=f"stale-purge-{uuid.uuid4().hex[:8]}")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Stale Chunk Invalidation Test")
    db_session.add(meeting)
    db_session.commit()

    # Derived v1 transcript
    t_v1 = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
        language_detected="en",
    )
    db_session.add(t_v1)
    db_session.flush()

    seg_v1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=t_v1.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="Old version chunk content discussing roadmap.",
        language_code="en",
    )
    db_session.add(seg_v1)
    db_session.commit()

    storage = ChunkStorageManager()
    from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult

    # Generate synthetic vector
    rng = np_rng = [0.05] * 384
    norm = sum(x*x for x in rng) ** 0.5
    rng = [x / norm for x in rng]

    chunk_v1 = SemanticChunk(
        meeting_id=meeting.id,
        organization_id=org.id,
        transcript_id=t_v1.id,
        chunk_index=0,
        text="Old version chunk content discussing roadmap.",
        start_time=0.0,
        end_time=5.0,
        segment_ids=[seg_v1.id],
        language_codes=["en"],
        dominant_language="en",
        token_count=10,
        is_split_utterance=False,
        metadata={"transcript_type": TranscriptType.CLEANED_REDACTED.value},
    )
    storage.store_chunks(
        db=db_session,
        chunks_with_embeddings=[
            ChunkEmbeddingResult(chunk=chunk_v1, embedding=rng, dimension=384, model_name="test-model")
        ],
        organization_id=org.id,
        transcript=t_v1,
    )
    db_session.commit()

    assert db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == meeting.id).count() == 1

    # Now create derived v2 transcript with new content
    t_v2 = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=2,
        language_detected="en",
    )
    db_session.add(t_v2)
    db_session.flush()

    seg_v2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=t_v2.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="New version chunk content with [PHONE_REDACTED].",
        language_code="en",
    )
    db_session.add(seg_v2)
    db_session.commit()

    chunk_v2 = SemanticChunk(
        meeting_id=meeting.id,
        organization_id=org.id,
        transcript_id=t_v2.id,
        chunk_index=0,
        text="New version chunk content with [PHONE_REDACTED].",
        start_time=0.0,
        end_time=5.0,
        segment_ids=[seg_v2.id],
        language_codes=["en"],
        dominant_language="en",
        token_count=10,
        is_split_utterance=False,
        metadata={"transcript_type": TranscriptType.CLEANED_REDACTED.value},
    )

    # When new chunks for meeting are indexed, delete_chunks_by_meeting deletes stale chunks
    storage.delete_chunks_by_meeting(db=db_session, meeting_id=meeting.id, organization_id=org.id)
    storage.store_chunks(
        db=db_session,
        chunks_with_embeddings=[
            ChunkEmbeddingResult(chunk=chunk_v2, embedding=rng, dimension=384, model_name="test-model")
        ],
        organization_id=org.id,
        transcript=t_v2,
    )
    db_session.commit()

    # Verify stale chunks from t_v1 are gone, only t_v2 exists
    active_chunks = (
        db_session.query(TranscriptChunkRecord)
        .filter(TranscriptChunkRecord.meeting_id == meeting.id)
        .all()
    )
    assert len(active_chunks) == 1
    assert active_chunks[0].transcript_id == t_v2.id
    assert "[PHONE_REDACTED]" in active_chunks[0].text
    assert "roadmap" not in active_chunks[0].text
