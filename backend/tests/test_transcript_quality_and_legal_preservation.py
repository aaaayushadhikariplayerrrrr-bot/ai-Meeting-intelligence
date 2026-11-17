"""Phase 8.11.2 — Legal-Safe Transcript Cleaning & Multilingual Privacy Redaction Tests.

Comprehensive validation covering:
1. Legal meaning preservation:
   - Section numbers, clause numbers, and statutory references.
   - Repeated negations across sentences ("We must not. Not under any circumstance.").
   - Repeated negations in Hindi/Gujarati ("नहीं नहीं", "નહીં નહીં").
   - Obligation words: shall, must, should, may, must not, shall not.
   - Conditional words: unless, except, subject to, provided that, notwithstanding.
2. Numeric & mathematical preservation:
   - Negative values: -5%, -10%, -25 degrees, -15.
   - Section numbers followed by counts: "Section 10. 10 days notice", "Rule 5. 5 members", "In Section 2, 2 exceptions".
   - Notice periods, quorum numbers, dates, years.
3. Hyphenated terms & compounds:
   - English: co-counsel, re-review, de-facto, pre-existing, sub-clause, cross-examination.
   - Hindi: उप-उपायुक्त, उप-उपबंध.
   - Genuine stutter cleanup: "w- we", "th- the".
4. Redaction precision & false-positive protection:
   - "banking sector 4 years ago" -> NOT redacted.
   - "Sector 4 servers" -> NOT redacted.
   - "block 12 deployment" -> NOT redacted.
   - "Flat No. 402, Green Acres" -> REDACTED.
   - "Plot 12, Sector 4, Gandhinagar" -> REDACTED.
   - Sentence boundary preservation (non-greedy matching).
5. Indic numeral redaction:
   - Devanagari phone: +९१ ९८७६५ ४३२१० or ९८७६५४३२१० -> [PHONE_REDACTED].
   - Gujarati phone: +૯૧ ૯૮૭૬૫ ૪૩૨૧૦ or ૯૮૭૬૫૪૩૨૧૦ -> [PHONE_REDACTED].
   - Devanagari Aadhaar: १२३४ ५६७८ ९०१२ with "आधार" -> [AADHAAR_REDACTED].
   - Gujarati Aadhaar: ૧૨૩૪ ૫૬૭૮ ૯૦૧૨ with "આધાર" -> [AADHAAR_REDACTED].
   - Surrounding Devanagari/Gujarati text remains intact in original script.
6. Pipeline integration & database immutability:
   - CLEANED, REDACTED, and CLEANED_REDACTED derivations maintain correct transformations.
   - Original transcript remains strictly immutable in database.
   - Memory indexing uses only REDACTED or CLEANED_REDACTED text.
   - Citation verification continues matching exact transcript segments.
"""

import json
import pytest
from datetime import datetime, timezone
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, ProcessingJob, JobStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.transcript_cleaning.cleaner import clean_segment_text, reduce_accidental_repetitions
from app.services.privacy_redaction.redactor import redact_text
from app.services.transcript_processing.processor import derive_processed_transcript
from app.services.memory.exceptions import PrivacyEnforcementError
from app.services.memory.job_handler import schedule_indexing_job, process_indexing_job_sync
from app.services.memory.rag.verification import CitationVerifier, SAFE_TRANSCRIPT_TYPES
from backend.tests.conftest import make_auth_headers


# ==============================================================================
# Area 1: Legal Meaning Preservation
# ==============================================================================

def test_legal_section_numbers_and_statutory_references():
    """Verify that section numbers, clause numbers, and statutory references are not collapsed or altered."""
    text = "Under Section 10. 10 days notice is required, pursuant to Clause 4(a) and Rule 5. 5 members must attend."
    res = clean_segment_text(text, language_code="en")

    # Both numbers in "10. 10" and "5. 5" must remain distinct
    assert "Section 10. 10 days" in res.cleaned_text
    assert "Rule 5. 5 members" in res.cleaned_text
    assert "Clause 4(a)" in res.cleaned_text
    assert "pursuant to" in res.cleaned_text


def test_repeated_negations_across_sentences_preserved():
    """Verify repeated negations across sentence boundaries are never collapsed."""
    t1 = "We must not. Not under any circumstance."
    res1 = clean_segment_text(t1, language_code="en")
    assert res1.cleaned_text == "We must not. Not under any circumstance."

    t2 = "Never. Never accept unsigned amendments."
    res2 = clean_segment_text(t2, language_code="en")
    assert res2.cleaned_text == "Never. Never accept unsigned amendments."


def test_repeated_negations_hindi_and_gujarati_preserved():
    """Verify rhetorical repeated negations in Hindi and Gujarati are strictly preserved."""
    # Hindi
    hi_text = "हम इस प्रस्ताव को नहीं नहीं स्वीकार करेंगे।"
    res_hi = clean_segment_text(hi_text, language_code="hi")
    assert "नहीं नहीं" in res_hi.cleaned_text or "नहीं-नहीं" in res_hi.cleaned_text

    # Gujarati
    gu_text = "અમે આ શરતોને ક્યારેય નહીં નહીં માનીએ."
    res_gu = clean_segment_text(gu_text, language_code="gu")
    assert "નહીં નહીં" in res_gu.cleaned_text or "નહીં-નહીં" in res_gu.cleaned_text


def test_obligation_words_strictly_preserved():
    """Verify legal obligation verbs (shall, must, should, may, must not, shall not) are never removed or altered."""
    legal_statements = [
        "The board shall convene within 14 days.",
        "The tenant must not sublet the premises.",
        "The contractor shall not disclose confidential data.",
        "Either party may terminate upon written notice.",
        "The committee should review all audit logs.",
    ]
    for statement in legal_statements:
        res = clean_segment_text(statement, language_code="en")
        assert res.cleaned_text == statement


def test_conditional_words_and_exceptions_strictly_preserved():
    """Verify conditional operators and exception phrases are strictly preserved."""
    conditions = [
        "The agreement is binding, unless revoked in writing.",
        "All terms apply except as provided under Schedule B.",
        "This grant is subject to board approval.",
        "Provided that payment is made on time, interest is waived.",
        "Notwithstanding any prior agreement, this clause prevails.",
    ]
    for condition in conditions:
        res = clean_segment_text(condition, language_code="en")
        assert res.cleaned_text == condition


# ==============================================================================
# Area 2: Numeric & Mathematical Preservation
# ==============================================================================

def test_negative_numeric_values_preserved():
    """Verify leading negative signs on percentages, metrics, and temperatures are not stripped."""
    metrics = [
        ("-5% revenue decline observed in Q3.", "-5% revenue decline observed in Q3."),
        ("-10% operating margin", "-10% operating margin"),
        ("-25 degrees recorded in the server room.", "-25 degrees recorded in the server room."),
        ("-15 basis points adjustment", "-15 basis points adjustment"),
    ]
    for raw, expected in metrics:
        res = clean_segment_text(raw, language_code="en")
        assert res.cleaned_text.startswith("-")
        assert expected in res.cleaned_text


def test_section_numbers_followed_by_counts_preserved():
    """Verify numbers separated by comma or period representing distinct legal entities are not collapsed."""
    # Comma separation
    t1 = "In Section 2, 2 exceptions were explicitly noted by counsel."
    res1 = clean_segment_text(t1, language_code="en")
    assert "Section 2, 2 exceptions" in res1.cleaned_text

    # Date followed by count
    t2 = "On May 5, 5 amendments were submitted to the committee."
    res2 = clean_segment_text(t2, language_code="en")
    assert "May 5, 5 amendments" in res2.cleaned_text


def test_notice_periods_quorum_counts_and_dates_preserved():
    """Verify notice periods, quorum counts, dates, and fiscal years are preserved intact."""
    text = "A 30-day notice period is required for a quorum of 7 members in fiscal year 2026."
    res = clean_segment_text(text, language_code="en")
    assert "30-day notice" in res.cleaned_text
    assert "quorum of 7 members" in res.cleaned_text
    assert "fiscal year 2026" in res.cleaned_text


# ==============================================================================
# Area 3: Hyphenated Terms & Compounds
# ==============================================================================

def test_english_hyphenated_legal_terms_preserved():
    """Verify legitimate hyphenated English legal and technical terms are never collapsed as stutters."""
    terms = [
        "co-counsel",
        "re-review",
        "de-facto",
        "pre-existing",
        "sub-clause",
        "sub-section",
        "cross-examination",
        "pro-rata",
        "ex-parte",
        "inter-alia",
    ]
    for term in terms:
        raw = f"The attorney discussed the {term} during the hearing."
        res = clean_segment_text(raw, language_code="en")
        assert term in res.cleaned_text


def test_hindi_hyphenated_compounds_preserved():
    """Verify Hindi administrative and legal hyphenated compounds are not collapsed."""
    compounds = [
        "उप-उपायुक्त",
        "उप-उपबंध",
    ]
    for comp in compounds:
        raw = f"{comp} ने नए नियमों की घोषणा की।"
        res = clean_segment_text(raw, language_code="hi")
        assert comp in res.cleaned_text


def test_genuine_stutters_still_cleaned():
    """Verify genuine accidental speech stutters are still cleanly reduced."""
    raw = "um, w- we need to deploy th- the system."
    res = clean_segment_text(raw, language_code="en")
    assert "w- we" not in res.cleaned_text
    assert "th- the" not in res.cleaned_text
    assert "We need to deploy the system." in res.cleaned_text


# ==============================================================================
# Area 4: Redaction Precision & Address False-Positive Protection
# ==============================================================================

def test_address_false_positives_technical_and_business_phrases():
    """Verify technical, computing, and industry usages of 'sector' and 'block' are NOT redacted."""
    non_addresses = [
        "The banking sector 4 years ago was quite different.",
        "Sector 4 servers were upgraded to Linux 6.2.",
        "We completed the block 12 deployment without downtime.",
        "In the financial sector 2 quarters of growth were reported.",
        "Allocated block 5 storage volume to the PostgreSQL cluster.",
        "Public sector 10-year bonds were issued yesterday.",
    ]
    for text in non_addresses:
        res = redact_text(text)
        assert res.is_redacted is False, f"False positive redaction on: {text}"
        assert "[ADDRESS_REDACTED]" not in res.redacted_text
        assert res.redacted_text == text


def test_legitimate_address_redaction_precision():
    """Verify legitimate residential and sector postal addresses ARE redacted."""
    real_addresses = [
        ("We delivered the documents to Flat No. 402, Green Acres yesterday.", "[ADDRESS_REDACTED]"),
        ("The registered office is at Plot 12, Sector 4, Gandhinagar for all notices.", "[ADDRESS_REDACTED]"),
        ("Please visit Plot No. 45, Sector 62, Noida tomorrow.", "[ADDRESS_REDACTED]"),
        ("The meeting took place at Sector 4, Gandhinagar.", "[ADDRESS_REDACTED]"),
    ]
    for raw, expected_token in real_addresses:
        res = redact_text(raw)
        assert res.is_redacted is True, f"Failed to redact address: {raw}"
        assert expected_token in res.redacted_text
        assert "ADDRESS" in res.categories_detected


def test_address_does_not_swallow_remaining_sentence():
    """Verify non-greedy address redaction does NOT consume trailing sentence text."""
    raw = "We met at Flat No. 402, Green Acres to finalize the agreement."
    res = redact_text(raw)
    assert res.is_redacted is True
    assert "to finalize the agreement." in res.redacted_text
    assert res.redacted_text == "We met at [ADDRESS_REDACTED] to finalize the agreement."

    raw2 = "Headquarters at Plot 12, Sector 4, Gandhinagar handles compliance."
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "handles compliance." in res2.redacted_text


# ==============================================================================
# Area 5: Indic Numeral Redaction
# ==============================================================================

def test_devanagari_phone_redaction():
    """Verify Devanagari numerals in mobile numbers are detected and redacted."""
    # With country code
    raw1 = "मेरा फोन नंबर +९१ ९८७६५ ४३२१० है।"
    res1 = redact_text(raw1)
    assert res1.is_redacted is True
    assert "[PHONE_REDACTED]" in res1.redacted_text
    assert "PHONE" in res1.categories_detected
    assert "९८७६५" not in res1.redacted_text
    assert "मेरा फोन नंबर" in res1.redacted_text  # Surrounding text preserved

    # 10-digit continuous
    raw2 = "कृपया ९८७६५४३२१० पर संपर्क करें।"
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "[PHONE_REDACTED]" in res2.redacted_text
    assert "९८७६५४३२१०" not in res2.redacted_text
    assert "कृपया [PHONE_REDACTED] पर संपर्क करें।" == res2.redacted_text


def test_gujarati_phone_redaction():
    """Verify Gujarati numerals in mobile numbers are detected and redacted."""
    # With country code
    raw1 = "મારો ફોન નંબર +૯૧ ૯૮૭૬૫ ૪૩૨૧૦ છે."
    res1 = redact_text(raw1)
    assert res1.is_redacted is True
    assert "[PHONE_REDACTED]" in res1.redacted_text
    assert "PHONE" in res1.categories_detected
    assert "૯૮૭૬૫" not in res1.redacted_text
    assert "મારો ફોન નંબર" in res1.redacted_text

    # 10-digit continuous
    raw2 = "કૃપા કરીને ૯૮૭૬૫૪૩૨૧૦ પર કૉલ કરો."
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "[PHONE_REDACTED]" in res2.redacted_text
    assert "૯૮૭૬૫૪૩૨૧૦" not in res2.redacted_text
    assert "કૃપા કરીને [PHONE_REDACTED] પર કૉલ કરો." == res2.redacted_text


def test_devanagari_aadhaar_redaction_with_affirmative_context():
    """Verify Devanagari 12-digit Aadhaar numbers with Hindi context are redacted."""
    # Space separated
    raw1 = "कर्मचारी का आधार कार्ड १२३४ ५६७८ ९०१२ सत्यापित हुआ।"
    res1 = redact_text(raw1)
    assert res1.is_redacted is True
    assert "[AADHAAR_REDACTED]" in res1.redacted_text
    assert "AADHAAR" in res1.categories_detected
    assert "१२३४" not in res1.redacted_text
    assert "कर्मचारी का आधार कार्ड [AADHAAR_REDACTED] सत्यापित हुआ।" == res1.redacted_text

    # Continuous
    raw2 = "यूआईडी १२३४५६७८९०१२ प्रस्तुत किया गया।"
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "[AADHAAR_REDACTED]" in res2.redacted_text
    assert "१२३४५६७८९०१२" not in res2.redacted_text


def test_gujarati_aadhaar_redaction_with_affirmative_context():
    """Verify Gujarati 12-digit Aadhaar numbers with Gujarati context are redacted."""
    # Space separated
    raw1 = "ગ્રાહકનો આધાર નંબર ૧૨૩૪ ૫૬૭૮ ૯૦૧૨ ચકાસવામાં આવ્યો."
    res1 = redact_text(raw1)
    assert res1.is_redacted is True
    assert "[AADHAAR_REDACTED]" in res1.redacted_text
    assert "AADHAAR" in res1.categories_detected
    assert "૧૨૩૪" not in res1.redacted_text
    assert "ગ્રાહકનો આધાર નંબર [AADHAAR_REDACTED] ચકાસવામાં આવ્યો." == res1.redacted_text

    # Continuous
    raw2 = "યુઆઈડી ૧૨૩૪૫૬૭૮૯૦૧૨ નોંધાયેલ છે."
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "[AADHAAR_REDACTED]" in res2.redacted_text
    assert "૧૨૩૪૫૬૭૮૯૦૧૨" not in res2.redacted_text


def test_indic_commercial_numbers_protected_from_redaction():
    """Verify 12-digit Indic numbers with commercial context (order, invoice) are NOT redacted as Aadhaar."""
    raw = "ऑर्डर # १२३४ ५६७८ ९०१२ सिस्टम में दर्ज है।"
    res = redact_text(raw)
    assert "[AADHAAR_REDACTED]" not in res.redacted_text
    assert "१२३४" in res.redacted_text


# ==============================================================================
# Area 6: Pipeline Integration, Immutability & Safe Indexing
# ==============================================================================

def test_pipeline_cleaned_redacted_cleaned_redacted_derivations(db_session):
    """Verify CLEANED, REDACTED, and CLEANED_REDACTED derivations maintain correct transformations."""
    # Setup test Org, User, Meeting
    org = Organization(name="Pipeline Test Org", slug="pipeline-test-org")
    user = User(
        email="compliance@pipelinetest.com",
        full_name="Compliance Officer",
        hashed_password="hashed_pw",
        is_active=True,
    )
    db_session.add_all([org, user])
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Pipeline Transformation Audit")
    db_session.add(meeting)
    db_session.commit()

    # Original raw text contains stutters AND PII
    raw_segment_text = "um, w- we agreed that Section 10. 10 days notice applies. Call +91 98765 43210 for details."
    orig_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(orig_transcript)
    db_session.flush()

    orig_seg = TranscriptSegment(
        transcript_id=orig_transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text=raw_segment_text,
        language_code="en",
    )
    db_session.add(orig_seg)
    db_session.commit()

    # 1. Derive CLEANED: stutters removed, repetition of "10. 10" preserved, phone number NOT redacted
    cleaned_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED,
        db=db_session,
    )
    db_session.commit()
    cleaned_seg = cleaned_t.segments[0]
    assert "w- we" not in cleaned_seg.text
    assert "Section 10. 10 days" in cleaned_seg.text
    assert "+91 98765 43210" in cleaned_seg.text  # Phone not redacted in CLEANED
    assert cleaned_seg.source_segment_id == orig_seg.id

    # 2. Derive REDACTED: phone redacted, stutters remain
    redacted_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )
    db_session.commit()
    redacted_seg = redacted_t.segments[0]
    assert "[PHONE_REDACTED]" in redacted_seg.text
    assert "+91 98765 43210" not in redacted_seg.text
    assert "w- we" in redacted_seg.text  # Stutters remain in raw REDACTED
    assert redacted_seg.source_segment_id == orig_seg.id

    # 3. Derive CLEANED_REDACTED: both cleaned and redacted
    cr_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )
    db_session.commit()
    cr_seg = cr_t.segments[0]
    assert "w- we" not in cr_seg.text
    assert "Section 10. 10 days" in cr_seg.text
    assert "[PHONE_REDACTED]" in cr_seg.text
    assert "+91 98765 43210" not in cr_seg.text

    # 4. Verify Original Transcript remains 100% UNMODIFIED
    db_session.refresh(orig_seg)
    assert orig_seg.text == raw_segment_text


def test_memory_indexing_strictly_rejects_unredacted_transcripts(db_session):
    """Verify background memory indexing strictly rejects unredacted (ORIGINAL/CLEANED) transcripts."""
    org = Organization(name="Index Safety Org", slug="index-safety-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Index Safety Verification")
    db_session.add(meeting)
    db_session.commit()

    # Create ONLY an ORIGINAL transcript with unredacted Aadhaar
    raw_aadhaar = "2345 6789 0123"
    orig_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(orig_transcript)
    db_session.flush()

    orig_seg = TranscriptSegment(
        transcript_id=orig_transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=3.0,
        text=f"The citizen provided Aadhaar {raw_aadhaar}.",
        language_code="en",
    )
    db_session.add(orig_seg)
    db_session.commit()

    # 1. Directly targeting ORIGINAL transcript must fail with PrivacyEnforcementError
    with pytest.raises(PrivacyEnforcementError) as exc_info:
        schedule_indexing_job(
            db=db_session,
            meeting_id=meeting.id,
            organization_id=org.id,
            transcript_id=orig_transcript.id,
        )
    assert "Only REDACTED or CLEANED_REDACTED transcripts may be indexed" in str(exc_info.value)

    # 2. Derive REDACTED transcript
    redacted_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )
    db_session.commit()

    # Verify REDACTED transcript is in SAFE_TRANSCRIPT_TYPES
    assert redacted_t.transcript_type in SAFE_TRANSCRIPT_TYPES
    redacted_seg = redacted_t.segments[0]
    assert raw_aadhaar not in redacted_seg.text
    assert "[AADHAAR_REDACTED]" in redacted_seg.text


def test_citation_verification_with_cleaned_redacted_segments(db_session):
    """Verify that structured citations correctly verify against exact database TranscriptSegments."""
    org = Organization(name="Citation Org", slug="citation-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(
        organization_id=org.id,
        title="Citation Verification Meeting",
        actual_start_time=datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add(meeting)
    db_session.commit()

    # Create safe transcript
    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=12.5,
        end_time=18.0,
        text="Section 10. 10 days notice is required under statutory rules.",
        language_code="en",
    )
    db_session.add(seg)
    db_session.commit()

    verifier = CitationVerifier()
    cit = verifier.verify_structured_evidence(
        db=db_session,
        evidence_segment_id=seg.id,
        meeting_id=meeting.id,
        organization_id=org.id,
        entity_ref_id="dec-123",
        citation_id="cit-1",
    )

    assert cit is not None
    assert cit.citation_id == "cit-1"
    assert cit.meeting_id == meeting.id
    assert cit.meeting_title == "Citation Verification Meeting"
    assert cit.start_time == 12.5
    assert cit.end_time == 18.0
    assert cit.quoted_text == "Section 10. 10 days notice is required under statutory rules."
