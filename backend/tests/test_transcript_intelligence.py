"""Tests for Phase 5: Transcript Intelligence, Cleaning & Privacy Redaction.

Covers:
- Language-aware filler removal and stutter reduction
- Meaningful lexical reduplication preservation (Hindi, Gujarati, English, Hinglish)
- Indian PII redaction (Email, Indian Phone, Aadhaar, PAN, API Keys, Passwords)
- False-positive protection for technical numbers and metrics
- Separate transcript derivation and original preservation
- Segment-to-source linking and version coexistence
- Audit logging without leaking sensitive values
- API endpoint validation and organization tenant isolation
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting
from app.core.security import hash_password
from backend.tests.conftest import make_auth_headers
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import AuditLog
from app.services.transcript_cleaning.cleaner import (
    clean_segment_text,
    normalize_whitespace,
    reduce_accidental_repetitions,
)
from app.services.privacy_redaction.redactor import redact_text, _luhn_checksum_valid
from app.services.transcript_processing.processor import derive_processed_transcript


# --- 1. Transcript Cleaning Unit Tests ---

def test_cleaner_english_fillers_and_stutters():
    """Verify English filler sounds (uh, um) and stutter hyphens are cleaned without altering meaning."""
    text = "um, uh, we need to w- we need to deploy the the API today."
    res = clean_segment_text(text, language_code="en")

    assert res.has_changed is True
    # "um" and "uh" removed, "w- we" fixed, "the the" deduplicated
    assert "um" not in res.cleaned_text.lower()
    assert "uh" not in res.cleaned_text.lower()
    assert "w- we" not in res.cleaned_text
    assert "the the" not in res.cleaned_text
    assert "API" in res.cleaned_text  # Technical term preserved
    assert res.cleaned_text.startswith("We")  # Sentence capitalized


def test_cleaner_preserves_meaningful_hindi_reduplication():
    """Verify Hindi lexicalized reduplications (e.g. 'धीरे-धीरे', 'बार-बार') are strictly preserved."""
    hindi_text = "हम धीरे-धीरे आगे बढ़ेंगे और बार-बार परीक्षण करेंगे।"
    res = clean_segment_text(hindi_text, language_code="hi")

    # Meaningful reduplications must NOT be collapsed
    assert "धीरे-धीरे" in res.cleaned_text or "धीरे धीरे" in res.cleaned_text
    assert "बार-बार" in res.cleaned_text or "बार बार" in res.cleaned_text
    assert "परीक्षण" in res.cleaned_text


def test_cleaner_preserves_meaningful_gujarati_reduplication():
    """Verify Gujarati lexicalized reduplications (e.g. 'ધીમે ધીમે', 'સાથે સાથે') are strictly preserved."""
    gu_text = "અમે ધીમે ધીમે સાથે સાથે કામ પૂર્ણ કરીશું."
    res = clean_segment_text(gu_text, language_code="gu")

    assert "ધીમે ધીમે" in res.cleaned_text
    assert "સાથે સાથે" in res.cleaned_text


def test_cleaner_preserves_context_sensitive_words():
    """Verify context-sensitive words like 'like', 'तो', 'मतलब', 'हाँ', 'હા' are NOT blindly stripped."""
    en_like = "This architecture looks like a microservices pattern."
    res_en = clean_segment_text(en_like, language_code="en")
    assert "like" in res_en.cleaned_text

    hi_text = "अगर सर्वर डाउन है तो मुझे बताओ, इसका मतलब क्या है?"
    res_hi = clean_segment_text(hi_text, language_code="hi")
    assert "तो" in res_hi.cleaned_text
    assert "मतलब" in res_hi.cleaned_text

    gu_text = "હા, આ યોજના ઉત્તમ છે, તો આપણે શરૂ કરીએ."
    res_gu = clean_segment_text(gu_text, language_code="gu")
    assert "હા" in res_gu.cleaned_text
    assert "તો" in res_gu.cleaned_text


def test_cleaner_punctuation_normalization():
    """Verify irregular punctuation spacing and duplicates are normalized."""
    text = "We discussed the budget , and timeline .. Are we ready ??"
    res = clean_segment_text(text, language_code="en")
    assert res.cleaned_text == "We discussed the budget, and timeline. Are we ready?"


# --- 2. Privacy Redaction Unit Tests ---

def test_redactor_email():
    """Verify email addresses are redacted with placeholder [EMAIL_REDACTED]."""
    text = "Please reach out to rahul.sharma@example.co.in or support@meeting-app.com for help."
    res = redact_text(text)
    assert res.is_redacted is True
    assert "rahul.sharma@example.co.in" not in res.redacted_text
    assert "support@meeting-app.com" not in res.redacted_text
    assert "[EMAIL_REDACTED]" in res.redacted_text
    assert "EMAIL" in res.categories_detected
    assert res.redaction_count == 2


def test_redactor_indian_phone_numbers():
    """Verify Indian mobile numbers, +91 formats, and STD landlines are redacted."""
    # 1. +91 formatted mobile
    t1 = "Call me at +91 98765 43210 immediately."
    r1 = redact_text(t1)
    assert r1.is_redacted is True
    assert "98765" not in r1.redacted_text
    assert "[PHONE_REDACTED]" in r1.redacted_text
    assert "PHONE" in r1.categories_detected

    # 2. 10-digit mobile
    t2 = "My backup number is 9876543210."
    r2 = redact_text(t2)
    assert r2.is_redacted is True
    assert "9876543210" not in r2.redacted_text
    assert "[PHONE_REDACTED]" in r2.redacted_text

    # 3. STD Landline
    t3 = "Office line is 011-23456789."
    r3 = redact_text(t3)
    assert r3.is_redacted is True
    assert "011-23456789" not in r3.redacted_text
    assert "[PHONE_REDACTED]" in r3.redacted_text


def test_redactor_government_ids():
    """Verify Indian Aadhaar and PAN numbers are redacted."""
    # Aadhaar (4-4-4 formatted)
    t_aadhaar = "The Aadhaar card number submitted was 3456 7890 1234."
    r_aadhaar = redact_text(t_aadhaar)
    assert r_aadhaar.is_redacted is True
    assert "3456" not in r_aadhaar.redacted_text
    assert "[AADHAAR_REDACTED]" in r_aadhaar.redacted_text or "[GOVT_ID_REDACTED]" in r_aadhaar.redacted_text
    assert "GOVT_ID" in r_aadhaar.categories_detected
    assert "AADHAAR" in r_aadhaar.categories_detected

    # PAN Card
    t_pan = "His PAN card is ABCDE1234F for billing."
    r_pan = redact_text(t_pan)
    assert r_pan.is_redacted is True
    assert "ABCDE1234F" not in r_pan.redacted_text
    assert "[GOVT_ID_REDACTED]" in r_pan.redacted_text
    assert "GOVT_ID" in r_pan.categories_detected


def test_redactor_api_keys_and_passwords():
    """Verify API tokens and secrets are redacted."""
    t_key = "We used the key sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5 in production."
    r_key = redact_text(t_key)
    assert r_key.is_redacted is True
    assert "sk-proj" not in r_key.redacted_text
    assert "[API_KEY_REDACTED]" in r_key.redacted_text
    assert "API_KEY" in r_key.categories_detected

    t_pass = "The database password=SuperSecretPassword99! must be changed."
    r_pass = redact_text(t_pass)
    assert r_pass.is_redacted is True
    assert "SuperSecretPassword99!" not in r_pass.redacted_text
    assert "[SECRET_REDACTED]" in r_pass.redacted_text


def test_redactor_false_positive_protection_technical_numbers():
    """Verify ordinary technical numbers, ports, timestamps, and metrics are NOT redacted."""
    technical_text = (
        "Server is running on port 8000 with 16 kHz mono WAV audio. "
        "Processed 300 seconds in 12.5 seconds with 50% GPU usage. "
        "Roadmap targets Q3 2026 with 100 users."
    )
    res = redact_text(technical_text)
    assert res.is_redacted is False
    assert res.redacted_text == technical_text
    assert res.redaction_count == 0


def test_redactor_credit_card_luhn_check():
    """Verify Luhn checksum distinguishes valid cards from random 16-digit IDs."""
    # Synthetic valid Luhn card number (4532 0150 0000 0007 is a valid Luhn number)
    valid_card = "4532-0150-0000-0007"
    assert _luhn_checksum_valid(valid_card) is True

    res_card = redact_text(f"Paid using card {valid_card}.")
    assert res_card.is_redacted is True
    assert "[CARD_REDACTED]" in res_card.redacted_text

    # Invalid Luhn 16-digit ID (e.g. random database order ID)
    invalid_luhn = "1234-5678-9012-3456"
    assert _luhn_checksum_valid(invalid_luhn) is False
    res_non_card = redact_text(f"Transaction ID is {invalid_luhn}.")
    assert "[CARD_REDACTED]" not in res_non_card.redacted_text


# --- 3. Database Persistence, Derivation & Versioning Tests ---

def test_original_transcript_preservation_and_derivation(db_session):
    """Verify original transcript is strictly preserved and derived versions coexist cleanly."""
    # 1. Setup Org and Meeting
    org = Organization(name="Privacy Test Org", slug="privacy-test-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Sprint Retro with Sensitive Data")
    db_session.add(meeting)
    db_session.commit()

    # 2. Setup Original Transcript
    orig_transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    db_session.add(orig_transcript)
    db_session.flush()

    orig_seg1 = TranscriptSegment(
        transcript_id=orig_transcript.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="um, uh, please contact Rahul at rahul.dev@example.com for access.",
        language_code="en",
        confidence=0.95,
    )
    orig_seg2 = TranscriptSegment(
        transcript_id=orig_transcript.id,
        sequence_order=1,
        start_time=5.0,
        end_time=10.0,
        text="His mobile is 9876543210 and w- we will meet tomorrow.",
        language_code="en",
        confidence=0.92,
    )
    db_session.add_all([orig_seg1, orig_seg2])
    db_session.commit()

    # 3. Derive CLEANED transcript
    cleaned_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED,
        db=db_session,
    )

    # 4. Derive REDACTED transcript
    redacted_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )

    # 5. Derive CLEANED_REDACTED transcript
    both_t = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    # --- VERIFICATIONS ---

    # A. Verify original transcript was NOT modified
    db_session.refresh(orig_transcript)
    assert orig_transcript.version == 1
    assert orig_transcript.transcript_type == TranscriptType.ORIGINAL.value
    orig_segs_db = list(orig_transcript.segments)
    assert len(orig_segs_db) == 2
    assert "um, uh," in orig_segs_db[0].text  # Imperfections preserved in original
    assert "rahul.dev@example.com" in orig_segs_db[0].text
    assert "9876543210" in orig_segs_db[1].text
    assert "w- we" in orig_segs_db[1].text

    # B. Verify CLEANED version
    assert cleaned_t.transcript_type == TranscriptType.CLEANED.value
    assert cleaned_t.version == 1
    assert cleaned_t.parent_transcript_id == orig_transcript.id
    cleaned_segs = list(cleaned_t.segments)
    assert "um, uh," not in cleaned_segs[0].text
    assert "w- we" not in cleaned_segs[1].text
    # Source segment IDs must point to original segments
    assert cleaned_segs[0].source_segment_id == orig_seg1.id
    assert cleaned_segs[1].source_segment_id == orig_seg2.id
    # Timestamps strictly preserved
    assert cleaned_segs[0].start_time == 0.0
    assert cleaned_segs[0].end_time == 5.0

    # C. Verify REDACTED version
    assert redacted_t.transcript_type == TranscriptType.REDACTED.value
    redacted_segs = list(redacted_t.segments)
    assert "[EMAIL_REDACTED]" in redacted_segs[0].text
    assert "rahul.dev@example.com" not in redacted_segs[0].text
    assert "[PHONE_REDACTED]" in redacted_segs[1].text
    assert "9876543210" not in redacted_segs[1].text
    assert redacted_segs[0].is_redacted is True
    assert "EMAIL" in json.loads(redacted_segs[0].redaction_metadata)

    # D. Verify CLEANED_REDACTED version
    assert both_t.transcript_type == TranscriptType.CLEANED_REDACTED.value
    both_segs = list(both_t.segments)
    assert "[EMAIL_REDACTED]" in both_segs[0].text
    assert "um, uh" not in both_segs[0].text
    assert "[PHONE_REDACTED]" in both_segs[1].text
    assert "w- we" not in both_segs[1].text

    # E. Test regeneration version incrementation
    cleaned_v2 = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED,
        db=db_session,
    )
    assert cleaned_v2.version == 2
    assert cleaned_v2.transcript_type == TranscriptType.CLEANED.value

    # F. Verify AuditLog entries created without sensitive values
    audit_logs = (
        db_session.query(AuditLog)
        .filter(AuditLog.organization_id == org.id)
        .all()
    )
    assert len(audit_logs) >= 3
    for log in audit_logs:
        assert "rahul.dev@example.com" not in log.details
        assert "9876543210" not in log.details


# --- 4. API Endpoint Integration Tests ---

def test_api_cleaning_and_redaction_endpoints(db_session, client):
    """Verify HTTP API endpoints for clean, redact, process, and retrieval."""
    org = Organization(name="API Phase 5 Org", slug="api-phase-5-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="API Privacy Test")
    db_session.add(meeting)
    db_session.commit()

    # Create original transcript
    orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
    )
    db_session.add(orig)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=3.0,
        text="um, please email test.user@company.com now.",
        language_code="en",
    )
    db_session.add(seg)
    db_session.commit()

    # User in org
    user = User(
        email="clean_admin@example.com",
        full_name="Clean Admin",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()

    headers = make_auth_headers(user, org)

    # 1. Clean endpoint
    resp_clean = client.post(
        f"/api/v1/transcription/{meeting.id}/clean",
        headers=headers,
        json={}
    )
    assert resp_clean.status_code == 201
    clean_data = resp_clean.json()
    assert clean_data["transcript_type"] == "CLEANED"
    assert clean_data["version"] == 1
    assert "um," not in clean_data["segments"][0]["text"]
    assert clean_data["parent_transcript_id"] == orig.id

    # 2. Redact endpoint
    resp_redact = client.post(
        f"/api/v1/transcription/{meeting.id}/redact",
        headers=headers,
        json={}
    )
    assert resp_redact.status_code == 201
    redact_data = resp_redact.json()
    assert redact_data["transcript_type"] == "REDACTED"
    assert "[EMAIL_REDACTED]" in redact_data["segments"][0]["text"]
    assert redact_data["segments"][0]["is_redacted"] is True

    # 3. Process (both clean and redact) endpoint
    resp_proc = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        headers=headers,
        json={"clean": True, "redact": True}
    )
    assert resp_proc.status_code == 201
    proc_data = resp_proc.json()
    assert proc_data["transcript_type"] == "CLEANED_REDACTED"
    assert "[EMAIL_REDACTED]" in proc_data["segments"][0]["text"]
    assert "um," not in proc_data["segments"][0]["text"]

    # 4. Retrieve all versions
    resp_all = client.get(
        f"/api/v1/transcription/{meeting.id}",
        headers=headers
    )
    assert resp_all.status_code == 200
    all_versions = resp_all.json()
    # Must have 4 versions: ORIGINAL, CLEANED, REDACTED, CLEANED_REDACTED
    types = {v["transcript_type"] for v in all_versions}
    assert "ORIGINAL" in types
    assert "CLEANED" in types
    assert "REDACTED" in types
    assert "CLEANED_REDACTED" in types

    # 5. Tenant isolation check: request from a different org header must fail with 404
    other_org = Organization(name="Other Org", slug="other-org", is_active=True)
    db_session.add(other_org)
    db_session.flush()

    other_user = User(
        email="other_admin@example.com",
        full_name="Other Admin",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(other_user)
    db_session.flush()

    m_other = OrganizationMembership(
        organization_id=other_org.id,
        user_id=other_user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m_other)
    db_session.commit()

    other_headers = make_auth_headers(other_user, other_org)
    resp_unauth = client.get(
        f"/api/v1/transcription/{meeting.id}",
        headers=other_headers
    )
    assert resp_unauth.status_code == 404


def test_missing_meeting_and_missing_transcript_handling(db_session, client):
    """Verify appropriate HTTP error codes for nonexistent meeting or missing source transcript."""
    org = Organization(name="Error Org", slug="error-org", is_active=True)
    db_session.add(org)
    db_session.flush()

    user = User(
        email="error_admin@example.com",
        full_name="Error Admin",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()

    headers = make_auth_headers(user, org)

    # 1. Nonexistent meeting ID -> 404
    resp_missing_m = client.post(
        "/api/v1/transcription/non-existent-id/clean",
        headers=headers,
        json={}
    )
    assert resp_missing_m.status_code == 404

    # 2. Existing meeting with NO transcript -> 400
    m_no_t = Meeting(organization_id=org.id, title="Empty Meeting")
    db_session.add(m_no_t)
    db_session.commit()

    resp_no_t = client.post(
        f"/api/v1/transcription/{m_no_t.id}/clean",
        headers=headers,
        json={}
    )
    assert resp_no_t.status_code == 400
    assert "No source transcript found" in resp_no_t.json()["detail"]


def test_hinglish_code_switching_cleaning_and_redaction():
    """Verify Hinglish code-switching preserves bilingual structure while redacting sensitive tokens."""
    hinglish_text = (
        "um, dekho yaar, project deadlines jaldi jaldi complete karni hai. "
        "Please send password=ProdSecret123 to rahul@office.in on WhatsApp +91 98765 12345."
    )
    # 1. Clean
    clean_res = clean_segment_text(hinglish_text, language_code="hi-en")
    assert "um" not in clean_res.cleaned_text.lower()
    assert "jaldi jaldi" in clean_res.cleaned_text  # Reduplication preserved
    assert "Dekho yaar" in clean_res.cleaned_text  # Capitalized sentence initial

    # 2. Redact
    redact_res = redact_text(clean_res.cleaned_text)
    assert redact_res.is_redacted is True
    assert "[SECRET_REDACTED]" in redact_res.redacted_text
    assert "[EMAIL_REDACTED]" in redact_res.redacted_text
    assert "[PHONE_REDACTED]" in redact_res.redacted_text
    assert "ProdSecret123" not in redact_res.redacted_text
    assert "rahul@office.in" not in redact_res.redacted_text
    assert "98765" not in redact_res.redacted_text


# --- 5. Security & Verification Tests ---

def test_cross_organization_job_access_blocked(db_session, client):
    """Verify that ProcessingJob endpoints strictly enforce tenant isolation and block cross-org access."""
    from app.models.meeting import ProcessingJob, JobType, JobStatus

    # Create Org A with meeting and job
    org_a = Organization(name="Tenant Alpha", slug="tenant-alpha", is_active=True)
    org_b = Organization(name="Tenant Bravo", slug="tenant-bravo", is_active=True)
    db_session.add_all([org_a, org_b])
    db_session.flush()

    user_a = User(email="usera_ti@orgalpha.com", full_name="User A", hashed_password=hash_password("Pass123!"), is_active=True)
    user_b = User(email="userb_ti@orgbravo.com", full_name="User B", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([user_a, user_b])
    db_session.flush()

    m_a = OrganizationMembership(organization_id=org_a.id, user_id=user_a.id, role=OrgRole.ORG_ADMIN.value, is_active=True)
    m_b = OrganizationMembership(organization_id=org_b.id, user_id=user_b.id, role=OrgRole.ORG_ADMIN.value, is_active=True)
    db_session.add_all([m_a, m_b])

    meeting_a = Meeting(organization_id=org_a.id, title="Alpha Secret Strategy")
    db_session.add(meeting_a)
    db_session.flush()

    job_a = ProcessingJob(
        meeting_id=meeting_a.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.COMPLETED.value,
        progress=100.0,
    )
    db_session.add(job_a)
    db_session.commit()

    headers_a = make_auth_headers(user_a, org_a)
    headers_b = make_auth_headers(user_b, org_b)

    # Org A access succeeds
    resp_alpha = client.get(
        f"/api/v1/transcription/jobs/{job_a.id}",
        headers=headers_a,
    )
    assert resp_alpha.status_code == 200
    assert resp_alpha.json()["job_id"] == job_a.id
    assert resp_alpha.json()["meeting_id"] == meeting_a.id

    # Org B access is blocked with 404 (tenant isolation)
    resp_bravo = client.get(
        f"/api/v1/transcription/jobs/{job_a.id}",
        headers=headers_b,
    )
    assert resp_bravo.status_code == 404
    assert "not found" in resp_bravo.json()["detail"].lower()


def test_audit_log_immutability_enforced(db_session):
    """Verify that AuditLog records cannot be modified or deleted via ORM operations."""
    import pytest

    org = Organization(name="Audit Compliance Org", slug="audit-compliance-org")
    db_session.add(org)
    db_session.commit()

    log = AuditLog(
        organization_id=org.id,
        action="TEST_ACTION",
        entity_type="Meeting",
        entity_id="test-meeting-1",
        details=json.dumps({"info": "initial audit data"}),
    )
    db_session.add(log)
    db_session.commit()

    # 1. Attempting to update must raise RuntimeError
    with pytest.raises(RuntimeError, match="AuditLog records are strictly append-only and cannot be updated"):
        log.action = "TAMPERED_ACTION"
        db_session.commit()

    db_session.rollback()

    # 2. Attempting to delete must raise RuntimeError
    fresh_log = db_session.query(AuditLog).filter(AuditLog.id == log.id).first()
    with pytest.raises(RuntimeError, match="AuditLog records are strictly append-only and cannot be deleted"):
        db_session.delete(fresh_log)
        db_session.commit()

    db_session.rollback()


def test_strict_privacy_safe_default_transcript_rules(db_session, client):
    """Verify strict privacy-safe rules for GET /default endpoint:
    1. Prefer CLEANED_REDACTED over all other versions.
    2. Otherwise select REDACTED when CLEANED_REDACTED is absent.
    3. CLEANED alone does NOT qualify (returns 404).
    4. ORIGINAL alone does NOT qualify (returns 404).
    5. No available redacted version returns 404 indicating no privacy-safe transcript is available.
    6. Ordinary version endpoint (/api/v1/transcription/{meeting_id}) remains available for specific versions.
    """
    org = Organization(name="Strict Privacy Org", slug="strict-privacy-org", is_active=True)
    db_session.add(org)
    db_session.flush()

    user = User(
        email="strict_admin@example.com",
        full_name="Strict Admin",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m)

    meeting = Meeting(organization_id=org.id, title="Strict Privacy Meeting")
    db_session.add(meeting)
    db_session.commit()

    headers = make_auth_headers(user, org)

    # Case A: When only ORIGINAL exists -> 404 (ORIGINAL alone does not qualify)
    orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(orig)
    db_session.flush()

    seg1 = TranscriptSegment(
        transcript_id=orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=2.0,
        text="Original raw text with sensitive@data.com",
        language_code="en",
    )
    db_session.add(seg1)
    db_session.commit()

    resp_orig_only = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_orig_only.status_code == 404
    assert "No privacy-safe (redacted) transcript available" in resp_orig_only.json()["detail"]

    # Verify ordinary endpoint still allows retrieving ORIGINAL
    resp_orig_ord = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=ORIGINAL", headers=headers)
    assert resp_orig_ord.status_code == 200
    assert len(resp_orig_ord.json()) == 1
    assert resp_orig_ord.json()[0]["transcript_type"] == "ORIGINAL"

    # Case B: When CLEANED is added (ORIGINAL + CLEANED exist) -> 404 (CLEANED alone does not qualify)
    cleaned = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED.value,
        version=1,
        parent_transcript_id=orig.id,
    )
    db_session.add(cleaned)
    db_session.flush()

    seg_c = TranscriptSegment(
        transcript_id=cleaned.id,
        sequence_order=0,
        start_time=0.0,
        end_time=2.0,
        text="Cleaned text with sensitive@data.com",
        language_code="en",
        source_segment_id=seg1.id,
    )
    db_session.add(seg_c)
    db_session.commit()

    resp_cleaned_only = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_cleaned_only.status_code == 404
    assert "No privacy-safe (redacted) transcript available" in resp_cleaned_only.json()["detail"]

    # Case C: When REDACTED is added (CLEANED_REDACTED absent) -> Returns REDACTED
    redacted = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.REDACTED.value,
        version=1,
        parent_transcript_id=orig.id,
    )
    db_session.add(redacted)
    db_session.flush()

    seg_r = TranscriptSegment(
        transcript_id=redacted.id,
        sequence_order=0,
        start_time=0.0,
        end_time=2.0,
        text="Original text with [EMAIL_REDACTED]",
        language_code="en",
        is_redacted=True,
        source_segment_id=seg1.id,
    )
    db_session.add(seg_r)
    db_session.commit()

    resp_redacted = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_redacted.status_code == 200
    assert resp_redacted.json()["transcript_type"] == "REDACTED"
    assert "[EMAIL_REDACTED]" in resp_redacted.json()["segments"][0]["text"]

    # Case D: When CLEANED_REDACTED is added -> CLEANED_REDACTED preferred over all others
    cleaned_redacted = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
        parent_transcript_id=orig.id,
    )
    db_session.add(cleaned_redacted)
    db_session.flush()

    seg_cr = TranscriptSegment(
        transcript_id=cleaned_redacted.id,
        sequence_order=0,
        start_time=0.0,
        end_time=2.0,
        text="Cleaned text with [EMAIL_REDACTED]",
        language_code="en",
        is_redacted=True,
        source_segment_id=seg1.id,
    )
    db_session.add(seg_cr)
    db_session.commit()

    resp_cr = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_cr.status_code == 200
    assert resp_cr.json()["transcript_type"] == "CLEANED_REDACTED"
    assert resp_cr.json()["version"] == 1

    # Case E: When CLEANED_REDACTED v2 is added -> Latest CLEANED_REDACTED preferred
    cleaned_redacted_v2 = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=2,
        parent_transcript_id=orig.id,
    )
    db_session.add(cleaned_redacted_v2)
    db_session.flush()

    seg_cr2 = TranscriptSegment(
        transcript_id=cleaned_redacted_v2.id,
        sequence_order=0,
        start_time=0.0,
        end_time=2.0,
        text="Cleaned v2 text with [EMAIL_REDACTED]",
        language_code="en",
        is_redacted=True,
        source_segment_id=seg1.id,
    )
    db_session.add(seg_cr2)
    db_session.commit()

    resp_cr_v2 = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_cr_v2.status_code == 200
    assert resp_cr_v2.json()["transcript_type"] == "CLEANED_REDACTED"
    assert resp_cr_v2.json()["version"] == 2


def test_evidence_linkage_and_metadata_preservation(db_session):
    """Verify source_segment_id links derived segments to source and preserves timing/speaker."""
    from app.models.transcription import Speaker

    org = Organization(name="Linkage Org", slug="linkage-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Linkage Verification")
    db_session.add(meeting)
    db_session.commit()

    speaker = Speaker(meeting_id=meeting.id, speaker_label="SPEAKER_01", display_name="Dr. Rao")
    db_session.add(speaker)
    db_session.commit()

    orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="hi-en",
    )
    db_session.add(orig)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=orig.id,
        speaker_id=speaker.id,
        sequence_order=0,
        start_time=12.5,
        end_time=18.0,
        text="um, please contact support@example.com for queries.",
        language_code="en",
        confidence=0.94,
    )
    db_session.add(seg)
    db_session.commit()

    # Derive CLEANED_REDACTED
    derived = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    derived_seg = derived.segments[0]
    # Check evidence linkage
    assert derived_seg.source_segment_id == seg.id
    assert derived_seg.source_segment.text == seg.text
    # Check preserved speaker, timing, sequence, confidence
    assert derived_seg.speaker_id == speaker.id
    assert derived_seg.start_time == 12.5
    assert derived_seg.end_time == 18.0
    assert derived_seg.sequence_order == 0
    assert derived_seg.confidence == 0.94
    assert derived_seg.is_redacted is True
    assert "[EMAIL_REDACTED]" in derived_seg.text

