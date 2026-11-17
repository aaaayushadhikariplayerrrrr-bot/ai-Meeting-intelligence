"""Phase 6D.4 — Aadhaar-like Identifier Detection & PII Redaction Tests.

Scope:
- Verifies detection of 12 consecutive digits, space-separated 4-4-4, hyphen-separated 4-4-4, and mixed separators.
- Verifies normalization of separators for validation without altering the original transcript.
- Verifies neutral redaction label [AADHAAR_REDACTED].
- Verifies false-positive protections:
  - Technical values: IP addresses and ports, version strings.
  - Dates and timestamps (ISO, calendar, numeric YYYYMMDDhhmm).
  - Phone numbers handled by phone recognizers (Indian mobile, STD landline, international).
  - Commercial identifiers: orders, invoices, transactions, accounts, tracking, tickets, POs, refs.
- Verifies multi-identifier segments in transcripts.
- Verifies database immutability: original transcript remains strictly unmodified.
- Verifies default transcript endpoint and RAG-safe paths never leak raw Aadhaar values.

DISCLAIMER & LIMITATIONS:
- Regex detection identifies Aadhaar-like 12-digit numeric structures; it does NOT verify genuine
  issuance with UIDAI or prove that an identifier is an active Aadhaar card.
- Deterministic, offline, zero-cloud dependency: no third-party cloud PII service is used.
- Complete detection cannot be guaranteed for numbers spoken phonetically or spelled out in
  Indic languages (Hindi, Gujarati, Hinglish) without verbal number normalization.
- Standalone 12-digit numbers without context are subject to unavoidable ambiguity (could be
  unlabelled serial/account numbers); conservative protections prioritize avoiding breakage of
  business and technical values while redacting personal identifiers.
"""

import json
import pytest
from app.models.identity import Organization, User
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import AuditLog
from app.services.privacy_redaction.redactor import redact_text
from app.services.transcript_processing.processor import derive_processed_transcript
from backend.tests.conftest import make_auth_headers


# --- 1. Unit Tests: Detection Variants ---

def test_aadhaar_twelve_consecutive_digits():
    """Verify 12 consecutive digits without separators are detected and redacted."""
    # Standard candidate starting with 2
    raw = "The customer stated identity number 234567890123 during check-in."
    res = redact_text(raw)
    assert res.is_redacted is True
    assert "234567890123" not in res.redacted_text
    assert "[AADHAAR_REDACTED]" in res.redacted_text
    assert "AADHAAR" in res.categories_detected
    assert "GOVT_ID" in res.categories_detected

    # Candidate starting with 9
    raw2 = "Verification passed for 987654321098."
    res2 = redact_text(raw2)
    assert res2.is_redacted is True
    assert "987654321098" not in res2.redacted_text
    assert "[AADHAAR_REDACTED]" in res2.redacted_text


def test_aadhaar_separated_by_spaces():
    """Verify 4-4-4 formatted numbers separated by single and multiple spaces."""
    # Single space
    raw = "Citizen Aadhaar number is 2345 6789 0123."
    res = redact_text(raw)
    assert res.is_redacted is True
    assert "2345 6789 0123" not in res.redacted_text
    assert "[AADHAAR_REDACTED]" in res.redacted_text

    # Multiple spaces (common in ASR transcripts)
    raw_multi = "Participant provided  3456  7890  1234  for KYC."
    res_multi = redact_text(raw_multi)
    assert res_multi.is_redacted is True
    assert "3456  7890  1234" not in res_multi.redacted_text
    assert "[AADHAAR_REDACTED]" in res_multi.redacted_text


def test_aadhaar_separated_by_hyphens_and_mixed():
    """Verify hyphen-separated and mixed space/hyphen formatting variants."""
    # Hyphen-separated
    raw_hyphen = "Identity record: 2345-6789-0123."
    res_hyphen = redact_text(raw_hyphen)
    assert res_hyphen.is_redacted is True
    assert "2345-6789-0123" not in res_hyphen.redacted_text
    assert "[AADHAAR_REDACTED]" in res_hyphen.redacted_text

    # Mixed separators (space and hyphen)
    raw_mix1 = "Card number 2345 6789-0123 submitted."
    res_mix1 = redact_text(raw_mix1)
    assert res_mix1.is_redacted is True
    assert "2345 6789-0123" not in res_mix1.redacted_text
    assert "[AADHAAR_REDACTED]" in res_mix1.redacted_text

    raw_mix2 = "Card number 2345-6789 0123 submitted."
    res_mix2 = redact_text(raw_mix2)
    assert res_mix2.is_redacted is True
    assert "2345-6789 0123" not in res_mix2.redacted_text
    assert "[AADHAAR_REDACTED]" in res_mix2.redacted_text


# --- 2. Unit Tests: False-Positive Protection & Negative Cases ---

def test_invalid_lengths_not_redacted():
    """Verify numbers with invalid digit counts are strictly NOT redacted as Aadhaar."""
    # 11 digits continuous
    t11_c = "Candidate code 23456789012 is pending review."
    r11_c = redact_text(t11_c)
    assert "[AADHAAR_REDACTED]" not in r11_c.redacted_text
    assert "23456789012" in r11_c.redacted_text

    # 11 digits grouped (4-4-3)
    t11_g = "Code 2345 6789 012 was incomplete."
    r11_g = redact_text(t11_g)
    assert "[AADHAAR_REDACTED]" not in r11_g.redacted_text
    assert "2345 6789 012" in r11_g.redacted_text

    # 13 digits continuous
    t13_c = "Long tracking barcode 2345678901234 recorded."
    r13_c = redact_text(t13_c)
    assert "[AADHAAR_REDACTED]" not in r13_c.redacted_text
    assert "2345678901234" in r13_c.redacted_text

    # 13 digits grouped (4-4-5)
    t13_g = "Value 2345 6789 01234 recorded."
    r13_g = redact_text(t13_g)
    assert "[AADHAAR_REDACTED]" not in r13_g.redacted_text
    assert "2345 6789 01234" in r13_g.redacted_text

    # 16-digit card groups (4-4-4-4) should not have sub-chunks matched as Aadhaar
    t16 = "Card 4532 1234 5678 9012 is on file."
    r16 = redact_text(t16)
    assert "[AADHAAR_REDACTED]" not in r16.redacted_text


def test_common_phone_numbers_handled_by_phone_recognizer():
    """Verify phone numbers are labeled [PHONE_REDACTED] and never misidentified as Aadhaar."""
    # 10-digit Indian mobile
    t_mobile = "Please call 9876543210 for questions."
    r_mobile = redact_text(t_mobile)
    assert "[PHONE_REDACTED]" in r_mobile.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_mobile.redacted_text
    assert "PHONE" in r_mobile.categories_detected

    # Indian mobile with +91
    t_plus91 = "Reach out at +91 98765 43210 or +919876543210."
    r_plus91 = redact_text(t_plus91)
    assert "[PHONE_REDACTED]" in r_plus91.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_plus91.redacted_text

    # Indian STD landline
    t_std = "Delhi office landline is 011-23456789."
    r_std = redact_text(t_std)
    assert "[PHONE_REDACTED]" in r_std.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_std.redacted_text

    # International number
    t_intl = "US support is +1-800-555-0199."
    r_intl = redact_text(t_intl)
    assert "[PHONE_REDACTED]" in r_intl.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_intl.redacted_text

    # 12 digits preceded by phone context
    t_ctx = "Call 234567890123 immediately."
    r_ctx = redact_text(t_ctx)
    assert "[AADHAAR_REDACTED]" not in r_ctx.redacted_text


def test_dates_and_timestamps_preserved():
    """Verify standard dates, ISO timestamps, and 12-digit numeric timestamps are preserved."""
    # Standard ISO date
    t_date = "The session was conducted on 2026-11-15."
    r_date = redact_text(t_date)
    assert "2026-11-15" in r_date.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_date.redacted_text

    # ISO timestamp with time and zone
    t_iso = "Log timestamp: 2026-11-15T14:30:00Z."
    r_iso = redact_text(t_iso)
    assert "2026-11-15T14:30:00Z" in r_iso.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_iso.redacted_text

    # Formatted date time
    t_dt = "Created at 2026-11-15 14:30:00 in system."
    r_dt = redact_text(t_dt)
    assert "2026-11-15 14:30:00" in r_dt.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_dt.redacted_text

    # Numeric YYYYMMDDhhmm timestamp with timezone
    t_num_ts = "Meeting scheduled at 202611151200 UTC."
    r_num_ts = redact_text(t_num_ts)
    assert "202611151200 UTC" in r_num_ts.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_num_ts.redacted_text

    # Numeric timestamp with explicit prefix
    t_num_ctx = "Logged at timestamp: 202611151200."
    r_num_ctx = redact_text(t_num_ctx)
    assert "202611151200" in r_num_ctx.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_num_ctx.redacted_text


def test_ip_addresses_and_technical_versions_preserved():
    """Verify IP addresses, ports, and version numbers are preserved."""
    # Standard IPv4
    t_ip = "Database connected at 192.168.1.1."
    r_ip = redact_text(t_ip)
    assert "192.168.1.1" in r_ip.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_ip.redacted_text

    # IPv4 with port
    t_port = "Backend listening on 192.168.1.100:8080 and localhost:8000."
    r_port = redact_text(t_port)
    assert "192.168.1.100:8080" in r_port.redacted_text
    assert "localhost:8000" in r_port.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_port.redacted_text

    # Semantic versioning
    t_ver = "Upgraded to release v1.2.3.4 and library 12.34.56.78."
    r_ver = redact_text(t_ver)
    assert "v1.2.3.4" in r_ver.redacted_text
    assert "12.34.56.78" in r_ver.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_ver.redacted_text

    # Version context prefix
    t_ver_ctx = "Deployed build Version: 234567890123."
    r_ver_ctx = redact_text(t_ver_ctx)
    assert "234567890123" in r_ver_ctx.redacted_text
    assert "[AADHAAR_REDACTED]" not in r_ver_ctx.redacted_text


def test_contextually_identifiable_order_and_invoice_identifiers():
    """Verify commercial, billing, and transactional identifiers are preserved."""
    examples = [
        ("The Order #234567890123 was processed.", "Order #234567890123"),
        ("Refer to Order ID: 2345-6789-0123.", "Order ID: 2345-6789-0123"),
        ("Invoice: 2345-6789-0123 is attached.", "Invoice: 2345-6789-0123"),
        ("Payment for Invoice #234567890123 received.", "Invoice #234567890123"),
        ("Vendor billed INV-234567890123 today.", "INV-234567890123"),
        ("Transaction ID 234567890123 completed.", "Transaction ID 234567890123"),
        ("Txn # 2345 6789 0123 status success.", "Txn # 2345 6789 0123"),
        ("Check Account number 234567890123 balance.", "Account number 234567890123"),
        ("Shipment Tracking # 234567890123 updated.", "Tracking # 234567890123"),
        ("Support Ticket ID: 234567890123 opened.", "Ticket ID: 234567890123"),
        ("PO # 234567890123 approved.", "PO # 234567890123"),
        ("Quote Ref: 234567890123 valid for 30 days.", "Ref: 234567890123"),
    ]
    for text, expected_preserved in examples:
        res = redact_text(text)
        assert "[AADHAAR_REDACTED]" not in res.redacted_text, f"Erroneously redacted in: '{text}'"
        assert expected_preserved in res.redacted_text, f"Missing preserved text in: '{text}'"


def test_multiple_identifiers_in_single_transcript():
    """Verify composite transcripts with Aadhaar, phone, email, order, and technical data."""
    composite = (
        "Client Rajesh (email: rajesh.kumar@example.com, mobile: +91 98765 43210) "
        "provided Aadhaar 2345 6789 0123 for Order #987654321098. "
        "Event logged at 2026-11-15T10:00:00Z from IP 192.168.1.50:8000."
    )
    res = redact_text(composite)
    assert res.is_redacted is True
    # Sensitive items redacted
    assert "[EMAIL_REDACTED]" in res.redacted_text
    assert "[PHONE_REDACTED]" in res.redacted_text
    assert "[AADHAAR_REDACTED]" in res.redacted_text
    assert "rajesh.kumar@example.com" not in res.redacted_text
    assert "98765 43210" not in res.redacted_text
    assert "2345 6789 0123" not in res.redacted_text

    # Business and technical items strictly preserved
    assert "Order #987654321098" in res.redacted_text
    assert "2026-11-15T10:00:00Z" in res.redacted_text
    assert "192.168.1.50:8000" in res.redacted_text

    # Category tracking
    assert "EMAIL" in res.categories_detected
    assert "PHONE" in res.categories_detected
    assert "AADHAAR" in res.categories_detected
    assert "GOVT_ID" in res.categories_detected


# --- 3. Database & Transcript Lineage Tests ---

def test_original_transcript_unmodified_in_database(db_session):
    """Verify the original ASR transcript in database is NEVER modified when deriving redacted version."""
    # 1. Setup organization and meeting
    org = Organization(name="Aadhaar Privacy Org", slug="aadhaar-privacy-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Customer Onboarding Session")
    db_session.add(meeting)
    db_session.commit()

    raw_aadhaar_text = "Applicant stated Aadhaar is 2345 6789 0123 and phone is 9876543210."

    # 2. Persist ORIGINAL transcript
    orig_t = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    db_session.add(orig_t)
    db_session.flush()

    orig_seg = TranscriptSegment(
        transcript_id=orig_t.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text=raw_aadhaar_text,
        language_code="en",
        confidence=0.96,
        is_redacted=False,
    )
    db_session.add(orig_seg)
    db_session.commit()

    # 3. Derive REDACTED transcript
    derived_redacted = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )
    db_session.commit()

    # 4. Strict assertions: Original transcript MUST remain untouched
    db_session.expire_all()
    queried_orig = db_session.query(Transcript).filter(Transcript.id == orig_t.id).one()
    queried_orig_seg = queried_orig.segments[0]

    assert queried_orig_seg.text == raw_aadhaar_text
    assert queried_orig_seg.is_redacted is False
    assert "2345 6789 0123" in queried_orig_seg.text

    # 5. Derived transcript contains redacted placeholder
    queried_derived = db_session.query(Transcript).filter(Transcript.id == derived_redacted.id).one()
    derived_seg = queried_derived.segments[0]

    assert derived_seg.is_redacted is True
    assert "[AADHAAR_REDACTED]" in derived_seg.text
    assert "[PHONE_REDACTED]" in derived_seg.text
    assert "2345 6789 0123" not in derived_seg.text
    assert "9876543210" not in derived_seg.text

    # Lineage verification
    assert queried_derived.parent_transcript_id == orig_t.id
    assert derived_seg.source_segment_id == orig_seg.id

    # Metadata verification
    meta = json.loads(derived_seg.redaction_metadata)
    assert "AADHAAR" in meta
    assert "GOVT_ID" in meta
    assert "PHONE" in meta


# --- 4. API Endpoint Integration Tests ---

def test_default_endpoint_never_exposes_raw_aadhaar(client, db_session):
    """Verify GET /{meeting_id}/default strictly serves privacy-safe transcripts without Aadhaar digits."""
    # 1. Setup Org and User
    org = Organization(name="Safe API Org", slug="safe-api-org")
    user = User(
        email="auditor@safe-api-org.com",
        full_name="Auditor User",
        hashed_password="dummy_password_hash",
        is_active=True,
        token_version=1,
    )
    db_session.add_all([org, user])
    db_session.commit()

    # Link user to organization via membership
    from app.models.identity import OrganizationMembership, OrgRole
    membership = OrganizationMembership(
        user_id=user.id,
        organization_id=org.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(membership)
    db_session.commit()

    headers = make_auth_headers(user, org)

    meeting = Meeting(organization_id=org.id, title="API Endpoint Aadhaar Verification")
    db_session.add(meeting)
    db_session.commit()

    raw_aadhaar_val = "2345 6789 0123"
    raw_text = f"Meeting notes: user verified Aadhaar {raw_aadhaar_val}."

    # A. Only ORIGINAL exists -> endpoint MUST return 404 (refusing to leak unredacted data)
    orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(orig)
    db_session.flush()

    seg_orig = TranscriptSegment(
        transcript_id=orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=3.0,
        text=raw_text,
        language_code="en",
    )
    db_session.add(seg_orig)
    db_session.commit()

    resp_orig_only = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_orig_only.status_code == 404
    assert "No privacy-safe (redacted) transcript available" in resp_orig_only.json()["detail"]

    # B. Derive REDACTED version via processor
    derived = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )
    db_session.commit()

    # C. Default endpoint now returns 200 with redacted transcript
    resp_default = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=headers)
    assert resp_default.status_code == 200
    data = resp_default.json()

    assert data["transcript_type"] == "REDACTED"
    segment_text = data["segments"][0]["text"]
    assert "[AADHAAR_REDACTED]" in segment_text
    assert raw_aadhaar_val not in segment_text
    assert "2345" not in segment_text
    assert "6789" not in segment_text
    assert "0123" not in segment_text
