# MEMORA Security & Privacy Architecture

This document defines the security policies, tenant isolation model, privacy enforcement controls, and known operational boundaries of the Multilingual Meeting Intelligence System (MEMORA).

---

## 1. Authentication & Session Security

* **Password Hashing**: Implemented via Argon2id (`argon2-cffi`) using parameters tuned to prevent GPU-accelerated dictionary attacks:
  * Memory cost: `102400` KiB (100 MB)
  * Time cost: `2` iterations
  * Parallelism: `8` threads
* **JWT Access Tokens**:
  * Signed using HMAC-SHA256 (`HS256`) with a cryptographically generated 256-bit `SECRET_KEY`.
  * Short-lived access tokens (`ACCESS_TOKEN_EXPIRE_MINUTES = 60`).
  * Per-user `token_version` tracking in SQLite: upon logout, the user's `token_version` is atomically incremented. All previously issued tokens are immediately rejected with HTTP 401 Unauthorized.
* **Brute-Force & Rate Limiting**:
  * Process-local in-memory sliding-window rate limiters protect sensitive endpoints:
    * Login: 5 attempts per 60 seconds per IP.
    * Registration: 3 accounts per 60 seconds per IP.
    * Memory Retrieval: 60 queries per 60 seconds per IP.

---

## 2. Multi-Tenant & Organization Isolation

* **Tenant Context Injection**: Every protected API route depends on `get_current_tenant_membership(current_user, org_id, db)`:
  * Validates that the authenticated user possesses an active membership in the requested organization (`X-Organization-Id` header).
  * Returns a validated `TenantContext` containing both user and organization entities.
* **Strict Database Query Scoping**:
  * Every query for meetings, media assets, transcripts, decisions, commitments, action items, and vector memory chunks enforces `organization_id == tenant.organization.id`.
  * Cross-tenant access attempts return HTTP 404 Not Found or HTTP 403 Forbidden without leaking existence of resources.

---

## 3. Role-Based Access Control (RBAC) & Transcript Privacy

MEMORA enforces dual-layer transcript exposure:

| Transcript Type | Description | ORG_ADMIN | MEMBER |
| :--- | :--- | :--- | :--- |
| **ORIGINAL** | Raw ASR output with spoken text | Access allowed (Audited) | HTTP 403 Forbidden |
| **CLEANED** | Filler/stutter removed, unredacted | Access allowed (Audited) | HTTP 403 Forbidden |
| **REDACTED** | Raw ASR output with PII masked | Access allowed | Access allowed |
| **CLEANED_REDACTED** | Cleaned text with PII masked | Access allowed | Access allowed |

* **Safe Selection Enforcement**: The `/meetings/{id}/safe` endpoint strictly selects `CLEANED_REDACTED` or `REDACTED`, throwing an error if only unredacted transcripts exist.
* **Audit Logging**: Any retrieval of unredacted transcripts by an `ORG_ADMIN` records an immutable row in `audit_logs` capturing user ID, meeting ID, retrieved transcript IDs, client IP, and timestamp.

---

## 4. Privacy Redaction Engine

* **Automated PII Masking**:
  * Regex and pattern analyzers mask sensitive entity tokens:
    * Indian Identity: Aadhaar Numbers (`\b[2-9]\d{3}\s\d{4}\s\d{4}\b`), PAN cards.
    * Contact Details: Email addresses, international & domestic phone numbers.
    * Financial Data: Credit/Debit card numbers (Luhn validated).
    * Secrets & Credentials: JWT tokens, Bearer authorization tokens, API keys, passwords.
* **Legal Conditioning Preservation**:
  * The redactor explicitly preserves statutory negations and contractual qualification clauses (`shall`, `must`, `may`, `must not`, `unless`, `except`, `subject to`, `provided that`).

---

## 5. Media Retention & Safe Deletion Lifecycle

* **Automated Post-Transcription Deletion**:
  * When `AUTO_DELETE_RAW_POST_TRANSCRIPTION=True`, the background worker unlinks the original uploaded media and intermediate normalized 16 kHz WAVs immediately after transcript segments are committed to the database.
  * Deletion operations are recorded in `audit_logs` with `deletion_type="application_unlink"`.
  * If file unlinking encounters an OS error, `is_raw_deleted` remains `False`, the failure is logged, and the completed transcript is retained intact.
* **Scheduled Cleanup**:
  * Background `RetentionScheduler` polls every 24 hours to clean up orphaned media assets exceeding `RETENTION_DAYS=30`.

---

## 6. Honest Security Boundaries & Limitations

1. **Regex-Based Redaction**: Rule-based PII detection catches structured identifiers (emails, cards, IDs) but cannot detect arbitrary colloquial names or contextual spoken secrets without specialized named entity recognition (NER).
2. **Application-Level File Unlinking**: File deletion executes OS-level unlinking (`Path.unlink()`). It does not perform multi-pass cryptographic disk overwrites (shredding).
3. **In-Memory Rate Limiting**: Rate limiters are process-local; in horizontally scaled multi-worker deployments, a centralized cache (such as Redis) would be required.
4. **Local Hardware Boundary**: The application is intended for private local deployment. Production exposure to public networks requires TLS termination, a reverse proxy (e.g. Nginx/Caddy), and strict CORS origin configuration.
