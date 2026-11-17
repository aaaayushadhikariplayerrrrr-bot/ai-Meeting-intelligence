# MEMORA: Multilingual Meeting Intelligence & Organizational Memory System

A privacy-first, locally hosted meeting intelligence and cross-meeting organizational memory platform designed for English, Hindi, and Hinglish (code-switched) business conversations.

---

## 1. Problem Statement & Technical Novelty

Traditional meeting transcription tools are typically single-meeting note-takers: they transcribe audio into flat text, provide a high-level summary, and discard the relationship between meetings.

**MEMORA** addresses four systemic shortcomings of modern meeting platforms:
1. **Cross-Meeting Memory**: Decisions and action items do not exist in isolation. MEMORA links decisions chronologically across meetings, tracks whether subsequent meetings supersede or contradict prior commitments, and alerts users to conflicting mandates.
2. **Privacy-Safe Ingestion & Raw Audio Deletion**: Under strict privacy policies, raw media (voice/video recordings) contains sensitive biometric identifiers. MEMORA automatically unlinks and deletes raw audio and intermediate WAVs immediately upon verified transcript persistence, preserving only timestamped text segments, PII-redacted derived versions, and vector embeddings.
3. **Multilingual & Code-Switched Support**: Designed to handle code-switched Hindi-English (Hinglish) discussions using Faster-Whisper on NVIDIA RTX GPUs, alongside semantic multilingual vector embeddings (`multilingual-e5-small`) running locally on CPU.
4. **Verifiable Evidence Navigation**: Every decision, action item, conflict, and Ask AI answer is grounded in exact transcript segments with segment-level timestamps. Clicking a citation jumps directly to the cited timestamp in the privacy-safe transcript.

---

## 2. Core Capabilities

* **Multi-Format Ingestion**: Upload audio (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`) or video (`.mp4`, `.webm`, `.mov`, `.mkv`), or ingest authorized YouTube streams. Audio is automatically extracted and normalized to 16 kHz mono WAV via FFmpeg.
* **GPU-Accelerated Transcription**: Accelerated Faster-Whisper (`large-v3-turbo` with CUDA float16/int8 fallback) with custom chunking, overlap deduplication, and progress callbacks.
* **Dual-Layer Privacy Redaction**: Automatic rule-based and regex PII detection redaction (Aadhaar IDs, PAN cards, credit cards, emails, phone numbers, API keys, JWT tokens) while strictly preserving legal conditioning words (`shall`, `must not`, `unless`, `subject to`).
* **Role-Based Access Control**:
  * `ORG_ADMIN`: Full access to original audio management, original transcripts, cleaned/redacted versions, and system diagnostics.
  * `MEMBER`: Access strictly limited to privacy-safe (redacted) transcripts, meeting intelligence, and cross-meeting organizational memory.
* **Cross-Meeting Decision & Commitment Lineage**:
  * **Decision History**: Track Active, Superseded, and Revoked decisions with explicit lineage trees.
  * **Commitment Tracking**: Monitor action items across meetings with overdue alerts, owner filters, and meeting origin links.
  * **Potential Conflicts**: Heuristic and evidence-backed detection of contradictory decisions across meetings.
  * **Unresolved Questions**: Track unanswered inquiries from past meetings until explicitly resolved.
* **Ask AI & Hybrid Retrieval**:
  * Hybrid retrieval combining BM25/SQLite FTS5 full-text keyword search and dense semantic vector similarity (`multilingual-e5-small`).
  * Honest provider attribution (deterministic rule-based synthesis when local generative LLMs are unavailable, abstaining with `INSUFFICIENT_EVIDENCE` when queries lack grounded proof).

---

## 3. Technology Stack

### Backend
* **Language & Runtime**: Python 3.13 (64-bit)
* **API Framework**: FastAPI, Pydantic v2, Starlette, Uvicorn
* **Database & ORM**: SQLite 3 with WAL mode (`PRAGMA journal_mode=WAL`), FTS5 full-text search triggers, SQLAlchemy 2.0 Declarative ORM, Alembic migrations
* **Speech Recognition**: Faster-Whisper / CTranslate2 on NVIDIA CUDA 12.8, FFmpeg / FFprobe
* **Embeddings & Vector Search**: HuggingFace `intfloat/multilingual-e5-small` (384-dimensional dense vectors), PyTorch CPU inference, NumPy cosine similarity
* **Authentication & Security**: Argon2id password hashing, JWT Bearer tokens with per-user `token_version` revocation on logout, rate-limiting on authentication endpoints

### Frontend
* **Framework**: React 19, TypeScript, Vite
* **UI Components & Icons**: Lucide React, vanilla modern CSS with design tokens
* **State & Networking**: Custom React Context (`AuthContext`), Fetch API with `AbortController` cancellation for tenant isolation

---

## 4. Hardware & Software Requirements

| Component | Minimum Specification | Verified System |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 or Ubuntu 22.04 LTS | Windows 11 (64-bit) |
| **CPU** | 4 Cores, 8 Threads (x86_64) | AMD Ryzen 7 260 (8 Cores, 16 Threads) |
| **RAM** | 16 GB System Memory | 16 GB DDR5 |
| **GPU (ASR)** | NVIDIA GPU with 6+ GB VRAM & CUDA compute capability 8.0+ | NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM) |
| **Python** | Python 3.11 – 3.13 | Python 3.13.16 |
| **Node.js** | Node.js v20+ & npm 10+ | Node.js v26.13.1 & npm 11.20.0 |
| **FFmpeg** | FFmpeg 6.0+ with ffprobe in PATH | FFmpeg 9.0.2 Gyan Full Build |

---

## 5. Quick Start & Setup Instructions

### Step 1: Clone Repository & Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd "Multilingual Meeting Intelligence System"

# Setup Python Virtual Environment in backend
cd backend
python -m venv venv
venv\Scripts\activate   # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` in both `backend` and `frontend`:
```bash
# In backend/
copy .env.example .env   # On Linux/macOS: cp .env.example .env

# Generate a cryptographically secure 256-bit secret for backend/.env:
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
# Update SECRET_KEY in backend/.env with the generated value
```

In `frontend/`:
```bash
cd ../frontend
copy .env.example .env
npm install
```

### Step 3: Run Database Migrations
```bash
# In backend/ (with venv activated)
python -m alembic upgrade head
```

### Step 4: Start the Backend & Frontend Servers

**Terminal 1 — Backend (Port 8000)**:
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend (Port 5173)**:
```bash
cd frontend
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 6. How to Use the System

1. **Register / Login**:
   * Create an account. An organization is created automatically (e.g. `Acme Corp`), and you are assigned `ORG_ADMIN`.
2. **Upload Meeting**:
   * Navigate to **Upload Meeting**. Select an audio or video file.
   * Background extraction validates codecs, normalizes audio to 16 kHz mono WAV, and queues the transcription job.
3. **View Transcripts & Intelligence**:
   * Monitor live progress on the dashboard.
   * View the original transcript (admin only) or privacy-redacted transcript.
   * Review structured topics, executive summaries, decisions, action items, and risks.
4. **Explore Cross-Meeting Intelligence**:
   * Switch to **Cross-Meeting Intelligence** to review decision timelines, supersede decisions, track open commitments, and inspect conflict warnings.
5. **Ask AI**:
   * Switch to **Organizational Memory** -> **Ask AI**.
   * Ask natural-language questions across past meetings (e.g., *"What did we decide about database indexing?"*).
   * Review citations and click to navigate to the exact transcript timestamp.

---

## 7. Privacy, Consent & Media Retention Policy

* **Pre-Transcription**: Raw media is streamed safely to tenant-isolated storage folders (`data/storage/<meeting_id>/`).
* **Post-Transcription Deletion**: Once transcription completes and segments are safely persisted in SQLite, `AUTO_DELETE_RAW_POST_TRANSCRIPTION=True` automatically unlinks and removes raw media files and intermediate WAVs from disk.
* **Audit Trail**: Every deletion operation logs an unalterable `AuditLog` entry recording timestamp, asset ID, user ID, and `deletion_type="application_unlink"`.
* **Local Inference & Zero Cloud API Dependence**: Core processing (speech recognition, PII redaction, embedding generation, vector storage, and hybrid search) runs strictly on local workstation hardware. No audio, transcripts, or queries are sent to external cloud LLM APIs. (Note: YouTube stream ingestion connects to YouTube to fetch authorized video streams if that input method is explicitly selected).

---

## 8. Test Commands

```bash
# Run all backend tests (from backend/ directory):
cd backend
set PYTHONPATH=..;.
venv\Scripts\python.exe -m pytest tests/ -v

# Run frontend tests:
cd frontend
npm run test

# Run frontend production build:
npm run build

# Check database schema drift:
cd backend
venv\Scripts\python.exe -m alembic check
```

---

## 9. Known Boundaries & Limitations

1. **Deterministic Fallback vs. Generative LLM**: In the absence of a local Ollama or high-VRAM LLM installation, MEMORA employs an honest deterministic rule-based answer generator. It synthesizes answers strictly from retrieved citation chunks rather than generating freeform synthetic prose.
2. **Application-Level Deletion vs. Forensic Sanitization**: Audio deletion unlinks files via standard OS calls (`Path.unlink()`). It does not perform multi-pass cryptographic disk sanitization or DoD 5220.22-M drive shredding.
3. **Regex PII Coverage**: Redaction rules target standard structured PII (Aadhaar, PAN, SSN, emails, phones, API keys, tokens). Unstructured contextual mentions of private data not matching known patterns may require manual redaction review.
4. **Speaker Diarization**: When speaker diarization models are unconfigured, speaker metadata defaults honestly to `null` (`NullDiarizationProvider`) rather than generating fabricated speaker labels.
5. **Multilingual ASR Scope**: Faster-Whisper provides multilingual recognition for English, Hindi, and Hinglish. Synthetic test suites pass, but real-world accuracy will naturally vary with heavy regional accents, background noise, and overlapping speakers.
6. **Local Deployment vs. Public Production**: MEMORA is hardened for local demonstration and intranet team environments. Deployment on the public internet requires an enterprise reverse proxy (e.g. Nginx or Caddy) with TLS termination, centralized Redis for distributed rate limiting, and PostgreSQL for horizontal multi-writer scaling.
7. **SQLite Single-Writer Concurrency**: SQLite in WAL mode permits unlimited concurrent readers, while write transactions are queued sequentially backed by a 30-second busy timeout. High-concurrency multi-worker workloads should consider PostgreSQL for multi-writer scaling.

---

## 10. Troubleshooting & FAQ

* **CUDA Out of Memory during Transcription**:
  Ensure `ASR_NUM_WORKERS=1` in `backend/app/core/config.py`. Faster-Whisper releases model memory when idle, and CPU thread limits (`ASR_CPU_THREADS=4`) prevent memory bloat.
* **Missing DLL Errors on Windows (cublas64_12.dll)**:
  MEMORA includes an automated dynamic DLL discovery mechanism in `backend/app/services/transcription/gpu_manager.py` that links active PyTorch CUDA libraries (`torch/lib`) into the Windows process search path. Ensure PyTorch with CUDA is installed in `backend/venv`.
* **Database Locked Errors**:
  SQLite WAL mode is enabled automatically. Verify that `PRAGMA busy_timeout=30000` is active in `backend/app/database/session.py`.
