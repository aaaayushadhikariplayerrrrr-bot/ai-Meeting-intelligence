# Multilingual Meeting Intelligence System

A privacy-aware, multilingual AI Meeting Intelligence and Organizational Memory System engineered to ingest, transcribe, analyze, and retain knowledge across meetings conducted in English, Hindi, Gujarati, and Hinglish (code-switched conversations).

---

## 1. Verified Hardware & Software Environment

| Component | Verified Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (64-bit) |
| **CPU** | AMD Ryzen 7 260 w/ Radeon 780M Graphics (8 Cores, 16 Logical Processors) |
| **Memory (RAM)** | 16 GB Physical RAM (15.31 GiB usable) |
| **Storage** | 812+ GB Free Space on Drive C: |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM) |
| **NVIDIA Driver** | 621.01 (CUDA UMD 13.5) |
| **PyTorch & CUDA** | PyTorch 2.11.0+cu128 (Active CUDA 12.8 device acceleration) |
| **FFmpeg & FFprobe** | FFmpeg 9.0.2-full_build (Gyan.dev build with libwhisper & audio decoders) |
| **Python** | Python 3.13.16 (`C:\Users\gurpr\AppData\Local\Programs\Python\Python313\python.exe`) |
| **Node.js & npm** | Node.js v26.13.1 (LTS) & npm 11.20.0 |
| **Ollama** | Not installed yet (reserved for Phase 5 LLM pipeline) |

---

## 2. Project Directory Structure

```text
Multilingual Meeting Intelligence System/
├── backend/
│   ├── alembic/                       # Alembic database migrations
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── health.py          # /health & /api/v1/health endpoints
│   │   │   │   ├── media.py           # Upload, YouTube, Meeting, Job & Deletion endpoints
│   │   │   │   └── transcription.py   # Start transcription, poll jobs, retrieve segments, HW diagnostics
│   │   │   ├── deps.py                # Tenant isolation & dev user dependencies
│   │   │   └── router.py              # Main API router aggregator
│   │   ├── core/
│   │   │   └── config.py              # Storage paths, upload limits, ASR model & GPU compute settings
│   │   ├── database/
│   │   │   ├── base.py                # SQLAlchemy 2.0 DeclarativeBase
│   │   │   └── session.py             # SQLite engine with PRAGMA foreign_keys=ON
│   │   ├── models/                    # Domain ORM models (Meeting, MediaAsset, Transcript, TranscriptSegment)
│   │   ├── schemas/                   # Pydantic schemas (MediaAssetResponse, MeetingResponse)
│   │   ├── services/                  # Modular services
│   │   │   ├── audio_extraction.py    # FFmpeg audio normalization (16kHz mono WAV)
│   │   │   ├── job_manager.py         # ProcessingJob status & progress tracking
│   │   │   ├── lifecycle.py           # Safe raw media deletion & AuditLog recording
│   │   │   ├── media_metadata.py      # FFprobe metadata extraction (duration, codecs, fps)
│   │   │   ├── media_storage.py       # Traversal-safe streaming storage & SHA-256
│   │   │   ├── media_validation.py    # Extensions, FFprobe content checks, YouTube URLs
│   │   │   ├── youtube_ingestion.py   # yt-dlp stream extraction with authorization check
│   │   │   └── transcription/         # Phase 4 ASR Speech Recognition Engine
│   │   │       ├── base.py            # ASREngine ABC, ASRResult, ASRSegment, TranscriptionOptions
│   │   │       ├── chunking.py        # Long-audio chunk calculation, WAV slicing, overlap deduplication
│   │   │       ├── diarization.py     # DiarizationProvider ABC & NullDiarizationProvider (honest, no fake IDs)
│   │   │       ├── faster_whisper_engine.py # Faster-Whisper + CTranslate2 on NVIDIA GPU & CPU fallback
│   │   │       ├── gpu_manager.py     # Dynamic Windows DLL discovery (torch/lib + CUDA) & VRAM monitoring
│   │   │       ├── job_handler.py     # Background transcription worker & versioned DB persistence
│   │   │       ├── manager.py         # Singleton ASR lifecycle & memory management
│   │   │       └── specialist_adapters.py # Stubs for AI4Bharat IndicConformer & Hinglish-Apex
│   │   └── main.py                    # FastAPI entrypoint, CORS & routes
│   ├── tests/
│   │   ├── conftest.py                # In-memory test DB fixture with foreign keys
│   │   ├── test_database.py           # 12 schema & relationship tests
│   │   ├── test_health.py             # 4 health check & CORS tests
│   │   ├── test_media_ingestion.py    # 11 media validation, extraction & upload tests
│   │   ├── test_real_transcription_e2e.py # 2 live GPU acceleration & CPU fallback tests on RTX 5060
│   │   └── test_transcription.py      # 12 ASR unit, persistence, versioning & API tests
│   ├── venv/                          # Python 3.13 virtual environment
│   ├── alembic.ini                    # Alembic configuration
│   └── requirements.txt               # Backend dependencies (FastAPI, Faster-Whisper, CTranslate2, yt-dlp)
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts                 # Typed API client (upload, YouTube, poll jobs, delete, transcribe)
│   │   ├── types/
│   │   │   └── index.ts               # TypeScript interfaces (Transcripts, Segments, HW Diagnostics)
│   │   ├── App.css                    # Modern responsive styling for Ingestion & Transcript Hub
│   │   ├── App.tsx                    # Ingestion Hub, Job Tracker, Transcript Viewer & Diagnostics
│   │   ├── index.css
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
├── data/
│   └── storage/                       # Secure local media & model storage
├── docs/
│   └── architecture.md                # System architecture, ERD & pipeline diagrams
├── meeting_intelligence.db            # SQLite development database
└── README.md                          # Runbook & documentation
```

---

## 3. Supported Media Formats & ASR Pipeline

- **Audio Formats**: WAV, MP3, M4A, FLAC
- **Video Formats**: MP4, MKV, MOV, WebM
- **YouTube Streams**: Valid YouTube URLs (`youtube.com`, `youtu.be`) with explicit user authorization confirmation.
- **ASR Audio Normalization Standard**:
  - Container: WAV
  - Codec: `pcm_s16le`
  - Sample Rate: 16,000 Hz (16 kHz)
  - Channels: 1 (Mono)
- **Supported Languages**:
  - English (`en`), Hindi (`hi`), Gujarati (`gu`), Hinglish (`hi-en` / code-switching).
  - Preserves original spoken language without auto-translating.

---

## 4. API Endpoints (Phases 3 & 4)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/transcription/start/{meeting_id}` | Start asynchronous multilingual transcription on GPU |
| `GET` | `/api/v1/transcription/{meeting_id}` | Retrieve versioned transcripts and timestamped segments |
| `GET` | `/api/v1/transcription/jobs/{job_id}` | Poll transcription job progress (0-100%) and error state |
| `GET` | `/api/v1/transcription/hardware/diagnostics` | Inspect live GPU, VRAM, CTranslate2 CUDA support & active model |
| `POST` | `/api/v1/media/upload` | Upload audio/video file with background 16kHz mono normalization |
| `POST` | `/api/v1/media/youtube` | Ingest YouTube meeting stream via `yt-dlp` |
| `GET` | `/api/v1/media/meetings` | List meetings with media assets and processing jobs |
| `GET` | `/api/v1/media/meetings/{id}` | Get specific meeting details |
| `GET` | `/api/v1/media/jobs/{id}` | Poll media ingestion job status and progress percentage |
| `DELETE` | `/api/v1/media/assets/{id}` | Safely delete raw media from disk and record audit log |
| `GET` | `/health` | System health check and database connectivity |

---

## 5. How to Run Locally

### 1. Start the Backend API

```powershell
# From project root
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ASR Diagnostics**: [http://127.0.0.1:8000/api/v1/transcription/hardware/diagnostics](http://127.0.0.1:8000/api/v1/transcription/hardware/diagnostics)

### 2. Start the Frontend Application

```powershell
cd frontend
$env:PATH = "C:\Program Files\nodejs;" + $env:PATH
npm.cmd run dev
```

- **Web App**: [http://localhost:5173](http://localhost:5173)

---

## 6. Verification & Test Suite

Run all automated unit and hardware tests:

```powershell
$env:PYTHONPATH = "backend"
.\backend\venv\Scripts\pytest.exe backend/tests -v
```

**Results (41 passed in 7.94s, 0 warnings)**:
- **Database Architecture (12 passed)**: Multi-tenancy, foreign keys, cascades, multilingual segment timestamps, coexistence of original and cleaned transcript versions.
- **Health & CORS (4 passed)**: Health endpoint, CORS headers, API v1 router.
- **Media Ingestion (11 passed)**: Formats, YouTube validation, FFprobe inspection, 16kHz mono normalization, SHA-256 calculation, path traversal defense, safe raw deletion.
- **Live GPU Acceleration & CPU Fallback (2 passed)**: Verified real model execution on NVIDIA GeForce RTX 5060 Laptop GPU with float16 precision and CPU fallback.
- **ASR Architecture & Persistence (12 passed)**: Long audio chunk calculation, WAV slicing, overlap deduplication, version incrementation, null diarization provider, specialist adapter stubs, API start and retrieve.

---

## 7. Next Recommended Phase

**Phase 5: Intelligence & Extraction Pipeline**
- Ollama local LLM integration for meeting summaries, topics, decisions, action items with owners and deadlines, risks, and unresolved questions.
