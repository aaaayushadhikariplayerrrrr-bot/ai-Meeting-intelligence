from pathlib import Path
from typing import List, Set, Union
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory is the project root (where repository root files reside)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multilingual Meeting Intelligence System"
    VERSION: str = "0.4.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./meeting_intelligence.db"

    # Storage Configuration
    # Safe storage directory outside the publicly accessible frontend
    STORAGE_DIR: str = str(ROOT_DIR / "data" / "storage")

    # Upload & Media Limits
    MAX_UPLOAD_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB limit
    MAX_MEDIA_DURATION_SECONDS: float = 14400.0      # 4 hours limit

    # Allowed Media Extensions
    ALLOWED_AUDIO_EXTENSIONS: Set[str] = {"wav", "mp3", "m4a", "flac"}
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {"mp4", "mkv", "mov", "webm"}

    # ASR & Multilingual Transcription Configuration
    ASR_ENGINE: str = "faster-whisper"
    ASR_MODEL_NAME: str = "large-v3-turbo"
    ASR_DEVICE: str = "cuda"  # Prefer "cuda" with automatic fallback to "cpu"
    ASR_COMPUTE_TYPE: str = "float16"  # "float16", "int8_float16", "bfloat16", "int8"
    ASR_FALLBACK_DEVICE: str = "cpu"
    ASR_FALLBACK_COMPUTE_TYPE: str = "int8"
    ASR_BEAM_SIZE: int = 5
    ASR_TEMPERATURE: float = 0.0
    ASR_VAD_FILTER: bool = True
    ASR_CHUNK_LENGTH_SECONDS: float = 300.0  # 5-minute chunks for long recordings
    ASR_CHUNK_OVERLAP_SECONDS: float = 2.0   # 2-second overlap between chunks
    AUTO_DELETE_RAW_POST_TRANSCRIPTION: bool = True  # Automatically delete raw uploaded media post-transcription
    AUTO_DELETE_NORMALIZED_POST_TRANSCRIPTION: bool = True  # Automatically delete associated intermediate normalized audio post-transcription
    ASR_CPU_THREADS: int = 4  # Bounded CPU threads for CTranslate2 / Whisper on CPU / fallback
    ASR_NUM_WORKERS: int = 1  # Single worker instance to prevent duplicate model memory in RAM/VRAM

    # Hardware & CPU Resource Allocation (Phase 9.3)
    # Bounded CPU threads prevent oversubscription between FastAPI, background worker, PyTorch, CTranslate2, and BLAS.
    CPU_MAX_THREADS: int = 4  # Bounded threads for OpenMP, OpenBLAS, MKL
    TORCH_NUM_THREADS: int = 4  # Intra-op parallelism threads for PyTorch CPU operations
    TORCH_NUM_INTEROP_THREADS: int = 2  # Inter-op parallelism threads for PyTorch (must be configured prior to parallel execution)

    # Background Processing Job Worker Configuration (Phase 6D.5)
    JOB_POLL_INTERVAL_SECONDS: float = 2.0

    # Retention Policy (Phase 6D.6)
    RETENTION_DAYS: int = 30
    RETENTION_CLEANUP_INTERVAL_SECONDS: float = 86400.0  # 24 hours (cleanup interval)
    ENABLE_RETENTION_SCHEDULER: bool = True

    # Structured Meeting Intelligence Configuration (Phase 7)
    INTELLIGENCE_PROVIDER: str = "deterministic"  # "disabled", "ollama", "deterministic", "gemini"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL_NAME: str = "llama3.2:3b"
    LLM_REQUEST_TIMEOUT_SECONDS: float = 60.0
    INTELLIGENCE_MAX_TRANSCRIPT_CHARS: int = 40000
    INTELLIGENCE_MAX_CHUNKS: int = 50
    INTELLIGENCE_OUTPUT_LANGUAGE: str = "en"
    INTELLIGENCE_MAX_RETRIES: int = 2

    # Gemini Cloud Intelligence & RAG Configuration
    GEMINI_ENABLED: bool = False
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_REQUEST_TIMEOUT_SECONDS: float = 120.0
    GEMINI_MAX_RETRIES: int = 2

    # System Diagnostics UI & Endpoint Gate (Developer Only, Defaults to False)
    ENABLE_SYSTEM_DIAGNOSTICS: bool = False

    # Memory & Organizational RAG Configuration (Phase 8)
    MEMORY_EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    MEMORY_EMBEDDING_DEVICE: str = "cpu"  # Prefer CPU inference initially to preserve VRAM
    MEMORY_EMBEDDING_BATCH_SIZE: int = 16
    MEMORY_EMBEDDING_DIMENSION: int = 384
    MEMORY_CHUNK_TARGET_TOKENS: int = 250
    MEMORY_CHUNK_MAX_TOKENS: int = 350
    MEMORY_CHUNK_OVERLAP_TOKENS: int = 30

    # Security & Authentication Configuration (Phase 6)
    SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Rate Limiting Configuration (Phase 6C.3 & Phase 8.5)
    RATE_LIMIT_LOGIN_MAX_ATTEMPTS: int = 5
    RATE_LIMIT_LOGIN_WINDOW_SECONDS: int = 60
    RATE_LIMIT_REGISTER_MAX_ATTEMPTS: int = 3
    RATE_LIMIT_REGISTER_WINDOW_SECONDS: int = 60
    RATE_LIMIT_MEMORY_MAX_REQUESTS: int = 60
    RATE_LIMIT_MEMORY_WINDOW_SECONDS: int = 60
    RATE_LIMIT_MAX_KEYS: int = 10000

    # CORS Configuration
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("DATABASE_URL", mode="after")
    def normalize_sqlite_path(cls, v: str) -> str:
        """Resolve relative SQLite paths to absolute paths based on project root."""
        if v.startswith("sqlite:///./"):
            relative_file = v[len("sqlite:///./"):]
            abs_db_path = (ROOT_DIR / relative_file).resolve().as_posix()
            return f"sqlite:///{abs_db_path}"
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("INTELLIGENCE_PROVIDER", mode="after")
    def validate_intelligence_provider(cls, v: str) -> str:
        norm = (v or "").strip().lower()
        allowed = {"disabled", "ollama", "deterministic", "gemini"}
        if norm not in allowed:
            raise ValueError(
                f"Invalid INTELLIGENCE_PROVIDER '{v}'. Must be one of: {sorted(list(allowed))}"
            )
        return norm

    @field_validator("OLLAMA_BASE_URL", mode="after")
    def normalize_ollama_url(cls, v: str) -> str:
        url = (v or "").strip().rstrip("/")
        if not url:
            return "http://127.0.0.1:11434"
        return url

    @field_validator("LLM_REQUEST_TIMEOUT_SECONDS", mode="after")
    def validate_llm_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be greater than 0.")
        return v

    @field_validator("INTELLIGENCE_MAX_TRANSCRIPT_CHARS", mode="after")
    def validate_max_chars(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("INTELLIGENCE_MAX_TRANSCRIPT_CHARS must be greater than 0.")
        return v

    @field_validator("INTELLIGENCE_MAX_CHUNKS", mode="after")
    def validate_max_chunks(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("INTELLIGENCE_MAX_CHUNKS must be greater than 0.")
        return v

    @field_validator("INTELLIGENCE_MAX_RETRIES", mode="after")
    def validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("INTELLIGENCE_MAX_RETRIES cannot be negative.")
        return v

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        """Validate that cryptographic keys have sufficient entropy and satisfy environment policies."""
        env = (self.ENVIRONMENT or "").strip().lower()
        secret = (self.SECRET_KEY or "").strip()

        # In automated testing environment, permit an explicitly provided or safe fallback test secret
        if env == "testing":
            if not secret:
                self.SECRET_KEY = "test-secret-key-for-automated-testing-only-minimum-32-chars"
            elif len(secret) < 32:
                raise ValueError("Test SECRET_KEY must be at least 32 characters long.")
            return self

        # In development, staging, production, or any non-testing environment:
        if not secret:
            raise ValueError(
                "SECRET_KEY is required in development and production environments. "
                "Please configure a cryptographically secure 32+ character secret in your .env file "
                "or environment variables (e.g. generated via 'openssl rand -hex 32' or "
                "'python -c \"import secrets; print(secrets.token_hex(32))\"')."
            )

        known_insecure = {
            "changeme",
            "secret",
            "secretkey",
            "default",
            "admin",
            "password",
            "generate-with-openssl-rand-hex-32-min-32-chars",
            "your-secret-key-here-must-be-32-chars-or-more",
            "test-secret-key-for-automated-testing-only-minimum-32-chars",
        }
        if secret.lower() in known_insecure or any(placeholder in secret.lower() for placeholder in ["generate-with-openssl", "your-secret-key"]):
            raise ValueError(
                "Configured SECRET_KEY is a known insecure placeholder. "
                "Please generate a unique, cryptographically secure 32+ character secret."
            )

        if len(secret) < 32:
            raise ValueError(
                f"SECRET_KEY must be at least 32 characters (256 bits) long for {self.JWT_ALGORITHM} (got {len(secret)} characters)."
            )

        if env == "production":
            insecure_prod_terms = ["placeholder", "example", "sample", "demo", "test", "insecure", "default"]
            if any(term in secret.lower() for term in insecure_prod_terms):
                raise ValueError(
                    "Configured SECRET_KEY contains an insecure placeholder or test pattern not permitted in production."
                )

        if self.JWT_ALGORITHM not in {"HS256", "HS384", "HS512"}:
            raise ValueError(
                f"Unsupported JWT_ALGORITHM '{self.JWT_ALGORITHM}'. Only HMAC algorithms (HS256, HS384, HS512) are permitted."
            )

        if self.ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
            raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be greater than 0.")

        return self

    model_config = SettingsConfigDict(
        env_file=[
            str(ROOT_DIR / "backend" / ".env"),
            str(ROOT_DIR / ".env"),
            ".env"
        ],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
