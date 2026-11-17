"""Multilingual Embedding Service using intfloat/multilingual-e5-small.

Provides:
- Lazy model loading to prevent startup overhead and GPU VRAM consumption.
- CPU inference by default, preserving GPU VRAM for ASR and local LLMs.
- Strict E5 task prefix enforcement ('passage: ' for indexable content, 'query: ' for search queries).
- Unit L2 vector normalization for fast, exact cosine similarity via dot products.
- Batch processing without holding full transcripts in tensor memory.
- Safe offline cache inspection to prevent unauthorized background downloads.
"""

import os
import logging
import threading
from pathlib import Path
from typing import List, Optional, Any, Union

from app.core.config import settings
from app.services.transcription.gpu_manager import (
    configure_cpu_resources,
    can_use_gpu_for_embedding,
)

import numpy as np
import torch
import torch.nn.functional as F
from app.services.memory.exceptions import (
    EmbeddingError,
    ModelNotCachedError,
    EmbeddingModelLoadError,
)
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult

logger = logging.getLogger(__name__)

E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "


def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling across non-padding token representations."""
    # Ensure attention_mask has shape [batch_size, seq_len, 1]
    mask_expanded = attention_mask.unsqueeze(-1).bool()
    last_hidden_masked = last_hidden_states.masked_fill(~mask_expanded, 0.0)
    sum_embeddings = last_hidden_masked.sum(dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
    return sum_embeddings / sum_mask


class MultilingualE5EmbeddingService:
    """Service generating normalized 384-dimensional multilingual embeddings using E5."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimension: Optional[int] = None,
        lazy_load: bool = True,
        allow_download: bool = False,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        # Bound CPU threads and OpenMP before embedding model initialization
        configure_cpu_resources()

        self.model_name = (model_name or getattr(settings, "MEMORY_EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")).strip()
        self.device = (device or getattr(settings, "MEMORY_EMBEDDING_DEVICE", "cpu")).strip().lower()
        self.batch_size = batch_size or getattr(settings, "MEMORY_EMBEDDING_BATCH_SIZE", 16)
        self.dimension = dimension or getattr(settings, "MEMORY_EMBEDDING_DIMENSION", 384)
        self.allow_download = allow_download

        self._tokenizer = tokenizer
        self._model = model
        self._is_loaded = tokenizer is not None and model is not None

        if not lazy_load and not self._is_loaded:
            self.load_model()

    @property
    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are instantiated in memory."""
        return self._is_loaded

    def get_hf_cache_dir(self) -> Path:
        """Resolve the default HuggingFace hub cache directory on the current host."""
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"
        user_home = Path.home()
        return user_home / ".cache" / "huggingface" / "hub"

    def is_cached(self) -> bool:
        """Check if model snapshots or weights exist in the local HuggingFace cache."""
        if self._model is not None and self._tokenizer is not None:
            return True

        cache_hub = self.get_hf_cache_dir()
        if not cache_hub.exists():
            return False

        # HuggingFace hub stores models under 'models--<org>--<repo>'
        normalized_name = self.model_name.replace("/", "--")
        expected_dir = cache_hub / f"models--{normalized_name}"
        if not expected_dir.exists():
            return False

        snapshots_dir = expected_dir / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True

        return False

    def load_model(self, tokenizer: Optional[Any] = None, model: Optional[Any] = None) -> None:
        """Explicitly load tokenizer and model weights into memory.

        Supports direct dependency injection for tests or preloaded instances.
        If allow_download is False and model is not cached, raises ModelNotCachedError.
        """
        if tokenizer is not None and model is not None:
            self._tokenizer = tokenizer
            self._model = model
            self._is_loaded = True
            logger.info(f"Loaded injected tokenizer and model for '{self.model_name}' on device '{self.device}'.")
            return

        if self._is_loaded:
            return

        # Safe cache check to prevent unauthorized ~470 MB downloads
        if not self.allow_download and not self.is_cached():
            raise ModelNotCachedError(
                f"Embedding model '{self.model_name}' (~470 MB) is not present in the local HuggingFace cache "
                f"({self.get_hf_cache_dir()}). Automatic download is disabled to prevent unexpected network latency "
                "or bandwidth usage. Pre-cache model weights or explicitly set allow_download=True."
            )

        # Coordinate GPU VRAM allocation with hardware resource manager (Phase 9.3)
        if self.device == "cuda":
            can_use, reason = can_use_gpu_for_embedding()
            if not can_use:
                logger.warning(
                    f"CUDA requested for embedding '{self.model_name}', but hardware manager denied GPU: {reason}. "
                    "Falling back safely to CPU."
                )
                self.device = "cpu"

        try:
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"Loading '{self.model_name}' tokenizer and model on device '{self.device}'...")
            local_only = not self.allow_download
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
            logger.info(f"Successfully loaded '{self.model_name}' on '{self.device}'.")
        except Exception as e:
            self._is_loaded = False
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model '{self.model_name}' on device '{self.device}': {type(e).__name__}: {str(e)}"
            ) from e

    def _ensure_loaded(self) -> None:
        """Ensure model and tokenizer are initialized before inference."""
        if not self._is_loaded:
            self.load_model()

    def _encode_text_batch(self, prefixed_texts: List[str]) -> List[List[float]]:
        """Internal worker encoding a pre-formatted batch of texts through model pooling and L2 normalization."""
        if not prefixed_texts:
            return []

        self._ensure_loaded()
        all_embeddings: List[List[float]] = []

        # Process in batches to control memory consumption
        for i in range(0, len(prefixed_texts), self.batch_size):
            batch = prefixed_texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # Move input tensors to target device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                # Resolve last hidden state from model outputs
                if hasattr(outputs, "last_hidden_state"):
                    last_hidden = outputs.last_hidden_state
                elif isinstance(outputs, (list, tuple)):
                    last_hidden = outputs[0]
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    last_hidden = outputs["last_hidden_state"]
                else:
                    last_hidden = getattr(outputs, "last_hidden_state", outputs[0])

                # Mean pooling and unit L2 normalization
                pooled = _average_pool(last_hidden, attention_mask)
                normalized = F.normalize(pooled, p=2, dim=1)

                # Validate dimension
                batch_dim = normalized.shape[1]
                if batch_dim != self.dimension:
                    raise EmbeddingError(
                        f"Expected embedding dimension {self.dimension}, but model produced {batch_dim}."
                    )

                batch_vectors = normalized.cpu().tolist()
                all_embeddings.extend(batch_vectors)

        return all_embeddings

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for indexable transcript passages with 'passage: ' prefix.

        Empty input returns an empty list without triggering model execution.
        """
        if not texts:
            return []

        prefixed_texts = [f"{E5_PASSAGE_PREFIX}{t.strip()}" for t in texts]
        return self._encode_text_batch(prefixed_texts)

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized embedding for a single user search query with 'query: ' prefix."""
        cleaned_query = (query or "").strip()
        prefixed_query = f"{E5_QUERY_PREFIX}{cleaned_query}"
        results = self._encode_text_batch([prefixed_query])
        if not results:
            raise EmbeddingError("Failed to produce embedding for query.")
        return results[0]

    def embed_chunks(self, chunks: List[SemanticChunk]) -> List[ChunkEmbeddingResult]:
        """Embed a list of SemanticChunk objects, returning ChunkEmbeddingResult items."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        embeddings = self.embed_passages(texts)

        results: List[ChunkEmbeddingResult] = []
        for chunk, emb in zip(chunks, embeddings):
            results.append(
                ChunkEmbeddingResult(
                    chunk=chunk,
                    embedding=emb,
                    dimension=self.dimension,
                    model_name=self.model_name,
                )
            )
        return results


# Module-level cached instance and thread safety lock
_CACHED_EMBEDDING_SERVICE: Optional[MultilingualE5EmbeddingService] = None
_embedding_service_lock = threading.Lock()


def get_embedding_service(
    device: Optional[str] = None,
    lazy_load: bool = True,
    allow_download: bool = False,
) -> MultilingualE5EmbeddingService:
    """Retrieve or initialize the shared MultilingualE5EmbeddingService instance."""
    global _CACHED_EMBEDDING_SERVICE
    with _embedding_service_lock:
        if _CACHED_EMBEDDING_SERVICE is None:
            _CACHED_EMBEDDING_SERVICE = MultilingualE5EmbeddingService(
                device=device,
                lazy_load=lazy_load,
                allow_download=allow_download,
            )
        return _CACHED_EMBEDDING_SERVICE
