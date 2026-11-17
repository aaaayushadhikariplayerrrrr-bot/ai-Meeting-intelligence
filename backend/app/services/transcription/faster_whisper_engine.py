"""Faster-Whisper ASR Engine Provider.

Implements the ASREngine interface using CTranslate2 and Faster-Whisper.
Provides GPU acceleration on NVIDIA RTX GPUs, automatic fallback to CPU,
long audio chunking, multilingual language detection, and honest execution reporting.
"""

import os
import time
import math
import tempfile
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple

from app.core.config import settings
from app.services.transcription.base import (
    ASREngine,
    ASRSegment,
    ASRResult,
    TranscriptionOptions,
)
from app.services.transcription.gpu_manager import (
    configure_windows_dll_directories,
    select_optimal_device_and_compute,
    release_gpu_memory,
    register_active_gpu_consumer,
    unregister_active_gpu_consumer,
)
from pathlib import Path
from app.services.transcription.chunking import (
    compute_audio_chunks,
    extract_wav_chunk,
    merge_and_deduplicate_segments,
)
from app.services.media_metadata import extract_media_metadata

logger = logging.getLogger(__name__)


class FasterWhisperEngine(ASREngine):
    """Faster-Whisper speech recognition engine powered by CTranslate2."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        configure_windows_dll_directories()
        self.model_name = model_name or settings.ASR_MODEL_NAME
        self.preferred_device = device or settings.ASR_DEVICE
        self.preferred_compute = compute_type or settings.ASR_COMPUTE_TYPE

        self.model = None
        self.actual_device: str = "unloaded"
        self.actual_compute_type: str = "unloaded"
        self.active_model_name: str = self.model_name

    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        """Load the Faster-Whisper model, trying CUDA first and falling back to CPU if needed."""
        target_model = model_name or self.model_name
        target_device = device or self.preferred_device

        # If already loaded with the same configuration, reuse
        if self.model is not None and self.active_model_name == target_model:
            logger.info(f"Reusing already loaded Faster-Whisper model: {target_model} on {self.actual_device}")
            return

        from faster_whisper import WhisperModel

        # Determine target device and compute type
        actual_device, actual_compute, reason = select_optimal_device_and_compute(
            preferred_device=target_device,
            preferred_compute=self.preferred_compute,
        )
        logger.info(f"Attempting to load model '{target_model}' on {actual_device} ({actual_compute}). Reason: {reason}")

        cpu_threads = getattr(settings, "ASR_CPU_THREADS", 4)
        num_workers = getattr(settings, "ASR_NUM_WORKERS", 1)

        try:
            self.model = WhisperModel(
                model_size_or_path=target_model,
                device=actual_device,
                compute_type=actual_compute,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=os.path.join(settings.STORAGE_DIR, "models"),
            )
            self.actual_device = actual_device
            self.actual_compute_type = actual_compute
            self.active_model_name = target_model
            if actual_device == "cuda":
                register_active_gpu_consumer("faster-whisper", target_model)
            logger.info(
                f"Successfully loaded Faster-Whisper model '{target_model}' on {actual_device} ({actual_compute}) "
                f"with cpu_threads={cpu_threads}, num_workers={num_workers}"
            )

        except Exception as e:
            if actual_device == "cuda":
                logger.warning(
                    f"Failed to load Faster-Whisper on CUDA: {e}. Falling back to CPU..."
                )
                unregister_active_gpu_consumer("faster-whisper")
                release_gpu_memory()
                self.model = WhisperModel(
                    model_size_or_path=target_model,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                    num_workers=num_workers,
                    download_root=os.path.join(settings.STORAGE_DIR, "models"),
                )
                self.actual_device = "cpu"
                self.actual_compute_type = "int8"
                self.active_model_name = target_model
                logger.info(
                    f"Model successfully loaded on CPU fallback (int8) with cpu_threads={cpu_threads}, num_workers={num_workers}."
                )
            else:
                raise

    def unload_model(self) -> None:
        """Unload model and free GPU / CPU memory."""
        unregister_active_gpu_consumer("faster-whisper")
        if self.model is not None:
            del self.model
            self.model = None
        self.actual_device = "unloaded"
        self.actual_compute_type = "unloaded"
        release_gpu_memory()
        logger.info("Faster-Whisper model unloaded and memory released.")

    def is_gpu_available(self) -> bool:
        """Check if CUDA acceleration is functional with CTranslate2."""
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata and active device status."""
        return {
            "engine_name": "faster-whisper",
            "model_name": self.active_model_name,
            "status": "loaded" if self.model is not None else "unloaded",
            "device": self.actual_device,
            "compute_type": self.actual_compute_type,
            "cpu_threads": getattr(settings, "ASR_CPU_THREADS", 4),
            "num_workers": getattr(settings, "ASR_NUM_WORKERS", 1),
            "is_gpu_available": self.is_gpu_available(),
            "supported_languages": ["en", "hi", "gu", "auto"],
            "supported_modes": ["standard", "hinglish"],
        }

    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ASRResult:
        """Transcribe normalized audio file."""
        if self.model is None:
            self.load_model()

        opts = options or TranscriptionOptions()

        # Probe duration for chunking & progress tracking
        try:
            metadata = extract_media_metadata(Path(audio_path))
            total_duration = metadata.duration_seconds or 0.0
        except Exception:
            total_duration = 0.0

        # Handle Hinglish language and mode handling:
        # Whisper natively only supports ISO 639-1 language codes (e.g. 'en', 'hi', 'gu').
        # Passing 'hi-en' directly causes Whisper runtime errors or unrecognized language errors.
        # We treat Hinglish as an application-level transcription mode.
        # For Hinglish / code-switching, setting whisper_language=None lets Whisper's multilingual
        # decoder transcribe code-switched Hindi-English speech without forcing translation,
        # guided by an initial prompt conditioning.
        transcription_mode = opts.transcription_mode or "standard"
        req_lang = opts.language

        if req_lang in ("hi-en", "hinglish"):
            transcription_mode = "hinglish"
            req_lang = None
        elif req_lang in ("auto", "None", ""):
            req_lang = None

        whisper_language = req_lang
        initial_prompt = opts.initial_prompt

        if transcription_mode == "hinglish":
            whisper_language = None  # Multilingual auto-detection for code-switching
            if not initial_prompt:
                initial_prompt = "Hindi and English code-switching conversation, Hinglish."

        # Check if long audio chunking is needed
        chunk_len = opts.chunk_length_seconds
        overlap = opts.chunk_overlap_seconds

        if chunk_len and total_duration > chunk_len:
            return self._transcribe_chunked(
                audio_path=audio_path,
                total_duration=total_duration,
                chunk_length=chunk_len,
                overlap=overlap,
                whisper_language=whisper_language,
                transcription_mode=transcription_mode,
                initial_prompt=initial_prompt,
                opts=opts,
                progress_callback=progress_callback,
            )

        return self._transcribe_single(
            audio_path=audio_path,
            duration=total_duration,
            whisper_language=whisper_language,
            transcription_mode=transcription_mode,
            initial_prompt=initial_prompt,
            opts=opts,
            progress_callback=progress_callback,
        )

    def _transcribe_single(
        self,
        audio_path: str,
        duration: float,
        whisper_language: Optional[str],
        transcription_mode: str,
        initial_prompt: Optional[str],
        opts: TranscriptionOptions,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ASRResult:
        """Transcribe single audio file without chunking."""
        if progress_callback:
            progress_callback(10.0)

        # Transcribe directly using faster-whisper generator
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=whisper_language,
            task="transcribe",  # Never translate
            beam_size=opts.beam_size,
            temperature=opts.temperature,
            vad_filter=opts.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=opts.vad_min_silence_duration_ms) if opts.vad_filter else None,
            initial_prompt=initial_prompt,
        )

        detected_lang = info.language
        lang_prob = round(info.language_probability, 4) if info.language_probability else None

        segments_list: List[ASRSegment] = []
        full_text_parts: List[str] = []
        seq = 0

        for seg in segments_gen:
            clean_text = seg.text.strip()
            if not clean_text:
                continue

            # Distinguish raw model log-prob from uncalibrated heuristic confidence score:
            raw_logprob = getattr(seg, "avg_logprob", None)
            conf_score = None
            if raw_logprob is not None:
                try:
                    conf_score = round(max(0.0, min(1.0, math.exp(raw_logprob))), 4)
                except OverflowError:
                    conf_score = 0.0

            asr_seg = ASRSegment(
                sequence_order=seq,
                start_time=round(seg.start, 2),
                end_time=round(seg.end, 2),
                text=clean_text,
                language_code=detected_lang or "en",
                confidence=conf_score,
                avg_logprob=round(raw_logprob, 4) if raw_logprob is not None else None,
            )
            segments_list.append(asr_seg)
            full_text_parts.append(clean_text)
            seq += 1

            if progress_callback and duration > 0:
                current_pct = min(95.0, 10.0 + (seg.end / duration) * 85.0)
                progress_callback(round(current_pct, 1))

        if progress_callback:
            progress_callback(100.0)

        return ASRResult(
            segments=segments_list,
            full_text=" ".join(full_text_parts),
            detected_language=detected_lang,
            language_probability=lang_prob,
            duration_seconds=round(duration or (segments_list[-1].end_time if segments_list else 0.0), 2),
            device_used=self.actual_device,
            compute_type_used=self.actual_compute_type,
            model_name=self.active_model_name,
            engine_name="faster-whisper",
            transcription_mode=transcription_mode,
        )

    def _transcribe_chunked(
        self,
        audio_path: str,
        total_duration: float,
        chunk_length: float,
        overlap: float,
        whisper_language: Optional[str],
        transcription_mode: str,
        initial_prompt: Optional[str],
        opts: TranscriptionOptions,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ASRResult:
        """Transcribe long audio file in overlapping chunks to conserve GPU memory and track progress."""
        chunks = compute_audio_chunks(total_duration, chunk_length=chunk_length, overlap=overlap)
        total_chunks = len(chunks)
        logger.info(f"Splitting {total_duration:.1f}s audio into {total_chunks} chunks ({chunk_length}s each, {overlap}s overlap)")

        chunk_results: List[Tuple[Any, List[ASRSegment]]] = []
        dominant_languages: List[str] = []
        overall_detected_lang = None
        overall_prob = None

        with tempfile.TemporaryDirectory(prefix="asr_chunks_") as temp_dir:
            for idx, chunk in enumerate(chunks):
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk.index:04d}.wav")
                extract_wav_chunk(
                    input_wav_path=audio_path,
                    output_wav_path=chunk_file,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                )

                # Transcribe this chunk
                segments_gen, info = self.model.transcribe(
                    chunk_file,
                    language=whisper_language,
                    task="transcribe",
                    beam_size=opts.beam_size,
                    temperature=opts.temperature,
                    vad_filter=opts.vad_filter,
                    vad_parameters=dict(min_silence_duration_ms=opts.vad_min_silence_duration_ms) if opts.vad_filter else None,
                    initial_prompt=initial_prompt,
                )

                chunk_lang = info.language
                dominant_languages.append(chunk_lang)
                if overall_detected_lang is None:
                    overall_detected_lang = chunk_lang
                    overall_prob = round(info.language_probability, 4) if info.language_probability else None

                chunk_segs: List[ASRSegment] = []
                c_seq = 0
                for seg in segments_gen:
                    clean_text = seg.text.strip()
                    if not clean_text:
                        continue

                    raw_logprob = getattr(seg, "avg_logprob", None)
                    conf_score = None
                    if raw_logprob is not None:
                        try:
                            conf_score = round(max(0.0, min(1.0, math.exp(raw_logprob))), 4)
                        except OverflowError:
                            conf_score = 0.0

                    chunk_segs.append(ASRSegment(
                        sequence_order=c_seq,
                        start_time=round(seg.start, 2),
                        end_time=round(seg.end, 2),
                        text=clean_text,
                        language_code=chunk_lang or "en",
                        confidence=conf_score,
                        avg_logprob=round(raw_logprob, 4) if raw_logprob is not None else None,
                    ))
                    c_seq += 1

                chunk_results.append((chunk, chunk_segs))

                if progress_callback:
                    pct = round(((idx + 1) / total_chunks) * 95.0, 1)
                    progress_callback(pct)

        # Merge segments across chunk boundaries with timestamp alignment and deduplication
        merged_segments = merge_and_deduplicate_segments(chunk_results, overlap=overlap)

        full_text = " ".join(s.text for s in merged_segments)

        if progress_callback:
            progress_callback(100.0)

        return ASRResult(
            segments=merged_segments,
            full_text=full_text,
            detected_language=overall_detected_lang,
            language_probability=overall_prob,
            duration_seconds=round(total_duration, 2),
            device_used=self.actual_device,
            compute_type_used=self.actual_compute_type,
            model_name=self.active_model_name,
            engine_name="faster-whisper",
            transcription_mode=transcription_mode,
        )
