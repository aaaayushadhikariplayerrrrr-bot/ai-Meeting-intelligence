"""Audio chunking and timestamp alignment utilities for long meeting recordings.

Handles:
- Time window chunk calculation with configurable overlap
- Fast slicing of normalized 16 kHz mono WAV files
- Offset adjustment for timestamps
- Deduplication of segments across chunk overlap boundaries
"""

import wave
import math
import os
import tempfile
from typing import List, Tuple, Optional
from app.services.transcription.base import ASRSegment


class AudioChunk:
    """Represents a time slice of an audio file."""
    def __init__(self, index: int, start_time: float, end_time: float, is_last: bool):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.is_last = is_last

    def __repr__(self) -> str:
        return f"<AudioChunk {self.index}: {self.start_time:.2f}s - {self.end_time:.2f}s>"


def compute_audio_chunks(
    total_duration: float,
    chunk_length: float = 300.0,
    overlap: float = 2.0
) -> List[AudioChunk]:
    """Compute overlapping time windows covering total audio duration.
    
    If total_duration <= chunk_length, returns a single chunk covering the entire duration.
    """
    if total_duration <= 0:
        return []

    if total_duration <= chunk_length:
        return [AudioChunk(index=0, start_time=0.0, end_time=total_duration, is_last=True)]

    chunks = []
    step = chunk_length - overlap
    current_start = 0.0
    index = 0

    while current_start < total_duration:
        current_end = min(current_start + chunk_length, total_duration)
        is_last = (current_end >= total_duration)
        chunks.append(AudioChunk(
            index=index,
            start_time=current_start,
            end_time=current_end,
            is_last=is_last
        ))
        if is_last:
            break
        current_start += step
        index += 1

    return chunks


def extract_wav_chunk(
    input_wav_path: str,
    output_wav_path: str,
    start_time: float,
    end_time: float
) -> None:
    """Slice a chunk from a normalized 16 kHz 16-bit mono WAV file using Python's standard wave library.
    
    This performs zero-reencode fast extraction without external process overhead.
    """
    with wave.open(input_wav_path, "rb") as src:
        n_channels = src.getnchannels()
        sample_width = src.getsampwidth()
        framerate = src.getframerate()
        total_frames = src.getnframes()

        start_frame = max(0, int(start_time * framerate))
        end_frame = min(total_frames, int(end_time * framerate))
        frames_to_read = max(0, end_frame - start_frame)

        src.setpos(start_frame)
        audio_data = src.readframes(frames_to_read)

    with wave.open(output_wav_path, "wb") as dst:
        dst.setnchannels(n_channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(framerate)
        dst.writeframes(audio_data)


def merge_and_deduplicate_segments(
    chunk_results: List[Tuple[AudioChunk, List[ASRSegment]]],
    overlap: float = 2.0
) -> List[ASRSegment]:
    """Merge segment lists from consecutive chunks with timestamp offsets and overlap deduplication.
    
    Prevents duplicate segments occurring within chunk overlaps and re-indexes sequence_order.
    """
    merged_segments: List[ASRSegment] = []
    global_seq = 0

    for chunk, segments in chunk_results:
        for seg in segments:
            # Adjust local chunk timestamps to absolute meeting timestamps
            abs_start = round(seg.start_time + chunk.start_time, 2)
            abs_end = round(seg.end_time + chunk.start_time, 2)

            # Check overlap duplicate with last accepted segment
            if merged_segments:
                last_seg = merged_segments[-1]
                # If this segment starts before the last segment ended, or overlaps significantly in time
                # and text is identical or substring, skip duplicate
                is_time_overlap = abs_start < last_seg.end_time
                clean_text_curr = seg.text.strip().lower()
                clean_text_last = last_seg.text.strip().lower()

                if is_time_overlap and (
                    clean_text_curr == clean_text_last
                    or clean_text_curr in clean_text_last
                    or (abs_end <= last_seg.end_time)
                ):
                    continue

                # If the previous chunk had a truncated partial utterance cut off at boundary
                # and this chunk captures the complete phrase, replace the fragment
                if is_time_overlap and (clean_text_last in clean_text_curr and clean_text_last != clean_text_curr):
                    merged_segments[-1] = ASRSegment(
                        sequence_order=last_seg.sequence_order,
                        start_time=min(last_seg.start_time, abs_start),
                        end_time=abs_end,
                        text=seg.text.strip(),
                        language_code=seg.language_code,
                        confidence=seg.confidence or last_seg.confidence,
                        avg_logprob=getattr(seg, "avg_logprob", None) or getattr(last_seg, "avg_logprob", None),
                        speaker_label=seg.speaker_label or last_seg.speaker_label,
                        words=seg.words or last_seg.words
                    )
                    continue

                # If overlap without duplicate text, adjust start boundary to prevent negative gap
                if abs_start < last_seg.end_time:
                    abs_start = last_seg.end_time

                if abs_end <= abs_start:
                    continue

            adjusted_seg = ASRSegment(
                sequence_order=global_seq,
                start_time=abs_start,
                end_time=abs_end,
                text=seg.text.strip(),
                language_code=seg.language_code,
                confidence=seg.confidence,
                avg_logprob=getattr(seg, "avg_logprob", None),
                speaker_label=seg.speaker_label,
                words=seg.words
            )
            merged_segments.append(adjusted_seg)
            global_seq += 1

    return merged_segments
