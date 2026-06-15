from __future__ import annotations

import math
import struct

PCM16_SAMPLE_WIDTH = 2
INPUT_SAMPLE_RATE = 16_000
INPUT_CHANNELS = 1


class AudioFormatError(ValueError):
    """Raised when a client sends unsupported or malformed audio."""


def validate_pcm16_frame(
    frame: bytes,
    *,
    sample_rate: int = INPUT_SAMPLE_RATE,
    min_duration_ms: float = 1.0,
    max_duration_ms: float = 250.0,
) -> None:
    if not frame:
        raise AudioFormatError("audio frame is empty")
    if len(frame) % PCM16_SAMPLE_WIDTH != 0:
        raise AudioFormatError("pcm_s16le frame length must be even")

    duration_ms = pcm16_duration_ms(frame, sample_rate=sample_rate)
    if duration_ms < min_duration_ms:
        raise AudioFormatError(f"audio frame is shorter than {min_duration_ms:g}ms")
    if duration_ms > max_duration_ms:
        raise AudioFormatError(f"audio frame is longer than {max_duration_ms:g}ms")


def pcm16_duration_ms(frame: bytes, *, sample_rate: int) -> float:
    samples = len(frame) // PCM16_SAMPLE_WIDTH
    return samples / sample_rate * 1000.0


def pcm16_rms(frame: bytes) -> float:
    if not frame:
        return 0.0

    total = 0
    count = 0
    for (sample,) in struct.iter_unpack("<h", frame):
        total += sample * sample
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)


def pcm16_has_energy(frame: bytes, *, threshold: float = 350.0) -> bool:
    return pcm16_rms(frame) >= threshold


def pcm16_to_float32(frame: bytes):
    import numpy as np

    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def silence_pcm16(sample_rate: int, duration_ms: float) -> bytes:
    samples = int(sample_rate * duration_ms / 1000.0)
    return b"\x00\x00" * samples

