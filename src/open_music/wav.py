from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .core import Audio


def read_wav(path: Path) -> tuple[int, Audio]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
        data = source.readframes(frames)

    if channels not in (1, 2):
        raise ValueError("only mono and stereo WAV files are supported")
    if sample_width == 1:
        values = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        integers = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        values = integers.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")

    return sample_rate, values.reshape(-1, channels)


def resample_linear(audio: Audio, source_rate: int, target_rate: int) -> Audio:
    if source_rate == target_rate:
        return audio
    output_frames = round(audio.shape[0] * target_rate / source_rate)
    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, audio.shape[0] - 1, output_frames)
    return np.column_stack(
        [np.interp(target_positions, source_positions, audio[:, channel]) for channel in range(audio.shape[1])]
    )


def write_wav(path: Path, sample_rate: int, audio: Audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(pcm.shape[1])
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())
