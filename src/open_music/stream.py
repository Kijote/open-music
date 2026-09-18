from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .core import Audio


@dataclass(frozen=True)
class GrainEvent:
    module_id: str
    start_frame: int


@dataclass(frozen=True)
class GranularStream:
    modules: dict[str, Audio]
    events: tuple[GrainEvent, ...]
    total_frames: int
    channels: int
    window_frames: int
    hop_frames: int


def _grain_id(audio: Audio) -> str:
    digest = hashlib.sha256(np.asarray(audio, dtype="<f8").tobytes()).hexdigest()[:16]
    return f"grain-{digest}"


def analyze_stream(
    audio: Audio,
    *,
    window_frames: int = 4096,
    hop_frames: int = 2048,
) -> GranularStream:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if window_frames < 2 or hop_frames < 1 or hop_frames > window_frames:
        raise ValueError("invalid granular window or hop")

    modules: dict[str, Audio] = {}
    events: list[GrainEvent] = []
    first = -(window_frames // 2)
    for start in range(first, values.shape[0], hop_frames):
        grain = np.zeros((window_frames, values.shape[1]), dtype=np.float64)
        source_start = max(0, start)
        source_end = min(values.shape[0], start + window_frames)
        if source_end > source_start:
            target_start = source_start - start
            grain[target_start : target_start + source_end - source_start] = values[
                source_start:source_end
            ]
        module_id = _grain_id(grain)
        modules.setdefault(module_id, grain)
        events.append(GrainEvent(module_id, start))

    return GranularStream(
        modules,
        tuple(events),
        int(values.shape[0]),
        int(values.shape[1]),
        window_frames,
        hop_frames,
    )


def render_stream(stream: GranularStream) -> Audio:
    output = np.zeros((stream.total_frames, stream.channels), dtype=np.float64)
    weights = np.zeros(stream.total_frames, dtype=np.float64)
    window = np.hanning(stream.window_frames)

    for event in stream.events:
        grain = stream.modules[event.module_id]
        source_start = max(0, -event.start_frame)
        target_start = max(0, event.start_frame)
        available = min(
            stream.window_frames - source_start,
            stream.total_frames - target_start,
        )
        if available <= 0:
            continue
        section = slice(source_start, source_start + available)
        target = slice(target_start, target_start + available)
        output[target] += grain[section] * window[section, None]
        weights[target] += window[section]

    covered = weights > np.finfo(np.float64).eps
    output[covered] /= weights[covered, None]
    return output


def stream_metrics(stream: GranularStream) -> dict[str, float | int]:
    event_count = len(stream.events)
    module_count = len(stream.modules)
    return {
        "event_count": event_count,
        "unique_module_count": module_count,
        "reuse_ratio": event_count / module_count if module_count else 0.0,
        "stored_frames": module_count * stream.window_frames,
        "source_frames": stream.total_frames,
        "storage_ratio": (
            module_count * stream.window_frames / stream.total_frames
            if stream.total_frames
            else 0.0
        ),
        "coverage_ratio": 1.0 if stream.total_frames else 0.0,
    }
