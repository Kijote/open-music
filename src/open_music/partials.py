from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import _frames, _inverse
from .core import Audio


@dataclass(frozen=True)
class PartialTrack:
    id: int
    start_frame: int
    end_frame: int
    active_frames: int
    mean_frequency_hz: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    spectral_energy: float


@dataclass(frozen=True)
class PartialDecomposition:
    components: tuple[Audio, ...]
    tracks: tuple[PartialTrack, ...]
    residual: Audio
    fft_size: int
    hop_size: int
    reconstruction_max_error: float


@dataclass
class _WorkingTrack:
    id: int
    observations: dict[int, tuple[int, float]]
    last_frame: int
    last_bin: int


def _frame_peaks(
    magnitude: np.ndarray,
    *,
    relative_threshold: float,
    maximum_peaks: int,
) -> list[tuple[int, float]]:
    if magnitude.size < 3:
        return []
    threshold = float(np.max(magnitude)) * relative_threshold
    indexes = np.flatnonzero(
        (magnitude[1:-1] > magnitude[:-2])
        & (magnitude[1:-1] >= magnitude[2:])
        & (magnitude[1:-1] >= threshold)
    ) + 1
    ordered = sorted(indexes, key=lambda index: (-magnitude[index], int(index)))
    return [(int(index), float(magnitude[index])) for index in ordered[:maximum_peaks]]


def decompose_spectral_partials(
    audio: Audio,
    sample_rate: int,
    *,
    fft_size: int = 2048,
    hop_size: int | None = None,
    relative_peak_threshold: float = 0.08,
    maximum_peaks_per_frame: int = 24,
    maximum_bin_drift: int = 4,
    maximum_gap_frames: int = 2,
    minimum_track_frames: int = 6,
    maximum_components: int = 32,
    mask_width_bins: float = 1.5,
) -> PartialDecomposition:
    """Track and reconstruct a content-dependent number of spectral partials."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if fft_size <= 1 or maximum_components <= 0:
        raise ValueError("fft_size and maximum_components must be positive")
    if not 0.0 < relative_peak_threshold <= 1.0:
        raise ValueError("relative_peak_threshold must be between zero and one")
    if maximum_peaks_per_frame <= 0 or minimum_track_frames <= 0:
        raise ValueError("peak and track limits must be positive")
    if maximum_bin_drift < 0 or maximum_gap_frames < 0 or mask_width_bins <= 0.0:
        raise ValueError("tracking tolerances must be non-negative")
    hop = hop_size or fft_size // 4
    values = np.asarray(audio, dtype=np.float64)
    was_mono = values.ndim == 1
    if was_mono:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("audio must contain one or more frames")

    framed, padding = _frames(values, fft_size, hop)
    window = np.sqrt(np.hanning(fft_size))
    spectra = np.fft.rfft(framed * window[None, :, None], axis=1)
    magnitude = np.mean(np.abs(spectra), axis=2)
    working: list[_WorkingTrack] = []
    next_id = 0
    for frame_index, frame_magnitude in enumerate(magnitude):
        peaks = _frame_peaks(
            frame_magnitude,
            relative_threshold=relative_peak_threshold,
            maximum_peaks=maximum_peaks_per_frame,
        )
        available = set(range(len(peaks)))
        active = [
            track
            for track in working
            if frame_index - track.last_frame <= maximum_gap_frames + 1
        ]
        for track in sorted(active, key=lambda item: item.id):
            compatible = [
                peak_index
                for peak_index in available
                if abs(peaks[peak_index][0] - track.last_bin) <= maximum_bin_drift
            ]
            if not compatible:
                continue
            selected = min(
                compatible,
                key=lambda peak_index: (
                    abs(peaks[peak_index][0] - track.last_bin),
                    -peaks[peak_index][1],
                    peaks[peak_index][0],
                ),
            )
            frequency_bin, amplitude = peaks[selected]
            track.observations[frame_index] = (frequency_bin, amplitude)
            track.last_frame = frame_index
            track.last_bin = frequency_bin
            available.remove(selected)
        for peak_index in sorted(available):
            frequency_bin, amplitude = peaks[peak_index]
            working.append(
                _WorkingTrack(
                    id=next_id,
                    observations={frame_index: (frequency_bin, amplitude)},
                    last_frame=frame_index,
                    last_bin=frequency_bin,
                )
            )
            next_id += 1

    eligible = [
        track for track in working if len(track.observations) >= minimum_track_frames
    ]
    eligible.sort(
        key=lambda track: (
            -sum(amplitude**2 for _, amplitude in track.observations.values()),
            track.id,
        )
    )
    selected_tracks = eligible[:maximum_components]
    raw_masks = np.zeros(
        (len(selected_tracks), spectra.shape[0], spectra.shape[1]), dtype=np.float64
    )
    frequency_bins = np.arange(spectra.shape[1])
    for component_index, track in enumerate(selected_tracks):
        observations = sorted(track.observations.items())
        for frame_index, (center_bin, _) in observations:
            raw_masks[component_index, frame_index] = np.exp(
                -0.5 * ((frequency_bins - center_bin) / mask_width_bins) ** 2
            )
    mask_total = np.sum(raw_masks, axis=0)
    normalized_masks = np.divide(
        raw_masks,
        np.maximum(mask_total[None, :, :], 1.0),
        out=np.zeros_like(raw_masks),
        where=True,
    )
    components = tuple(
        _inverse(
            spectra * mask[:, :, None], window, hop, padding, values.shape[0]
        )
        for mask in normalized_masks
    )
    combined = np.sum(components, axis=0) if components else np.zeros_like(values)
    residual = values - combined
    bin_hz = sample_rate / fft_size
    summaries = []
    for track in selected_tracks:
        bins = np.asarray([item[0] for item in track.observations.values()])
        amplitudes = np.asarray([item[1] for item in track.observations.values()])
        frames = sorted(track.observations)
        summaries.append(
            PartialTrack(
                id=track.id,
                start_frame=frames[0],
                end_frame=frames[-1] + 1,
                active_frames=len(frames),
                mean_frequency_hz=float(np.mean(bins) * bin_hz),
                minimum_frequency_hz=float(np.min(bins) * bin_hz),
                maximum_frequency_hz=float(np.max(bins) * bin_hz),
                spectral_energy=float(np.sum(amplitudes**2)),
            )
        )
    error = float(np.max(np.abs(values - combined - residual)))
    if was_mono:
        components = tuple(component[:, 0] for component in components)
        residual = residual[:, 0]
    return PartialDecomposition(
        components=components,
        tracks=tuple(summaries),
        residual=residual,
        fft_size=fft_size,
        hop_size=hop,
        reconstruction_max_error=error,
    )
