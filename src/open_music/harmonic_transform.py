from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import _frames, _inverse
from .core import Audio
from .validation import ReplacementValidation, validate_replacement


@dataclass(frozen=True)
class HarmonicTransformation:
    audio: Audio
    frequency_ratio: float
    duration_ratio: float
    spectral_shift_ratio: float
    gain: float
    validation: ReplacementValidation


def _as_channels(audio: Audio) -> tuple[np.ndarray, bool]:
    values = np.asarray(audio, dtype=np.float64)
    was_mono = values.ndim == 1
    if was_mono:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("audio must contain one or more frames")
    return values, was_mono


def resample_duration(audio: Audio, target_frames: int) -> Audio:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    values, was_mono = _as_channels(audio)
    if values.shape[0] == target_frames:
        result = values.copy()
    else:
        source_positions = np.linspace(0.0, 1.0, values.shape[0])
        target_positions = np.linspace(0.0, 1.0, target_frames)
        result = np.stack(
            [
                np.interp(target_positions, source_positions, values[:, channel])
                for channel in range(values.shape[1])
            ],
            axis=1,
        )
    return result[:, 0] if was_mono else result


def pitch_shift_preserve_duration(
    audio: Audio,
    frequency_ratio: float,
    *,
    fft_size: int = 2048,
    hop_size: int | None = None,
) -> Audio:
    """Move spectral bins by an explicit ratio while preserving frame count."""
    if frequency_ratio <= 0.0:
        raise ValueError("frequency_ratio must be positive")
    values, was_mono = _as_channels(audio)
    hop = hop_size or fft_size // 4
    framed, padding = _frames(values, fft_size, hop)
    window = np.sqrt(np.hanning(fft_size))
    spectra = np.fft.rfft(framed * window[None, :, None], axis=1)
    bins = np.arange(spectra.shape[1], dtype=np.float64)
    source_bins = bins / frequency_ratio
    shifted = np.zeros_like(spectra)
    for frame in range(spectra.shape[0]):
        for channel in range(spectra.shape[2]):
            shifted[frame, :, channel] = np.interp(
                source_bins,
                bins,
                spectra[frame, :, channel].real,
                left=0.0,
                right=0.0,
            ) + 1j * np.interp(
                source_bins,
                bins,
                spectra[frame, :, channel].imag,
                left=0.0,
                right=0.0,
            )
    result = _inverse(shifted, window, hop, padding, values.shape[0])
    return result[:, 0] if was_mono else result


def _robust_gain(target: np.ndarray, candidate: np.ndarray) -> float:
    target_values = target if target.ndim == 1 else np.mean(target, axis=1)
    candidate_values = candidate if candidate.ndim == 1 else np.mean(candidate, axis=1)
    frame_size = min(256, target_values.size)
    ratios = []
    for start in range(0, target_values.size, frame_size):
        left = target_values[start : start + frame_size]
        right = candidate_values[start : start + frame_size]
        left_rms = float(np.sqrt(np.mean(left**2)))
        right_rms = float(np.sqrt(np.mean(right**2)))
        if left_rms > 1e-12 and right_rms > 1e-12:
            ratios.append(left_rms / right_rms)
    return float(np.median(ratios)) if ratios else 0.0


def transform_and_validate_harmonic_family(
    source: Audio,
    target: Audio,
    source_fundamental_hz: float,
    target_fundamental_hz: float,
    *,
    fft_size: int = 2048,
    validation_fft_sizes: tuple[int, ...] = (256, 1024, 4096),
    maximum_lag_frames: int = 256,
) -> HarmonicTransformation:
    if source_fundamental_hz <= 0.0 or target_fundamental_hz <= 0.0:
        raise ValueError("fundamentals must be positive")
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    duration_ratio = target_values.shape[0] / source_values.shape[0]
    frequency_ratio = target_fundamental_hz / source_fundamental_hz
    stretched = np.asarray(resample_duration(source_values, target_values.shape[0]))
    # Resampling duration already scales frequency by inverse duration ratio.
    spectral_ratio = frequency_ratio * duration_ratio
    shifted = np.asarray(
        pitch_shift_preserve_duration(stretched, spectral_ratio, fft_size=fft_size)
    )
    gain = _robust_gain(target_values, shifted)
    transformed = shifted * gain
    validation = validate_replacement(
        target_values,
        transformed,
        fft_sizes=validation_fft_sizes,
        maximum_lag_frames=maximum_lag_frames,
    )
    return HarmonicTransformation(
        audio=transformed,
        frequency_ratio=frequency_ratio,
        duration_ratio=duration_ratio,
        spectral_shift_ratio=spectral_ratio,
        gain=gain,
        validation=validation,
    )
