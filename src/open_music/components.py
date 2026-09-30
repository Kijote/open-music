from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Audio


@dataclass(frozen=True)
class SpectralComponents:
    harmonic: Audio
    percussive: Audio
    residual: Audio
    fft_size: int
    hop_size: int
    frame_count: int
    harmonic_mask_mean: float
    percussive_mask_mean: float
    residual_mask_mean: float
    reconstruction_max_error: float


def _frames(values: np.ndarray, fft_size: int, hop_size: int) -> tuple[np.ndarray, int]:
    padding = fft_size // 2
    padded = np.pad(values, ((padding, padding), (0, 0)))
    count = max(1, 1 + int(np.ceil((padded.shape[0] - fft_size) / hop_size)))
    required = fft_size + (count - 1) * hop_size
    padded = np.pad(padded, ((0, required - padded.shape[0]), (0, 0)))
    framed = np.stack(
        [padded[index * hop_size : index * hop_size + fft_size] for index in range(count)]
    )
    return framed, padding


def _inverse(
    spectra: np.ndarray,
    window: np.ndarray,
    hop_size: int,
    padding: int,
    output_frames: int,
) -> np.ndarray:
    frames = np.fft.irfft(spectra, n=window.size, axis=1) * window[None, :, None]
    size = window.size + (frames.shape[0] - 1) * hop_size
    output = np.zeros((size, frames.shape[2]), dtype=np.float64)
    normalization = np.zeros(size, dtype=np.float64)
    window_square = window**2
    for index, frame in enumerate(frames):
        start = index * hop_size
        output[start : start + window.size] += frame
        normalization[start : start + window.size] += window_square
    output = np.divide(
        output,
        normalization[:, None],
        out=np.zeros_like(output),
        where=normalization[:, None] > 1e-12,
    )
    return output[padding : padding + output_frames]


def _median_filter_axis(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    if width % 2 == 0:
        raise ValueError("median filter widths must be odd")
    padding = [(0, 0)] * values.ndim
    padding[axis] = (width // 2, width // 2)
    padded = np.pad(values, padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=axis)
    return np.median(windows, axis=-1)


def separate_spectral_components(
    audio: Audio,
    *,
    fft_size: int = 2048,
    hop_size: int | None = None,
    harmonic_filter_frames: int = 17,
    percussive_filter_bins: int = 17,
    mask_power: float = 2.0,
    residual_ambiguity: float = 1.0,
) -> SpectralComponents:
    """Separate coherent tonal, transient, and ambiguous energy with soft masks.

    The same masks are applied to every channel so stereo relationships are not
    independently classified. The residual owns spectrally ambiguous energy;
    no bin is discarded and the returned layers sum to the input.
    """
    if fft_size <= 1:
        raise ValueError("fft_size must be greater than one")
    hop = hop_size or fft_size // 4
    if not 0 < hop <= fft_size:
        raise ValueError("hop_size must be between one and fft_size")
    if mask_power <= 0.0:
        raise ValueError("mask_power must be positive")
    if residual_ambiguity < 0.0:
        raise ValueError("residual_ambiguity cannot be negative")
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
    harmonic_evidence = _median_filter_axis(magnitude, harmonic_filter_frames, axis=0)
    percussive_evidence = _median_filter_axis(magnitude, percussive_filter_bins, axis=1)
    harmonic_score = harmonic_evidence**mask_power
    percussive_score = percussive_evidence**mask_power
    ambiguity_score = residual_ambiguity * np.sqrt(harmonic_score * percussive_score)
    total = harmonic_score + percussive_score + ambiguity_score + 1e-15
    harmonic_mask = harmonic_score / total
    percussive_mask = percussive_score / total
    residual_mask = ambiguity_score / total

    harmonic = _inverse(
        spectra * harmonic_mask[:, :, None], window, hop, padding, values.shape[0]
    )
    percussive = _inverse(
        spectra * percussive_mask[:, :, None], window, hop, padding, values.shape[0]
    )
    # Compute this complement in the time domain to make conservation an API
    # invariant even after finite-precision overlap-add.
    residual = values - harmonic - percussive
    error = float(np.max(np.abs(values - harmonic - percussive - residual)))
    if was_mono:
        harmonic = harmonic[:, 0]
        percussive = percussive[:, 0]
        residual = residual[:, 0]
    return SpectralComponents(
        harmonic=harmonic,
        percussive=percussive,
        residual=residual,
        fft_size=fft_size,
        hop_size=hop,
        frame_count=spectra.shape[0],
        harmonic_mask_mean=float(np.mean(harmonic_mask)),
        percussive_mask_mean=float(np.mean(percussive_mask)),
        residual_mask_mean=float(np.mean(residual_mask)),
        reconstruction_max_error=error,
    )
