from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Audio


@dataclass(frozen=True)
class ValidationThresholds:
    minimum_duration_ratio: float = 0.9
    minimum_frame_similarity: float = 0.92
    maximum_log_energy_error: float = 0.18
    minimum_compatible_ratio: float = 0.9
    maximum_incompatible_run: int = 3
    maximum_p95_spectral_error: float = 0.12


@dataclass(frozen=True)
class ResolutionValidation:
    fft_size: int
    hop_size: int
    frame_count: int
    compatible_ratio: float
    p95_spectral_error: float
    maximum_spectral_error: float
    p95_log_energy_error: float
    maximum_incompatible_run: int
    incompatible_spans: tuple[tuple[int, int], ...]
    accepted: bool


@dataclass(frozen=True)
class ReplacementValidation:
    accepted: bool
    score: float
    lag_frames: int
    gain: float
    duration_ratio: float
    resolutions: tuple[ResolutionValidation, ...]
    rejection_reasons: tuple[str, ...]
    suggested_split_frames: tuple[int, ...]


def _mono(audio: Audio) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        return values
    return np.mean(values, axis=1)


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if lag >= 0:
        shifted[lag:] = values[: values.size - lag] if lag else values
    else:
        shifted[:lag] = values[-lag:]
    return shifted


def _align(target: np.ndarray, candidate: np.ndarray, maximum_lag: int) -> tuple[np.ndarray, int]:
    size = max(target.size, candidate.size)
    left = np.pad(target, (0, size - target.size))
    right = np.pad(candidate, (0, size - candidate.size))
    fft_size = 1 << (2 * size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(left, fft_size) * np.conj(np.fft.rfft(right, fft_size)),
        fft_size,
    )
    prefix_energy = np.concatenate(([0.0], np.cumsum(right**2)))
    left_norm = float(np.linalg.norm(left))
    best_lag = 0
    best_score = float("-inf")
    for lag in range(-maximum_lag, maximum_lag + 1):
        correlation_index = lag if lag >= 0 else fft_size + lag
        if lag >= 0:
            shifted_energy = prefix_energy[size - lag]
        else:
            shifted_energy = prefix_energy[size] - prefix_energy[-lag]
        denominator = left_norm * float(np.sqrt(max(shifted_energy, 0.0)))
        score = float(correlation[correlation_index] / denominator) if denominator else 0.0
        if score > best_score:
            best_score = score
            best_lag = lag
    return _shift(right, best_lag), best_lag


def _fit_gain(target: np.ndarray, candidate: np.ndarray) -> float:
    # A least-squares gain lets one incompatible region bias every compatible
    # frame. Use the median local RMS ratio so a local note/envelope mismatch
    # remains local instead of globally rescaling the candidate.
    frame_size = min(256, target.size)
    ratios: list[float] = []
    for start in range(0, target.size, frame_size):
        left = target[start : start + frame_size]
        right = candidate[start : start + frame_size]
        left_rms = float(np.sqrt(np.mean(left**2)))
        right_rms = float(np.sqrt(np.mean(right**2)))
        if left_rms > 1e-12 and right_rms > 1e-12:
            ratios.append(left_rms / right_rms)
    return float(np.median(ratios)) if ratios else 0.0


def _frames(values: np.ndarray, fft_size: int, hop_size: int) -> np.ndarray:
    if values.size < fft_size:
        values = np.pad(values, (0, fft_size - values.size))
    count = max(1, 1 + int(np.ceil((values.size - fft_size) / hop_size)))
    padded_size = fft_size + (count - 1) * hop_size
    padded = np.pad(values, (0, padded_size - values.size))
    return np.stack(
        [padded[index * hop_size : index * hop_size + fft_size] for index in range(count)]
    )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return tuple(runs)


def _resolution_validation(
    target: np.ndarray,
    candidate: np.ndarray,
    fft_size: int,
    thresholds: ValidationThresholds,
) -> ResolutionValidation:
    hop_size = max(1, fft_size // 4)
    window = np.hanning(fft_size)
    target_frames = _frames(target, fft_size, hop_size)
    candidate_frames = _frames(candidate, fft_size, hop_size)
    frame_count = max(target_frames.shape[0], candidate_frames.shape[0])
    if target_frames.shape[0] < frame_count:
        target_frames = np.pad(target_frames, ((0, frame_count - target_frames.shape[0]), (0, 0)))
    if candidate_frames.shape[0] < frame_count:
        candidate_frames = np.pad(
            candidate_frames, ((0, frame_count - candidate_frames.shape[0]), (0, 0))
        )

    target_spectra = np.log1p(np.abs(np.fft.rfft(target_frames * window, axis=1)))
    candidate_spectra = np.log1p(np.abs(np.fft.rfft(candidate_frames * window, axis=1)))
    products = np.sum(target_spectra * candidate_spectra, axis=1)
    norms = np.linalg.norm(target_spectra, axis=1) * np.linalg.norm(candidate_spectra, axis=1)
    similarities = np.divide(products, norms, out=np.zeros_like(products), where=norms > 0)
    spectral_error = 1.0 - np.clip(similarities, 0.0, 1.0)

    target_energy = np.sqrt(np.mean(target_frames**2, axis=1) + 1e-15)
    candidate_energy = np.sqrt(np.mean(candidate_frames**2, axis=1) + 1e-15)
    log_energy_error = np.abs(np.log((target_energy + 1e-12) / (candidate_energy + 1e-12)))
    incompatible = (similarities < thresholds.minimum_frame_similarity) | (
        log_energy_error > thresholds.maximum_log_energy_error
    )
    compatible_ratio = float(np.mean(~incompatible))
    p95_spectral_error = float(np.quantile(spectral_error, 0.95))
    maximum_run = _longest_true_run(incompatible)
    accepted = (
        compatible_ratio >= thresholds.minimum_compatible_ratio
        and p95_spectral_error <= thresholds.maximum_p95_spectral_error
        and maximum_run <= thresholds.maximum_incompatible_run
    )
    return ResolutionValidation(
        fft_size=fft_size,
        hop_size=hop_size,
        frame_count=frame_count,
        compatible_ratio=compatible_ratio,
        p95_spectral_error=p95_spectral_error,
        maximum_spectral_error=float(np.max(spectral_error)),
        p95_log_energy_error=float(np.quantile(log_energy_error, 0.95)),
        maximum_incompatible_run=maximum_run,
        incompatible_spans=_true_runs(incompatible),
        accepted=accepted,
    )


def validate_replacement(
    target: Audio,
    candidate: Audio,
    *,
    fft_sizes: tuple[int, ...] = (256, 1024, 4096),
    maximum_lag_frames: int = 256,
    thresholds: ValidationThresholds | None = None,
) -> ReplacementValidation:
    """Validate one replacement without temporal spectral averaging.

    The candidate may receive one explicit global gain and a bounded lag. Every
    FFT resolution must then satisfy local frame limits; good regions cannot
    compensate for an incompatible note, envelope, or transient elsewhere.
    """
    if not fft_sizes or any(size <= 1 for size in fft_sizes):
        raise ValueError("fft_sizes must contain values greater than one")
    if maximum_lag_frames < 0:
        raise ValueError("maximum_lag_frames cannot be negative")
    limits = thresholds or ValidationThresholds()
    target_mono = _mono(target)
    candidate_mono = _mono(candidate)
    if target_mono.size == 0 or candidate_mono.size == 0:
        raise ValueError("target and candidate cannot be empty")

    duration_ratio = min(target_mono.size, candidate_mono.size) / max(
        target_mono.size, candidate_mono.size
    )
    bounded_lag = min(maximum_lag_frames, max(target_mono.size, candidate_mono.size) - 1)
    aligned, lag = _align(target_mono, candidate_mono, bounded_lag)
    padded_target = np.pad(target_mono, (0, aligned.size - target_mono.size))
    gain = _fit_gain(padded_target, aligned)
    transformed = aligned * gain
    resolutions = tuple(
        _resolution_validation(padded_target, transformed, fft_size, limits)
        for fft_size in fft_sizes
    )

    reasons: list[str] = []
    if duration_ratio < limits.minimum_duration_ratio:
        reasons.append("duration")
    for resolution in resolutions:
        if not resolution.accepted:
            reasons.append(f"local_time_frequency:{resolution.fft_size}")
    accepted = not reasons
    local_score = min(resolution.compatible_ratio for resolution in resolutions)
    spectral_score = 1.0 - max(resolution.p95_spectral_error for resolution in resolutions)
    score = float(np.clip(min(duration_ratio, local_score, spectral_score), 0.0, 1.0))
    split_frames = sorted(
        {
            boundary
            for resolution in resolutions
            if not resolution.accepted
            for start, end in resolution.incompatible_spans
            for boundary in (
                start * resolution.hop_size,
                end * resolution.hop_size + resolution.fft_size,
            )
            if 0 < boundary < padded_target.size
        }
    )
    return ReplacementValidation(
        accepted=accepted,
        score=score,
        lag_frames=lag,
        gain=gain,
        duration_ratio=duration_ratio,
        resolutions=resolutions,
        rejection_reasons=tuple(reasons),
        suggested_split_frames=tuple(split_frames),
    )
