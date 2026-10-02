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


@dataclass(frozen=True)
class AdaptiveSegment:
    target_start: int
    target_end: int
    candidate_start: int
    candidate_end: int
    depth: int
    accepted: bool
    score: float
    lag_frames: int
    gain: float
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveValidation:
    segments: tuple[AdaptiveSegment, ...]
    accepted_target_ratio: float
    fully_accepted: bool


@dataclass(frozen=True)
class MaterializedAdaptiveReplacement:
    event_only: Audio
    residual: Audio
    weights: np.ndarray
    validation: AdaptiveValidation
    covered_frame_ratio: float


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
    target_norms = np.linalg.norm(target_spectra, axis=1)
    candidate_norms = np.linalg.norm(candidate_spectra, axis=1)
    norms = target_norms * candidate_norms
    similarities = np.divide(products, norms, out=np.zeros_like(products), where=norms > 0)
    # Matching silence is compatible. Only a one-sided silent frame is a
    # mismatch; treating two zero spectra as cosine zero creates false cuts in
    # the tails between real attacks.
    both_silent = (target_norms <= 1e-12) & (candidate_norms <= 1e-12)
    similarities[both_silent] = 1.0
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


def validate_with_adaptive_subdivision(
    target: Audio,
    candidate: Audio,
    *,
    minimum_segment_frames: int = 512,
    maximum_depth: int = 4,
    fft_sizes: tuple[int, ...] = (256, 1024, 4096),
    maximum_lag_frames: int = 256,
    thresholds: ValidationThresholds | None = None,
) -> AdaptiveValidation:
    """Validate compatible subregions without globally shrinking every event."""
    if minimum_segment_frames <= 0:
        raise ValueError("minimum_segment_frames must be positive")
    if maximum_depth < 0:
        raise ValueError("maximum_depth cannot be negative")
    target_values = np.asarray(target, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    segments: list[AdaptiveSegment] = []

    def visit(
        target_start: int,
        target_end: int,
        candidate_start: int,
        candidate_end: int,
        depth: int,
    ) -> None:
        target_section = target_values[target_start:target_end]
        candidate_section = candidate_values[candidate_start:candidate_end]
        result = validate_replacement(
            target_section,
            candidate_section,
            fft_sizes=fft_sizes,
            maximum_lag_frames=maximum_lag_frames,
            thresholds=thresholds,
        )
        target_length = target_end - target_start
        candidate_length = candidate_end - candidate_start
        can_split = (
            not result.accepted
            and depth < maximum_depth
            and target_length >= 2 * minimum_segment_frames
            and candidate_length >= 2 * minimum_segment_frames
        )
        valid_splits = [
            frame
            for frame in result.suggested_split_frames
            if minimum_segment_frames <= frame <= target_length - minimum_segment_frames
        ]
        if can_split and valid_splits:
            target_split = min(valid_splits, key=lambda frame: abs(frame - target_length / 2))
            candidate_split = round(target_split * candidate_length / target_length)
            if minimum_segment_frames <= candidate_split <= candidate_length - minimum_segment_frames:
                visit(
                    target_start,
                    target_start + target_split,
                    candidate_start,
                    candidate_start + candidate_split,
                    depth + 1,
                )
                visit(
                    target_start + target_split,
                    target_end,
                    candidate_start + candidate_split,
                    candidate_end,
                    depth + 1,
                )
                return
        segments.append(
            AdaptiveSegment(
                target_start=target_start,
                target_end=target_end,
                candidate_start=candidate_start,
                candidate_end=candidate_end,
                depth=depth,
                accepted=result.accepted,
                score=result.score,
                lag_frames=result.lag_frames,
                gain=result.gain,
                rejection_reasons=result.rejection_reasons,
            )
        )

    visit(0, target_values.shape[0], 0, candidate_values.shape[0], 0)
    accepted_frames = sum(
        segment.target_end - segment.target_start for segment in segments if segment.accepted
    )
    total_frames = target_values.shape[0]
    ratio = accepted_frames / total_frames if total_frames else 0.0
    return AdaptiveValidation(tuple(segments), ratio, bool(segments) and ratio == 1.0)


def materialize_adaptive_replacement(
    target: Audio,
    candidate: Audio,
    *,
    crossfade_frames: int = 128,
    minimum_segment_frames: int = 512,
    maximum_depth: int = 4,
    fft_sizes: tuple[int, ...] = (256, 1024, 4096),
    maximum_lag_frames: int = 256,
    thresholds: ValidationThresholds | None = None,
) -> MaterializedAdaptiveReplacement:
    """Render only validated subregions and leave every rejected sample residual.

    The returned invariant is ``event_only + residual == target``. Raised-cosine
    edges prevent accepted/rejected boundaries from introducing clicks; the
    crossfade remainder deliberately stays in the residual.
    """
    if crossfade_frames < 0:
        raise ValueError("crossfade_frames cannot be negative")
    target_values = np.asarray(target, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    target_was_mono = target_values.ndim == 1
    if target_was_mono:
        target_values = target_values[:, None]
    if candidate_values.ndim == 1:
        candidate_values = candidate_values[:, None]
    if target_values.shape[1] != candidate_values.shape[1]:
        if candidate_values.shape[1] == 1:
            candidate_values = np.repeat(candidate_values, target_values.shape[1], axis=1)
        elif target_values.shape[1] == 1:
            candidate_values = np.mean(candidate_values, axis=1, keepdims=True)
        else:
            raise ValueError("target and candidate channel counts are incompatible")

    validation = validate_with_adaptive_subdivision(
        target_values,
        candidate_values,
        minimum_segment_frames=minimum_segment_frames,
        maximum_depth=maximum_depth,
        fft_sizes=fft_sizes,
        maximum_lag_frames=maximum_lag_frames,
        thresholds=thresholds,
    )
    event_only = np.zeros_like(target_values)
    weights = np.zeros(target_values.shape[0], dtype=np.float64)
    for segment in validation.segments:
        if not segment.accepted:
            continue
        length = segment.target_end - segment.target_start
        source = candidate_values[segment.candidate_start : segment.candidate_end]
        aligned_channels = []
        for channel in range(source.shape[1]):
            padded = np.pad(source[:, channel], (0, max(0, length - source.shape[0])))[:length]
            aligned_channels.append(_shift(padded, segment.lag_frames))
        transformed = np.stack(aligned_channels, axis=1) * segment.gain
        envelope = np.ones(length, dtype=np.float64)
        fade = min(crossfade_frames, length // 2)
        if fade:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, endpoint=False))
            envelope[:fade] = ramp
            envelope[-fade:] = ramp[::-1]
        destination = slice(segment.target_start, segment.target_end)
        event_only[destination] = transformed * envelope[:, None]
        weights[destination] = envelope

    residual = target_values - event_only
    covered = float(np.count_nonzero(weights > 0.0) / weights.size) if weights.size else 0.0
    if target_was_mono:
        event_only = event_only[:, 0]
        residual = residual[:, 0]
    return MaterializedAdaptiveReplacement(
        event_only=event_only,
        residual=residual,
        weights=weights,
        validation=validation,
        covered_frame_ratio=covered,
    )
