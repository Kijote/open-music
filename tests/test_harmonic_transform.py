import numpy as np

from open_music.harmonic_transform import (
    pitch_shift_preserve_duration,
    resample_duration,
    transform_and_validate_harmonic_family,
)

SAMPLE_RATE = 8_000


def tone(frequency: float, frames: int = 16_384) -> np.ndarray:
    time = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * time)


def dominant_frequency(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    return float(np.argmax(spectrum) * SAMPLE_RATE / audio.size)


def test_pitch_shift_preserves_duration_and_moves_frequency() -> None:
    source = tone(220)

    shifted = pitch_shift_preserve_duration(source, 2.0, fft_size=1024)

    assert shifted.shape == source.shape
    assert abs(dominant_frequency(shifted) - 440) < 5.0


def test_duration_resampling_has_requested_length() -> None:
    source = tone(220)

    assert resample_duration(source, 12_000).shape == (12_000,)


def test_explicit_transformation_validates_transposed_family() -> None:
    source = tone(220) + 0.5 * tone(440)
    target = tone(330) + 0.5 * tone(660)

    result = transform_and_validate_harmonic_family(
        source,
        target,
        220,
        330,
        fft_size=1024,
        validation_fft_sizes=(256,),
        maximum_lag_frames=32,
    )

    assert abs(result.frequency_ratio - 1.5) < 1e-12
    assert result.validation.accepted
