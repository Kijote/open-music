import numpy as np

from open_music.partials import decompose_spectral_partials


SAMPLE_RATE = 8_000


def tone(frequency: float, frames: int = 16_384) -> np.ndarray:
    time = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * time)


def test_two_tones_create_multiple_adaptive_partials() -> None:
    audio = tone(220) + 0.7 * tone(440)

    result = decompose_spectral_partials(
        audio,
        SAMPLE_RATE,
        fft_size=512,
        minimum_track_frames=8,
        maximum_components=8,
    )

    frequencies = sorted(track.mean_frequency_hz for track in result.tracks)
    assert len(result.components) >= 2
    assert any(abs(frequency - 220) < 25 for frequency in frequencies)
    assert any(abs(frequency - 440) < 25 for frequency in frequencies)


def test_partials_and_residual_reconstruct_exactly() -> None:
    rng = np.random.default_rng(12)
    audio = tone(330) + rng.standard_normal(16_384) * 0.02

    result = decompose_spectral_partials(audio, SAMPLE_RATE, fft_size=512)
    reconstructed = sum(result.components, np.zeros_like(audio)) + result.residual

    assert np.allclose(reconstructed, audio)
    assert result.reconstruction_max_error == 0.0


def test_short_impulse_is_not_mislabeled_as_persistent_partial() -> None:
    audio = np.zeros(8_192)
    audio[4_096] = 1.0

    result = decompose_spectral_partials(
        audio,
        SAMPLE_RATE,
        fft_size=512,
        minimum_track_frames=8,
    )

    assert not result.components
    assert np.array_equal(result.residual, audio)
