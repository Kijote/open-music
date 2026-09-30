import numpy as np

from open_music.components import separate_spectral_components


def energy(values: np.ndarray) -> float:
    return float(np.sum(values**2))


def test_components_reconstruct_stereo_input_exactly() -> None:
    rng = np.random.default_rng(4)
    audio = rng.standard_normal((8192, 2)) * 0.1

    result = separate_spectral_components(audio, fft_size=512)

    assert np.allclose(result.harmonic + result.percussive + result.residual, audio)
    assert result.reconstruction_max_error == 0.0
    assert result.harmonic.shape == audio.shape
    assert result.percussive.shape == audio.shape
    assert result.residual.shape == audio.shape


def test_sustained_tone_prefers_harmonic_component() -> None:
    time = np.arange(16_384) / 8_000
    tone = np.sin(2 * np.pi * 220 * time)

    result = separate_spectral_components(tone, fft_size=512)

    assert energy(result.harmonic) > energy(result.percussive) * 4.0


def test_impulse_train_prefers_percussive_component() -> None:
    impulses = np.zeros(16_384)
    impulses[::512] = 1.0

    result = separate_spectral_components(impulses, fft_size=512)

    assert energy(result.percussive) > energy(result.harmonic) * 2.0


def test_mask_means_form_a_partition() -> None:
    rng = np.random.default_rng(9)
    result = separate_spectral_components(rng.standard_normal(4096), fft_size=256)

    total = (
        result.harmonic_mask_mean
        + result.percussive_mask_mean
        + result.residual_mask_mean
    )
    assert abs(total - 1.0) < 1e-12
