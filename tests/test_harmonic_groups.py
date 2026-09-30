import numpy as np

from open_music.harmonic_groups import group_harmonic_partials
from open_music.partials import decompose_spectral_partials

SAMPLE_RATE = 8_000


def tone(frequency: float, frames: int = 32_768) -> np.ndarray:
    time = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * time)


def test_groups_fundamental_and_harmonics_into_one_family() -> None:
    audio = tone(220) + 0.6 * tone(440) + 0.3 * tone(660)
    partials = decompose_spectral_partials(
        audio,
        SAMPLE_RATE,
        fft_size=1024,
        minimum_track_frames=8,
        maximum_components=12,
    )

    grouped = group_harmonic_partials(partials)

    family = min(grouped.families, key=lambda item: abs(item.fundamental_hz - 220))
    assert {member.harmonic_number for member in family.members} >= {1, 2, 3}


def test_grouping_preserves_partial_reconstruction() -> None:
    audio = tone(180) + tone(360) + 0.2 * tone(517)
    partials = decompose_spectral_partials(audio, SAMPLE_RATE, fft_size=1024)

    grouped = group_harmonic_partials(partials)
    reconstructed = sum(
        (family.audio for family in grouped.families), np.zeros_like(audio)
    )
    reconstructed += sum(grouped.standalone, np.zeros_like(audio))
    reconstructed += grouped.residual

    assert np.allclose(reconstructed, audio)
    assert grouped.reconstruction_max_error < 1e-12
