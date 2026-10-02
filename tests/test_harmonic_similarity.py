import numpy as np

from open_music.harmonic_groups import group_harmonic_partials
from open_music.harmonic_similarity import (
    cluster_harmonic_families,
    harmonic_family_similarity_matrix,
)
from open_music.partials import decompose_spectral_partials

SAMPLE_RATE = 8_000


def tone(frequency: float, frames: int = 32_768) -> np.ndarray:
    time = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * time)


def family(fundamental: float, second: float, third: float):
    audio = tone(fundamental) + second * tone(2 * fundamental) + third * tone(3 * fundamental)
    partials = decompose_spectral_partials(
        audio, SAMPLE_RATE, fft_size=1024, minimum_track_frames=8, maximum_components=12
    )
    grouped = group_harmonic_partials(partials)
    selected = min(grouped.families, key=lambda item: abs(item.fundamental_hz - fundamental))
    return selected, partials


def test_similarity_is_pitch_invariant_but_timbre_sensitive() -> None:
    first, first_partials = family(180, 0.6, 0.3)
    transposed, transposed_partials = family(240, 0.6, 0.3)
    different, different_partials = family(180, 0.05, 0.9)
    families = (first, transposed, different)
    decompositions = (first_partials, transposed_partials, different_partials)

    matrix = harmonic_family_similarity_matrix(families, decompositions)

    assert matrix[0][1] > 0.98
    assert matrix[0][2] < matrix[0][1]


def test_clustering_requires_every_family_pair_to_match() -> None:
    first, first_partials = family(180, 0.6, 0.3)
    second, second_partials = family(240, 0.6, 0.3)
    third, third_partials = family(300, 0.6, 0.3)
    families = (first, second, third)
    matrix = (
        (1.0, 0.99, 0.50),
        (0.99, 1.0, 0.99),
        (0.50, 0.99, 1.0),
    )

    assert cluster_harmonic_families(families, matrix) == ((0, 1), (2,))
    assert all(item.components for item in (first_partials, second_partials, third_partials))


def test_similarity_preserves_harmonic_evolution_over_time() -> None:
    half = 16_384
    first_audio = np.concatenate(
        (tone(180, half) + 0.8 * tone(360, half), tone(180, half) + 0.1 * tone(360, half))
    )
    reversed_audio = np.concatenate(
        (tone(180, half) + 0.1 * tone(360, half), tone(180, half) + 0.8 * tone(360, half))
    )
    pairs = []
    for audio in (first_audio, reversed_audio):
        partials = decompose_spectral_partials(
            audio, SAMPLE_RATE, fft_size=1024, minimum_track_frames=8, maximum_components=12
        )
        grouped = group_harmonic_partials(partials)
        selected = min(grouped.families, key=lambda item: abs(item.fundamental_hz - 180))
        pairs.append((selected, partials))

    matrix = harmonic_family_similarity_matrix(
        (pairs[0][0], pairs[1][0]),
        (pairs[0][1], pairs[1][1]),
    )

    assert matrix[0][1] < 0.95
