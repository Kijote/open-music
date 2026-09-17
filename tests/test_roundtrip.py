import numpy as np

from open_music import Event, Sample, match_sample, measure, render

SAMPLE_RATE = 8_000


def decaying_sine(frequency: float, frames: int) -> np.ndarray:
    time = np.arange(frames) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * time) * np.exp(-18 * time)


def test_modular_representation_reconstructs_reference_exactly() -> None:
    kick = Sample("kick", decaying_sine(60, 640), SAMPLE_RATE, "CC0-1.0", "generated")
    snare_audio = decaying_sine(190, 320) + 0.25 * decaying_sine(613, 320)
    snare = Sample("snare", snare_audio, SAMPLE_RATE, "CC0-1.0", "generated")
    library = {sample.id: sample for sample in (kick, snare)}
    events = [
        Event("kick", 0),
        Event("snare", 2_000, gain=0.7),
        Event("kick", 4_000, gain=0.9),
        Event("snare", 6_000, gain=0.65),
    ]

    reference = render(library, events, total_frames=8_000, sample_rate=SAMPLE_RATE)
    reconstruction = render(library, events, total_frames=8_000, sample_rate=SAMPLE_RATE)
    result = measure(reference, reconstruction)

    assert result.mean_absolute_error == 0.0
    assert result.snr_db == float("inf")
    assert result.explained_energy == 1.0


def test_library_matching_selects_the_closest_module() -> None:
    kick = Sample("kick", decaying_sine(60, 640), SAMPLE_RATE, "CC0-1.0", "generated")
    tom = Sample("tom", decaying_sine(130, 640), SAMPLE_RATE, "CC0-1.0", "generated")
    library = {sample.id: sample for sample in (kick, tom)}

    query = kick.audio[:, 0] * 0.42
    match = match_sample(query, library)

    assert match.sample_id == "kick"
    assert match.similarity > 0.999999


def test_metrics_expose_unexplained_residual() -> None:
    reference = np.ones((100, 2))
    approximation = reference * 0.75

    result = measure(reference, approximation)

    assert result.mean_absolute_error == 0.25
    assert 0.93 < result.explained_energy < 0.94
