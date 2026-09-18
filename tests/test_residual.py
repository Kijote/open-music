import numpy as np

from open_music.residual import discover_residual_layers

SAMPLE_RATE = 8_000


def repeated_residual() -> np.ndarray:
    rng = np.random.default_rng(7)
    pulse = rng.standard_normal(320) * np.exp(-np.arange(320) / 35.0)
    pulse[0] = 4.0
    audio = np.zeros((5_000, 2))
    audio[512:832] += pulse[:, None] * 0.5
    audio[3_072:3_392] += pulse[:, None] * 0.8
    return audio


def test_discovers_reused_residual_module_and_converges() -> None:
    result = discover_residual_layers(
        repeated_residual(),
        SAMPLE_RATE,
        license_spdx="CC0-1.0",
        source_id="generated",
        maximum_candidate_seconds=0.1,
    )

    accepted = [residual_pass for residual_pass in result.passes if residual_pass.accepted]
    assert len(accepted) == 1
    assert len(result.modules) == 1
    assert len(result.events) == 2
    assert accepted[0].energy_after < accepted[0].energy_before
    assert np.sum(np.square(result.residual)) < 1e-20
    assert result.stop_reason == "residual_floor"


def test_rejects_singleton_residual_instead_of_memorizing_it() -> None:
    audio = repeated_residual()
    audio[3_072:] = 0.0

    result = discover_residual_layers(
        audio,
        SAMPLE_RATE,
        license_spdx="CC0-1.0",
        source_id="generated",
        maximum_candidate_seconds=0.1,
    )

    assert result.stop_reason == "no_reusable_clusters"
    assert not result.modules
    assert not result.events
    assert len(result.passes) == 1
    assert not result.passes[0].accepted
    assert np.array_equal(result.residual, audio)


def test_maximum_passes_is_a_deterministic_stop() -> None:
    result = discover_residual_layers(
        repeated_residual(),
        SAMPLE_RATE,
        license_spdx="CC0-1.0",
        source_id="generated",
        maximum_candidate_seconds=0.1,
        maximum_passes=1,
    )

    assert result.stop_reason == "maximum_passes"
    assert len(result.passes) == 1
    assert result.passes[0].accepted
