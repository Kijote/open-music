import numpy as np
import pytest

from open_music.stream import analyze_stream, render_stream, stream_metrics


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("frames", [1, 2_047, 2_048, 9_731])
def test_granular_stream_roundtrip_covers_boundaries(frames: int, channels: int) -> None:
    rng = np.random.default_rng(frames + channels)
    audio = rng.standard_normal((frames, channels)) * 0.1

    stream = analyze_stream(audio)
    reconstruction = render_stream(stream)

    assert reconstruction.shape == audio.shape
    assert np.allclose(reconstruction, audio, atol=1e-12)
    assert stream_metrics(stream)["coverage_ratio"] == 1.0


def test_stream_reports_low_modularity_when_grains_are_unique() -> None:
    audio = np.arange(16_384, dtype=np.float64)[:, None]
    stream = analyze_stream(audio)
    metrics = stream_metrics(stream)

    assert metrics["event_count"] == metrics["unique_module_count"]
    assert metrics["reuse_ratio"] == 1.0
    assert metrics["storage_ratio"] > 1.0


def test_silence_reuses_one_content_addressed_grain() -> None:
    stream = analyze_stream(np.zeros((16_384, 1)))
    metrics = stream_metrics(stream)

    assert metrics["unique_module_count"] == 1
    assert metrics["event_count"] > 1
    assert np.all(render_stream(stream) == 0.0)


def test_invalid_granular_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid granular"):
        analyze_stream(np.zeros((10, 1)), window_frames=128, hop_frames=256)
