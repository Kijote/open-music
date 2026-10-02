import numpy as np

from open_music.retrieval import (
    local_time_frequency_descriptor,
    retrieve_global_candidates,
)


def tone(frequency: float, frames: int = 4096, sample_rate: int = 8_000) -> np.ndarray:
    time = np.arange(frames) / sample_rate
    return np.sin(2 * np.pi * frequency * time)


def test_descriptor_preserves_temporal_order() -> None:
    low = tone(180)
    high = tone(620)
    forward = np.concatenate((low, high))
    reversed_order = np.concatenate((high, low))

    same = float(local_time_frequency_descriptor(forward) @ local_time_frequency_descriptor(forward))
    reversed_score = float(
        local_time_frequency_descriptor(forward)
        @ local_time_frequency_descriptor(reversed_order)
    )

    assert same > 0.999
    assert reversed_score < same - 0.1


def test_global_retrieval_is_not_limited_by_prior_families() -> None:
    blocks = (tone(180), tone(620), tone(180) * 0.4)

    candidates = retrieve_global_candidates(
        blocks,
        neighbors_per_block=1,
        minimum_score=0.5,
    )

    matches = {(item.query_index, item.candidate_index) for item in candidates}
    assert (0, 2) in matches
    assert (2, 0) in matches


def test_global_retrieval_filters_duration_mismatch() -> None:
    blocks = (tone(180), tone(180, frames=1024))

    assert not retrieve_global_candidates(blocks, minimum_duration_ratio=0.8)
