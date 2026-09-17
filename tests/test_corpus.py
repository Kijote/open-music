import json
from pathlib import Path

import pytest

from open_music.corpus import CorpusAsset, checksum, load_manifest, verify_asset


def asset(data: bytes) -> CorpusAsset:
    return CorpusAsset(
        id="sample",
        url="https://example.invalid/sample.wav",
        cache_path="sample.wav",
        size=len(data),
        checksum_algorithm="sha256",
        checksum_value=checksum(data, "sha256"),
        license_spdx="CC0-1.0",
        source={},
    )


def test_checksum_verification_accepts_exact_content() -> None:
    verify_asset(asset(b"open-music"), b"open-music")


def test_checksum_verification_rejects_changed_content() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_asset(asset(b"open-music"), b"open-musix")


def test_manifest_rejects_non_commercial_audio(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "id": "invalid",
        "target_sample_rate": 44100,
        "assets": [
            {
                "id": "sample",
                "url": "https://example.invalid/sample.wav",
                "cache_path": "sample.wav",
                "size": 1,
                "checksum": {"algorithm": "sha256", "value": "x"},
                "license": {
                    "spdx": "CC-BY-NC-4.0",
                    "redistributable": True,
                    "derivatives": True,
                    "commercial_use": False
                },
                "source": {}
            }
        ],
        "tracks": [{"id": "track", "events": [{"sample_id": "sample"}]}]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not allowed"):
        load_manifest(path)
