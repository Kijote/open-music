from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_REDISTRIBUTABLE_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"}


@dataclass(frozen=True)
class CorpusAsset:
    id: str
    url: str
    cache_path: str
    size: int
    checksum_algorithm: str
    checksum_value: str
    license_spdx: str
    source: dict[str, Any]


@dataclass(frozen=True)
class Corpus:
    id: str
    target_sample_rate: int
    assets: tuple[CorpusAsset, ...]
    tracks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExternalCorpus:
    id: str
    recordings: tuple[CorpusAsset, ...]


def _load_asset(item: dict[str, Any], ids: set[str]) -> CorpusAsset:
    license_data = item["license"]
    spdx = license_data["spdx"]
    if spdx not in ALLOWED_REDISTRIBUTABLE_LICENSES:
        raise ValueError(f"license {spdx} is not allowed in the redistributable corpus")
    required_flags = ("redistributable", "derivatives", "commercial_use")
    if not all(license_data.get(flag) is True for flag in required_flags):
        raise ValueError(f"asset {item['id']} does not grant all required rights")
    if item["id"] in ids:
        raise ValueError(f"duplicate asset id: {item['id']}")
    ids.add(item["id"])
    return CorpusAsset(
        id=item["id"],
        url=item["url"],
        cache_path=item["cache_path"],
        size=int(item["size"]),
        checksum_algorithm=item["checksum"]["algorithm"],
        checksum_value=item["checksum"]["value"],
        license_spdx=spdx,
        source=item["source"],
    )


def load_manifest(path: Path) -> Corpus:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported corpus manifest schema")
    if not raw.get("assets") or not raw.get("tracks"):
        raise ValueError("corpus must contain assets and tracks")

    assets: list[CorpusAsset] = []
    ids: set[str] = set()
    for item in raw["assets"]:
        assets.append(_load_asset(item, ids))

    for track in raw["tracks"]:
        for event in track["events"]:
            if event["sample_id"] not in ids:
                raise ValueError(f"unknown sample id: {event['sample_id']}")

    return Corpus(
        id=raw["id"],
        target_sample_rate=int(raw["target_sample_rate"]),
        assets=tuple(assets),
        tracks=tuple(raw["tracks"]),
    )


def load_external_manifest(path: Path) -> ExternalCorpus:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported external corpus manifest schema")
    if not raw.get("recordings"):
        raise ValueError("external corpus must contain recordings")

    ids: set[str] = set()
    recordings = tuple(_load_asset(item, ids) for item in raw["recordings"])
    return ExternalCorpus(id=raw["id"], recordings=recordings)


def checksum(data: bytes, algorithm: str) -> str:
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    if algorithm == "git-sha1":
        header = f"blob {len(data)}\0".encode()
        return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
    raise ValueError(f"unsupported checksum algorithm: {algorithm}")


def verify_asset(asset: CorpusAsset, data: bytes) -> None:
    if len(data) != asset.size:
        raise ValueError(f"{asset.id}: expected {asset.size} bytes, received {len(data)}")
    actual = checksum(data, asset.checksum_algorithm)
    if actual != asset.checksum_value:
        raise ValueError(f"{asset.id}: checksum mismatch")


def fetch_corpus(corpus: Corpus, cache_dir: Path) -> dict[str, Path]:
    return fetch_assets(corpus.assets, cache_dir)


def fetch_assets(assets: tuple[CorpusAsset, ...], cache_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    cache_root = cache_dir.resolve()
    for asset in assets:
        destination = (cache_root / asset.cache_path).resolve()
        if cache_root not in destination.parents:
            raise ValueError(f"unsafe cache path: {asset.cache_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            data = destination.read_bytes()
            try:
                verify_asset(asset, data)
            except ValueError:
                destination.unlink()
            else:
                resolved[asset.id] = destination
                continue

        with urllib.request.urlopen(asset.url, timeout=30) as response:
            data = response.read()
        verify_asset(asset, data)
        destination.write_bytes(data)
        resolved[asset.id] = destination

    return resolved
