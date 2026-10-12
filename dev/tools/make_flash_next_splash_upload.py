#!/usr/bin/env python3
"""Stage the Swift Qwen3.8-Flash-Next-V3 Splash package for a Hub upload.

Builds a hardlink tree (zero extra disk) of the prepared package at
``~/models/swift-qwen38-flash-next-v3/prepared`` under an upload directory,
adds the manifest ``artifacts`` digest list required by the Splash runtime
package installer, plus a Hub-facing ``config.json`` and ``README.md``.

Usage: python3 dev/tools/make_flash_next_splash_upload.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import models as installer  # noqa: E402

SOURCE = Path.home() / "models/swift-qwen38-flash-next-v3/prepared"
STAGE = Path.home() / "models/upload-flash-next-splash/Qwen3.8-Flash-Next-Splash"
SHA_CACHE = Path("/tmp/swift_v3_sha256.txt")

DIGEST_CHUNK = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(DIGEST_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_digests() -> dict[str, str]:
    """Path -> sha256 from a prior hashing pass (sha256 size path lines)."""
    if not SHA_CACHE.exists():
        return {}
    digests = {}
    for line in SHA_CACHE.read_text().splitlines():
        fields = line.split()
        if len(fields) == 3:
            digest, _, name = fields
            digests[name] = digest
    return digests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="rebuild the staging tree from scratch"
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="recompute digests instead of reusing the cached pass",
    )
    args = parser.parse_args()

    if STAGE.exists() and args.force:
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)

    cached = {} if args.hash else cached_digests()
    records = []
    files = sorted(
        path
        for path in SOURCE.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    for path in files:
        relative = path.relative_to(SOURCE).as_posix()
        link = STAGE / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.hardlink_to(path)
        digest = cached.get(relative)
        if digest is None:
            print(f"hashing {relative} ...", flush=True)
            digest = sha256(path)
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": digest}
        )

    manifest = json.loads((SOURCE / "manifest.json").read_text())
    manifest["artifacts"] = records
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    config = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "splash-packed-q4-qwen4exp",
        "format": "splash-packed-q4-qwen4exp",
        "schema_version": 5,
        "quantization_config": {
            "quant_method": "tiled-q4-q8",
            "bits": 4,
            "group_size": 64,
        },
        "speculative": {"method": "mtp-ngram", "proposal_tokens": 7},
        "runtime": "https://github.com/npanj/slipstream",
        "package_manifest": "manifest.json",
    }
    (STAGE / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    validated = installer.validate_package_manifest(STAGE / "manifest.json")
    print(
        f"staged {len(records)} artifacts "
        f"({sum(r['size'] for r in records) / 2**30:.2f} GiB), "
        f"manifest validated: {validated['model']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
