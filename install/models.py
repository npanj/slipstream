#!/usr/bin/env python3

"""Verify and atomically install Splash packages from Hugging Face snapshots.

The native model descriptor validates architecture, tensors, headers, and
execution geometry before mapping weights.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

if __package__:
    from . import paths
else:
    import paths

MODELS = paths.MODELS
ALIGNMENT = 16384
HUB_ENDPOINT = "https://huggingface.co"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
TOKENIZER_FILES = {
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
# Qwen4Exp packages ship the chat template inline in tokenizer_config.json.
QWEN4EXP_TOKENIZER_FILES = TOKENIZER_FILES - {"chat_template.jinja"}
# The draft-vocab ranking table is a u32 id list the runtime reads with
# pread(); it is not an mmap'd packed weight file, so the section alignment
# rule does not apply to it.
UNALIGNED_DATA_FILES = {"target/draft-vocab.bin"}
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
PACKAGE_FORMATS = {
    "splash-packed-q4-qwen4exp": (
        5,
        "MDFN0031",
        {"target": "qwen4exp", "draft": "DFlash2DraftModel"},
    ),
}
# Splash 1.0 package formats.
RETIRED_FORMATS = {
    "splash-packed-q4": "incoai/Qwen3.8-27B-Splash",
    "splash-packed-q4-moe": "incoai/Qwen3.6-35B-A3B-Splash",
}


class ModelError(RuntimeError):
    pass


class GgufRepository(ModelError):
    """The repository holds GGUF shards rather than a runtime package."""

    def __init__(self, model_id: str, revision: str, files: dict[str, int]):
        super().__init__(f"{model_id} is a GGUF repository")
        self.model_id, self.revision, self.files = model_id, revision, files


# A GGUF repository is installed as a real folder: its shards, the MTP head, and
# the package the launcher prepares from them in prepared/. The marker pins the
# Hub revision, so a resumed download never mixes two commits and a prepared
# package never goes stale under an updated repository.
GGUF_MARKER = ".slipstream-gguf.json"
# Speculative drafting needs Qwen3.8-Flash-Next's MTP head, which most GGUF
# repositories (the Swift variant's among them) leave out. This copy is
# converted from the original model, so it drafts for every Flash-Next GGUF.
MTP_SIDECAR = "MTP/mtp-shared-Q4_K_M.gguf"
MTP_REPO = "nitinpanj/qwen38-flash-next-v3"
MTP_REVISION = "e2982050848c67fe8aa9073e8c4205d272cc4f20"
MTP_SIZE = 1907151936
# The GGUF architectures the converter turns into a package (models/qwen4exp and models/qwen38).
GGUF_ARCHITECTURES = ("qwen4exp", "qwen38", "qwen3.5", "qwen2")
GGUF_HEADER_BYTES = 256 * 1024
# llama.cpp's gguf-split names: <stem>-00001-of-00003.gguf.
GGUF_SPLIT = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
# What preparing writes before it frees the source: up to 8 converter workers'
# layers of 1.5 GiB.
PREPARE_IN_FLIGHT_BYTES = 12 << 30


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def validate_repo_id(value: str) -> str:
    # Keep argument validation available before the Hub dependency is installed.
    if (
        not isinstance(value, str)
        or not REPO_ID.fullmatch(value)
        or "--" in value
        or ".." in value
        or value.endswith(".git")
    ):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    return value


def parse_repo_id(value: str) -> str:
    try:
        return validate_repo_id(value)
    except ModelError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def validate_package_manifest(path: Path):
    manifest = read_json(path)
    format_ = manifest.get("format")
    format_name = format_.get("name") if isinstance(format_, dict) else None
    if format_name in RETIRED_FORMATS:
        raise ModelError(
            f"{format_name} is a Splash 1.0 package format, which the Slipstream v2 "
            "engine does not load; it serves Qwen3.8-Flash-Next "
            "(splash-packed-q4-qwen4exp packages, or a folder of its GGUF files)"
        )
    layout = PACKAGE_FORMATS.get(format_name) if isinstance(format_name, str) else None
    if (
        layout is None
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != layout[0]
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"].strip()
        or not isinstance(manifest.get("execution_geometry"), dict)
    ):
        raise ModelError("repository is not a supported Splash runtime package")
    expected_format = {
        "section_alignment_bytes": ALIGNMENT,
        "target_layer_magic": layout[1],
        "draft_layer_magic": "MDFD0004",
        "vision_magic": "MDFV0001",
    }
    if any(
        type(format_.get(key)) is not type(value) or format_[key] != value
        for key, value in expected_format.items()
    ):
        raise ModelError("runtime package has an unsupported packed weight format")
    for key, architecture in layout[2].items():
        declaration = manifest.get(key)
        if (
            not isinstance(declaration, dict)
            or declaration.get("architecture") != architecture
        ):
            raise ModelError(f"runtime package has an unsupported {key} architecture")

    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ModelError("runtime package manifest has no artifact list")
    artifact_paths = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not isinstance(record["path"], str)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        pure = PurePosixPath(record["path"])
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or pure.as_posix() != record["path"]
            or any(character in record["path"] for character in "\\*?[]")
            or any(ord(character) < 32 for character in record["path"])
            or record["path"] == "manifest.json"
            or type(record["size"]) is not int
            or record["size"] <= 0
            or not is_hex_digest(record["sha256"], 64)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        if record["path"] in artifact_paths:
            raise ModelError("runtime package artifact paths are not unique")
        if (
            pure.suffix == ".bin"
            and record["path"] not in UNALIGNED_DATA_FILES
            and record["size"] % ALIGNMENT
        ):
            raise ModelError(
                f"runtime package packed file is unaligned: {record['path']}"
            )
        artifact_paths.add(record["path"])
    if any(
        parent.as_posix() in artifact_paths
        for name in artifact_paths
        for parent in PurePosixPath(name).parents
    ):
        raise ModelError("runtime package artifact paths overlap")
    if format_name == "splash-packed-q4-qwen4exp":
        required_files = {
            "target/embedding.bin",
            "target/head.bin",
            "target/ngram.bin",
            "target/draft-vocab.bin",
            "target/mtp-layer.bin",
            "target/mtp-combiner.bin",
            "draft/model.bin",
            *(f"target/layer-{index}.bin" for index in range(48)),
            *(f"draft/layer-{index}.bin" for index in range(5)),
            *(f"tokenizer/{name}" for name in QWEN4EXP_TOKENIZER_FILES),
        }
    elif format_name in ("splash-packed-q8", "splash-packed-q4"):
        required_files = {
            "target/embedding.bin",
            "target/head.bin",
            *(f"target/layer-{index}.bin" for index in range(64)),
            *(f"tokenizer/{name}" for name in TOKENIZER_FILES),
        }
    else:
        required_files = set()
    missing = required_files - artifact_paths
    if missing:
        raise ModelError(
            "runtime package artifact list is missing: " + ", ".join(sorted(missing))
        )
    return manifest


def verify_artifacts(root: Path, manifest, *, full: bool):
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ModelError(f"installed artifact has the wrong size: {record['path']}")
        if (
            path.suffix == ".bin"
            and record["path"] not in UNALIGNED_DATA_FILES
            and record["size"] % ALIGNMENT
        ):
            raise ModelError(f"installed packed file is unaligned: {record['path']}")
        if full and sha256(path) != record["sha256"].lower():
            raise ModelError(f"installed artifact checksum changed: {record['path']}")


def installed_root(models: Path, model_id: str) -> Path:
    return models / validate_repo_id(model_id)


def verify_installed(
    models: Path,
    *,
    model_id: str,
    full: bool,
):
    root = installed_root(models, model_id)
    manifest = validate_package_manifest(root / "manifest.json")
    verify_artifacts(root, manifest, full=full)
    return model_id


def _snapshot_revision(snapshot: Path, model_id: str) -> str:
    if (
        snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "models--" + model_id.replace("/", "--")
        or not is_hex_digest(snapshot.name, 40)
    ):
        raise ModelError(
            "installed package is not a snapshot of the requested Hub repository"
        )
    return snapshot.name


def _retain_snapshot_ref(snapshot: Path, model_id: str, installation: Path):
    revision = _snapshot_revision(snapshot, model_id)
    # Each installation owns its references; Hub branch updates and other
    # installations must not unpin this installation's current weights.
    owner_path = installation.parent.resolve() / installation.name
    owner = hashlib.sha256(os.fsencode(owner_path)).hexdigest()
    ref = snapshot.parent.parent / "refs" / "splash" / owner / revision
    try:
        existing = ref.read_text()
    except FileNotFoundError:
        pass
    else:
        if existing != revision:
            raise ModelError(f"invalid installed snapshot reference: {ref}")
        return ref
    ref.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ref.parent) as temporary:
        temporary.write(revision.encode())
        temporary.flush()
        try:
            os.link(temporary.name, ref)
        except FileExistsError:
            if ref.read_text() != revision:
                raise ModelError(f"invalid installed snapshot reference: {ref}")
    return ref


def _download_snapshot(model_id: str, token):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    options = {
        "repo_id": model_id,
        "repo_type": "model",
        "token": token or False,
        "endpoint": HUB_ENDPOINT,
    }
    for _ in range(3):
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            model_id, revision="main", files_metadata=True
        )
        if not is_hex_digest(info.sha, 40):
            raise ModelError("Hub did not resolve the model to a snapshot commit")
        manifest_file = next(
            (item for item in info.siblings if item.rfilename == "manifest.json"), None
        )
        if manifest_file is None:
            files = {item.rfilename: item.size for item in info.siblings}
            if any(_is_gguf_shard(name) for name in files):
                raise GgufRepository(model_id, info.sha, files)
            raise ModelError(
                "repository has neither a Splash runtime package manifest.json "
                "nor GGUF files"
            )
        if (
            type(manifest_file.size) is not int
            or not 0 < manifest_file.size <= MAX_MANIFEST_BYTES
        ):
            raise ModelError("runtime package manifest.json has an invalid size")
        # Resolve the named revision through the Hub cache before pinning every
        # artifact. A branch update between metadata and download retries before
        # any weights are downloaded.
        manifest_path = Path(
            hf_hub_download(filename="manifest.json", revision="main", **options)
        )
        revision = _snapshot_revision(manifest_path.parent, model_id)
        if revision == info.sha:
            break
    else:
        raise ModelError("Hub main changed repeatedly during installation; retry")

    options["revision"] = revision
    try:
        manifest = validate_package_manifest(manifest_path)
    except ModelError:
        manifest_path = Path(
            hf_hub_download(
                filename="manifest.json",
                force_download=True,
                **options,
            )
        )
        manifest = validate_package_manifest(manifest_path)
    manifest_sha = sha256(manifest_path)
    published = {item.rfilename: item for item in info.siblings}
    for record in manifest["artifacts"]:
        item = published.get(record["path"])
        if item is None or item.size != record["size"]:
            raise ModelError(f"Hub artifact does not match manifest: {record['path']}")
        lfs = getattr(item, "lfs", None)
        if lfs is not None and lfs.sha256.lower() != record["sha256"].lower():
            raise ModelError(
                f"Hub artifact hash does not match manifest: {record['path']}"
            )
    snapshot = Path(
        snapshot_download(
            allow_patterns=[
                "manifest.json",
                *(r["path"] for r in manifest["artifacts"]),
            ],
            **options,
        )
    )
    if sha256(snapshot / "manifest.json") != manifest_sha:
        raise ModelError("runtime package manifest changed during download")
    if _snapshot_revision(snapshot, model_id) != revision:
        raise ModelError("Hub returned a different runtime package revision")
    # Repair only corrupt cached artifacts. A manifest error is deterministic
    # and must not trigger a second download of all model weights.
    for record in manifest["artifacts"]:
        path = snapshot / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or sha256(path) != record["sha256"].lower()
        ):
            hf_hub_download(filename=record["path"], force_download=True, **options)
            verify_artifacts(snapshot, {"artifacts": [record]}, full=True)
    return snapshot.resolve()


def _cached_snapshot(model_id):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "manifest.json", revision="main")
    if not isinstance(path, str):
        return None
    snapshot = Path(path).parent
    try:
        _snapshot_revision(snapshot, model_id)
        manifest = validate_package_manifest(snapshot / "manifest.json")
        verify_artifacts(snapshot, manifest, full=True)
    except (ModelError, OSError):
        return None
    return snapshot.resolve()


def _hub_token():
    try:
        from huggingface_hub import get_token
    except ImportError as error:
        raise ModelError(
            "missing dependency huggingface_hub; reinstall Splash"
        ) from error
    return os.environ.get("HF_TOKEN") or get_token()


def _hub_error(error: Exception, token, what: str) -> ModelError:
    from huggingface_hub.errors import HfHubHTTPError

    message = str(error)
    if token:
        message = message.replace(token, "[redacted]")
    if (
        isinstance(error, HfHubHTTPError)
        and error.response is not None
        and error.response.status_code in (401, 403)
    ):
        message += (
            "; set HF_TOKEN or run 'hf auth login' with access to this repository"
        )
    return ModelError(f"could not download {what}: {message}")


def resolve_snapshot(model_id: str):
    validate_repo_id(model_id)
    token = _hub_token()
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        return _download_snapshot(model_id, token)
    except Exception as error:
        if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError)):
            if cached := _cached_snapshot(model_id):
                print("Using a verified cached model while offline.", flush=True)
                return cached
        if isinstance(error, ModelError):
            raise
        raise _hub_error(
            error, token, f"Splash runtime package {model_id}@main"
        ) from error


def _is_gguf_shard(name: str) -> bool:
    # The converter reads the shards at the top of the folder; MTP/ holds the
    # draft head, which it finds by name.
    return "/" not in name and name.endswith(".gguf")


def is_gguf_installation(root: Path) -> bool:
    return not root.is_symlink() and (root / GGUF_MARKER).is_file()


def gguf_downloaded(root: Path) -> bool:
    """Whether a GGUF installation's download finished, or is no longer needed.

    Preparing uses the shards up by default, so a prepared installation has
    none, and one whose package was removed must download them again.
    """
    if gguf_prepared(root):
        return True
    try:
        downloaded = read_json(root / GGUF_MARKER).get("downloaded") is True
    except ModelError:
        downloaded = False
    return downloaded and any(_is_gguf_shard(path.name) for path in root.glob("*.gguf"))


def remove_gguf_source(root: Path) -> None:
    """Delete the shards and MTP head of a prepared installation: its package replaces them."""
    if not gguf_prepared(root):
        return
    for path in [*sorted(root.glob("*.gguf")), root / MTP_SIDECAR]:
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            print(
                f"Deleted {path.relative_to(root)} ({size / 2**30:.1f} GiB); "
                "the prepared package replaces it",
                flush=True,
            )


def gguf_prepared(root: Path) -> bool:
    """Whether the launcher has already prepared a package from the shards."""
    return (root / "prepared/manifest.json").is_file() and (
        (root / "prepared/target/layer-0.bin").is_file()
    )


def gguf_model_files(names) -> list[str]:
    """The repository's GGUF files, which must be one model: a single file, or one
    complete set of split files, at the top level (where the converter reads them)."""
    shards = sorted(name for name in names if _is_gguf_shard(name))
    if not shards:
        variants = sorted(n for n in names if n.endswith(".gguf") and n != MTP_SIDECAR)
        if variants:
            raise ModelError(
                f"repository keeps its GGUF files in sub-folders ({variants[0]}, ...); "
                "Slipstream converts one model from the top level of a repository"
            )
        raise ModelError(
            "repository has neither a Splash runtime package manifest.json nor GGUF files"
        )
    if len(shards) == 1:
        return shards
    splits = [GGUF_SPLIT.match(name) for name in shards]
    if all(splits) and len({(m["stem"], m["count"]) for m in splits}) == 1:
        if [int(m["index"]) for m in splits] == list(
            range(1, int(splits[0]["count"]) + 1)
        ):
            return shards
        raise ModelError(
            f"repository is missing some of {splits[0]['stem']}'s split GGUF files"
        )
    raise ModelError(
        f"repository holds {len(shards)} GGUF files that are not one model's split files "
        f"({', '.join(shards[:3])}{', ...' if len(shards) > 3 else ''}); "
        "Slipstream converts one model per repository"
    )


def gguf_architecture(header: bytes):
    """general.architecture from the start of a GGUF file, or None if it is not there."""
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    offset = 0

    def take(size):
        nonlocal offset
        if offset + size > len(header):
            raise IndexError
        offset += size
        return header[offset - size : offset]

    def string():
        (length,) = struct.unpack("<Q", take(8))
        return take(length)

    def skip(kind):
        if kind in scalar:
            take(scalar[kind])
        elif kind == 8:
            string()
        elif kind == 9:
            item, count = struct.unpack("<IQ", take(12))
            for _ in range(count):
                skip(item)
        else:
            raise IndexError

    try:
        if take(4) != b"GGUF":
            return None
        _version, _tensors, count = struct.unpack("<IQQ", take(20))
        for _ in range(count):
            key = string()
            (kind,) = struct.unpack("<I", take(4))
            if key == b"general.architecture" and kind == 8:
                return string().decode("utf-8", "replace")
            skip(kind)
    except (IndexError, struct.error):
        return None
    return None


def _check_gguf_architecture(model_id: str, revision: str, shard: str, token):
    """Read the first shard's header from the Hub: the converter takes qwen4exp only."""
    from huggingface_hub import get_session, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(model_id, shard, revision=revision, endpoint=HUB_ENDPOINT)
    headers = build_hf_headers(token=token or False)
    headers["Range"] = f"bytes=0-{GGUF_HEADER_BYTES - 1}"
    try:
        response = get_session().get(url, headers=headers, timeout=60)
        response.raise_for_status()
    except Exception as error:
        raise _hub_error(error, token, f"the header of {shard}") from error
    architecture = gguf_architecture(response.content[:GGUF_HEADER_BYTES])
    if architecture not in GGUF_ARCHITECTURES:
        raise ModelError(
            f"{shard} is a {architecture or 'unrecognised'} model; Slipstream converts "
            f"{' and '.join(GGUF_ARCHITECTURES)} (Qwen3.8-Flash-Next) GGUF files only"
        )


def check_repository(model_id: str) -> dict:
    """What `pull` would install from a repository, without downloading it.

    Raises ModelError with the reason when Slipstream cannot serve it; only a
    package's manifest.json is fetched (into the Hub cache, as pull does).
    """
    from huggingface_hub import HfApi, hf_hub_download

    validate_repo_id(model_id)
    token = _hub_token()
    try:
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            model_id, revision="main", files_metadata=True
        )
    except Exception as error:
        raise _hub_error(error, token, f"{model_id}@main") from error
    if not is_hex_digest(info.sha, 40):
        raise ModelError("Hub did not resolve the model to a snapshot commit")
    files = {item.rfilename: item.size or 0 for item in info.siblings}
    result = {"model": model_id, "revision": info.sha}
    if "manifest.json" in files:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="manifest.json",
                revision=info.sha,
                repo_type="model",
                token=token or False,
                endpoint=HUB_ENDPOINT,
            )
        except Exception as error:
            raise _hub_error(error, token, f"{model_id}'s manifest.json") from error
        manifest = validate_package_manifest(Path(path))
        artifacts = manifest["artifacts"]
        for record in artifacts:
            if files.get(record["path"]) != record["size"]:
                raise ModelError(
                    f"Hub artifact does not match manifest: {record['path']}"
                )
        return {
            **result,
            "kind": "package",
            "bytes": files["manifest.json"] + sum(r["size"] for r in artifacts),
        }
    shards = gguf_model_files(files)
    _check_gguf_architecture(model_id, info.sha, shards[0], token)
    has_mtp = MTP_SIDECAR in files
    return {
        **result,
        "kind": "gguf",
        "files": len(shards),
        "bytes": sum(files[name] for name in shards)
        + (files[MTP_SIDECAR] if has_mtp else MTP_SIZE),
        "mtp": model_id if has_mtp else MTP_REPO,
    }


def _missing_bytes(root: Path, files: dict[str, int]) -> int:
    missing = 0
    for name, size in files.items():
        path = root / name
        if not (path.is_file() and path.stat().st_size == size):
            missing += size or 0
    return missing


def install_gguf(found: GgufRepository, root: Path):
    """Download a GGUF repository's shards, and the MTP head it lacks, into root."""
    from huggingface_hub import hf_hub_download, snapshot_download

    model_id = found.model_id
    shards = {n: found.files[n] for n in gguf_model_files(found.files)}
    wanted = dict(shards)
    has_mtp = MTP_SIDECAR in found.files
    if has_mtp:
        wanted[MTP_SIDECAR] = found.files[MTP_SIDECAR]

    marker = root / GGUF_MARKER
    if is_gguf_installation(root):
        recorded = read_json(marker)
        revision = recorded.get("revision")
        if recorded.get("model") != model_id or not is_hex_digest(revision, 40):
            raise ModelError(f"{marker} does not describe {model_id}")
    else:
        revision = found.revision
    if revision != found.revision:
        # The listing describes main; the pinned commit's files may differ.
        from huggingface_hub import HfApi

        token = _hub_token()
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            model_id, revision=revision, files_metadata=True
        )
        files = {item.rfilename: item.size for item in info.siblings}
        return install_gguf(GgufRepository(model_id, revision, files), root)

    needed = _missing_bytes(root, wanted)
    if not has_mtp:
        needed += _missing_bytes(root, {MTP_SIDECAR: MTP_SIZE})
    if not gguf_prepared(root):
        # Preparing uses the shards up as it converts them (unless the server
        # keeps them, when the converter checks for room itself), so it needs
        # room only for the package's growth over them and the parts in flight.
        needed += (
            sum(size or 0 for size in shards.values()) // 20 + PREPARE_IN_FLIGHT_BYTES
        )
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if needed > free:
        raise ModelError(
            f"{model_id} needs {needed / 2**30:.1f} GiB more disk space "
            f"(the download, and room to prepare it); {free / 2**30:.1f} GiB is free"
        )
    token = _hub_token()
    # Refused before anything is written: the converter takes qwen4exp only.
    _check_gguf_architecture(model_id, revision, sorted(shards)[0], token)
    marker.write_text(json.dumps({"model": model_id, "revision": revision}) + "\n")

    common = {"repo_type": "model", "token": token or False, "endpoint": HUB_ENDPOINT}
    total = sum(size or 0 for size in wanted.values())
    print(
        f"Downloading {model_id} ({len(shards)} GGUF files, {total / 2**30:.1f} GiB) "
        f"into {root}",
        flush=True,
    )
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=root,
            allow_patterns=sorted(wanted),
            **common,
        )
    except Exception as error:
        raise _hub_error(error, token, f"{model_id}@{revision[:12]}") from error
    if not has_mtp:
        print(
            f"{model_id} has no MTP draft head; fetching {MTP_SIDECAR} from {MTP_REPO} "
            "for speculative drafting",
            flush=True,
        )
        try:
            hf_hub_download(
                repo_id=MTP_REPO,
                filename=MTP_SIDECAR,
                revision=MTP_REVISION,
                local_dir=root,
                **common,
            )
        except Exception as error:
            raise _hub_error(error, token, f"the MTP head from {MTP_REPO}") from error
        wanted[MTP_SIDECAR] = MTP_SIZE
    for name, size in wanted.items():
        path = root / name
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            raise ModelError(f"downloaded file has the wrong size: {path}")
    marker.write_text(
        json.dumps({"model": model_id, "revision": revision, "downloaded": True}) + "\n"
    )
    print(f"Downloaded {model_id} into {root}", flush=True)


@contextmanager
def installation_lock(models: Path):
    lock_path = models / ".install.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another Splash model installation is running; waiting...",
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def install_snapshot(snapshot: Path, destination: Path):
    if destination.exists() and not destination.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".prepare-{destination.name}-", dir=destination.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(snapshot, temporary, target_is_directory=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def _exclude_from_backups(models: Path):
    # Every model can be downloaded again, and the store holds hundreds of
    # gigabytes. The exclusion is sticky (an extended attribute) and needs no
    # administrator rights; a failure only means Time Machine backs it up.
    try:
        subprocess.run(
            ["tmutil", "addexclusion", str(models)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def prepare(args):
    validate_repo_id(args.model)
    models = args.models.resolve()
    if not models.is_dir():
        models.mkdir(parents=True, exist_ok=True)
        _exclude_from_backups(models)
    root = installed_root(models, args.model)
    with installation_lock(models):
        if is_gguf_installation(root):
            if gguf_downloaded(root):
                print(f"GGUF model {args.model} is already downloaded in {root}")
                return
            # An interrupted download resumes at the commit it started from.
            print(f"Resuming the download of {args.model} into {root}", flush=True)
            try:
                resolve_snapshot(args.model)
            except GgufRepository as found:
                install_gguf(found, root)
                return
            raise ModelError(
                f"{args.model} is no longer a GGUF repository; move {root} aside"
            )
        try:
            _snapshot_revision(root.resolve(), args.model)
            manifest = validate_package_manifest(root / "manifest.json")
            verify_artifacts(root, manifest, full=False)
        except (ModelError, OSError):
            if root.exists() and not root.is_symlink():
                raise ModelError(
                    f"cannot identify the local package at {root}; move it aside before installing"
                ) from None
            print(
                f"Installing {args.model}; missing artifacts will be downloaded.",
                flush=True,
            )
            try:
                snapshot = resolve_snapshot(args.model)
            except GgufRepository as found:
                if root.is_symlink():
                    root.unlink()
                install_gguf(found, root)
                return
            ref = _retain_snapshot_ref(snapshot, args.model, root)
            install_snapshot(snapshot, root)
            manifest = validate_package_manifest(root / "manifest.json")
            verify_artifacts(root, manifest, full=False)
            print(f"Installed verified Splash model {args.model} in {root}")
        else:
            print(f"Splash model {args.model} is already installed in {root}")
            try:
                ref = _retain_snapshot_ref(root.resolve(), args.model, root)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                    raise
                print(
                    "Warning: the verified model can be used, but its Hub cache "
                    "reference could not be retained; protect this snapshot from "
                    f"external cache pruning: {error}",
                    file=sys.stderr,
                )
                return
        # Retire this installation's previous pins only after publishing and
        # verifying its new destination. Other installations own other folders.
        try:
            for previous in ref.parent.iterdir():
                if previous != ref and is_hex_digest(previous.name, 40):
                    previous.unlink()
        except OSError as error:
            # Keeping an old pin uses cache space but cannot invalidate the
            # verified installation or its successfully retained current pin.
            print(
                f"Warning: could not retire old Hub cache references: {error}",
                file=sys.stderr,
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install Splash runtime weights")
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument(
        "--model",
        required=True,
        type=parse_repo_id,
        help="Hugging Face repository ID (owner/repo)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify").add_argument("--full", action="store_true")
    checker = commands.add_parser(
        "check", help="say whether pull can install the model; download nothing"
    )
    checker.add_argument(
        "--json", action="store_true", help="one JSON object on stdout"
    )
    return parser.parse_args(argv)


def check(args) -> int:
    """`slipstream pull --check`: 0 and what would be installed, or 1 and why not."""
    try:
        result = {"supported": True, **check_repository(args.model)}
    except ModelError as error:
        result = {"supported": False, "model": args.model, "reason": str(error)}
    if args.json:
        print(json.dumps(result))
    elif result["supported"]:
        if result["kind"] == "package":
            what = "a Slipstream package"
        elif result.get("mtp"):
            what = f"Qwen3.8-Flash-Next GGUF ({result['files']} files; the MTP head from {result['mtp']})"
        else:
            what = f"GGUF ({result['files']} files)"
        print(f"{args.model} can be pulled: {what}, {result['bytes'] / 1e9:.1f} GB.")
    else:
        print(f"{args.model} cannot be served: {result['reason']}", file=sys.stderr)
    return 0 if result["supported"] else 1


def main(argv=None):
    args = parse_args(argv)
    if args.command == "check":
        return check(args)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            selected = verify_installed(
                args.models.resolve(),
                model_id=args.model,
                full=args.full,
            )
            print(
                f"Splash model {selected} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Partial files stay; the next prepare resumes from them.
        print("Download stopped; run it again to continue.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
