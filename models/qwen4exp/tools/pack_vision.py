#!/usr/bin/env python3
"""Pack a Qwen vision tower from safetensors into Slipstream's vision/model.bin.

Reads the `model.visual.*` bf16 tensors (Swift or base Flash-Next checkpoints
carry all 333 in one shard) and writes them in the exact section order
`loadQwenVisionWeights` (runtime/model/QwenVision.cpp) consumes:

  patch-embed weight+bias, position table, then per block norm1, qkv,
  projection, norm2, up (fc1), down (fc2) weights+biases, then the merger
  norm, up and down projections.

The tower is shared across text models; only the language-space projection
width varies, so `--out-hidden` selects it (2560 for Flash-Next, 5120 for the
27B). The fc1 inner width pads 4304 -> 4352 rows of zeros, matching the
engine's tiled layout. Text-only V3 packages simply omit `vision/`; the
loader and server already treat it as optional.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))

from dev.tools.package_format import ALIGNMENT, WeightFile  # noqa: E402

VISION_MAGIC = b"MDFV0001"

DEPTH = 27
HIDDEN = 1152
PATCH_DIMENSION = 3 * 2 * 16 * 16  # in_channels * temporal * patch^2
INTERMEDIATE = 4304
PADDED_INTERMEDIATE = 4352
MERGED = 4608
POSITION_GRID = 48
BF16 = 2


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def section_sizes(out_hidden: int) -> list[int]:
    """Every section's bytes in loader order (runtime/model/QwenVision.cpp)."""
    sections = [
        HIDDEN * PATCH_DIMENSION * BF16,
        HIDDEN * BF16,
        POSITION_GRID * POSITION_GRID * HIDDEN * BF16,
    ]
    for _ in range(DEPTH):
        sections += [
            HIDDEN * BF16,  # norm1 weight
            HIDDEN * BF16,  # norm1 bias
            3 * HIDDEN * HIDDEN * BF16,  # qkv weight
            3 * HIDDEN * BF16,  # qkv bias
            HIDDEN * HIDDEN * BF16,  # proj weight
            HIDDEN * BF16,  # proj bias
            HIDDEN * BF16,  # norm2 weight
            HIDDEN * BF16,  # norm2 bias
            # NOTE: fc1 rows and fc2 columns both use the padded width.
            PADDED_INTERMEDIATE * HIDDEN * BF16,  # fc1 weight
            PADDED_INTERMEDIATE * BF16,  # fc1 bias
            HIDDEN * PADDED_INTERMEDIATE * BF16,  # fc2 weight
            HIDDEN * BF16,  # fc2 bias
        ]
    sections += [
        HIDDEN * BF16,  # merger norm weight
        HIDDEN * BF16,  # merger norm bias
        MERGED * MERGED * BF16,  # merger fc1 weight
        MERGED * BF16,  # merger fc1 bias
        out_hidden * MERGED * BF16,  # merger fc2 weight
        out_hidden * BF16,  # merger fc2 bias
    ]
    return sections


def expected_file_size(out_hidden: int) -> int:
    """Byte size of a complete vision/model.bin, header included."""
    total = len(VISION_MAGIC) + 4 + 4
    for size in section_sizes(out_hidden):
        total = align(total) + size
    return align(total)


class SafetensorsSource:
    """Minimal reader: header index kept in memory, tensor bytes read on demand."""

    def __init__(self, root: Path):
        self.root = Path(root)
        files = (
            [self.root]
            if self.root.is_file()
            else sorted(self.root.glob("*.safetensors"))
        )
        if not files:
            raise SystemExit(f"error: no safetensors files in {self.root}")
        self._headers: dict[str, tuple[Path, dict]] = {}
        for path in files:
            with open(path, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            base = 8 + length
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                self._headers[name] = (path, entry, base)

    def raw(self, name: str) -> tuple[bytes, list[int], str]:
        try:
            path, entry, base = self._headers[name]
        except KeyError:
            raise SystemExit(f"error: tensor {name} not in source") from None
        start, end = entry["data_offsets"]
        with open(path, "rb") as handle:
            handle.seek(base + start)
            return handle.read(end - start), entry["shape"], entry["dtype"]


def bf16_bytes(source: SafetensorsSource, name: str, shape: list[int]) -> bytes:
    payload, actual_shape, dtype = source.raw(name)
    if dtype != "BF16":
        raise SystemExit(f"error: {name} is {dtype}, expected BF16")
    if list(actual_shape) != shape:
        raise SystemExit(f"error: {name} has shape {actual_shape}, expected {shape}")
    if len(payload) != int(np.prod(shape)) * BF16:
        raise SystemExit(f"error: {name} payload size does not match {shape}")
    return payload


def padded_mlp(source: SafetensorsSource, prefix: str) -> tuple[bytes, bytes, bytes]:
    """The engine tiles the inner width at 4352, so fc1 widens 4304 -> 4352
    rows of zeros and fc2 widens 4304 -> 4352 columns; biases keep their
    unpadded sizes."""
    weight, shape, dtype = source.raw(f"{prefix}.mlp.linear_fc1.weight")
    if dtype != "BF16" or list(shape) != [INTERMEDIATE, HIDDEN]:
        raise SystemExit(f"error: unexpected fc1 {shape} {dtype}")
    rows = np.frombuffer(weight, dtype=np.uint16).reshape(INTERMEDIATE, HIDDEN)
    padded = np.zeros((PADDED_INTERMEDIATE, HIDDEN), dtype=np.uint16)
    padded[:INTERMEDIATE] = rows
    bias, bias_shape, bias_dtype = source.raw(f"{prefix}.mlp.linear_fc1.bias")
    if bias_dtype != "BF16" or list(bias_shape) != [INTERMEDIATE]:
        raise SystemExit("error: unexpected fc1 bias shape/dtype")
    bias_rows = np.frombuffer(bias, dtype=np.uint16)
    padded_bias = np.zeros(PADDED_INTERMEDIATE, dtype=np.uint16)
    padded_bias[:INTERMEDIATE] = bias_rows
    down, down_shape, down_dtype = source.raw(f"{prefix}.mlp.linear_fc2.weight")
    if down_dtype != "BF16" or list(down_shape) != [HIDDEN, INTERMEDIATE]:
        raise SystemExit(f"error: unexpected fc2 {down_shape} {down_dtype}")
    down_rows = np.frombuffer(down, dtype=np.uint16).reshape(HIDDEN, INTERMEDIATE)
    padded_down = np.zeros((HIDDEN, PADDED_INTERMEDIATE), dtype=np.uint16)
    padded_down[:, :INTERMEDIATE] = down_rows
    return padded.tobytes(), padded_bias.tobytes(), padded_down.tobytes()


def pack_vision(source: SafetensorsSource, out_hidden: int, destination: Path) -> Path:
    prefix = "model.visual"
    out = WeightFile(destination / "model.bin", VISION_MAGIC, DEPTH, 0)
    # The conv kernel flattens row-major to PATCH_DIMENSION elements, which is
    # the order the encoder reads.
    patch = np.frombuffer(
        bf16_bytes(
            source, f"{prefix}.patch_embed.proj.weight", [HIDDEN, 3, 2, 16, 16]
        ),
        dtype=np.uint16,
    ).tobytes()
    out.section(patch, HIDDEN * PATCH_DIMENSION * BF16)
    out.section(
        bf16_bytes(source, f"{prefix}.patch_embed.proj.bias", [HIDDEN]),
        HIDDEN * BF16,
    )
    out.section(
        bf16_bytes(
            source,
            f"{prefix}.pos_embed.weight",
            [POSITION_GRID * POSITION_GRID, HIDDEN],
        ),
        POSITION_GRID * POSITION_GRID * HIDDEN * BF16,
    )
    for block in range(DEPTH):
        base = f"{prefix}.blocks.{block}"
        out.section(bf16_bytes(source, f"{base}.norm1.weight", [HIDDEN]))
        out.section(bf16_bytes(source, f"{base}.norm1.bias", [HIDDEN]))
        out.section(
            bf16_bytes(source, f"{base}.attn.qkv.weight", [3 * HIDDEN, HIDDEN])
        )
        out.section(bf16_bytes(source, f"{base}.attn.qkv.bias", [3 * HIDDEN]))
        out.section(
            bf16_bytes(source, f"{base}.attn.proj.weight", [HIDDEN, HIDDEN])
        )
        out.section(bf16_bytes(source, f"{base}.attn.proj.bias", [HIDDEN]))
        out.section(bf16_bytes(source, f"{base}.norm2.weight", [HIDDEN]))
        out.section(bf16_bytes(source, f"{base}.norm2.bias", [HIDDEN]))
        up_weight, up_bias, down_weight = padded_mlp(source, base)
        out.section(up_weight, PADDED_INTERMEDIATE * HIDDEN * BF16)
        out.section(up_bias, PADDED_INTERMEDIATE * BF16)
        out.section(down_weight, HIDDEN * PADDED_INTERMEDIATE * BF16)
        out.section(bf16_bytes(source, f"{base}.mlp.linear_fc2.bias", [HIDDEN]))
    out.section(bf16_bytes(source, f"{prefix}.merger.norm.weight", [HIDDEN]))
    out.section(bf16_bytes(source, f"{prefix}.merger.norm.bias", [HIDDEN]))
    out.section(
        bf16_bytes(
            source, f"{prefix}.merger.linear_fc1.weight", [MERGED, MERGED]
        )
    )
    out.section(bf16_bytes(source, f"{prefix}.merger.linear_fc1.bias", [MERGED]))
    out.section(
        bf16_bytes(
            source, f"{prefix}.merger.linear_fc2.weight", [out_hidden, MERGED]
        )
    )
    out.section(
        bf16_bytes(source, f"{prefix}.merger.linear_fc2.bias", [out_hidden])
    )
    out.finish()
    return destination / "model.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Safetensors file or directory holding model.visual.* tensors",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Vision directory to write model.bin into (package vision/)",
    )
    parser.add_argument(
        "--out-hidden",
        type=int,
        default=2560,
        help="Language-space width: 2560 for Flash-Next, 5120 for the 27B",
    )
    args = parser.parse_args()
    source = SafetensorsSource(args.source)
    visual = [name for name in source._headers if name.startswith("model.visual")]
    if len(visual) != 333:
        raise SystemExit(f"error: found {len(visual)} visual tensors, expected 333")
    packed = pack_vision(source, args.out_hidden, args.output)
    size = packed.stat().st_size
    expected = expected_file_size(args.out_hidden)
    print(f"Wrote {packed} ({size / 2**30:.3f} GiB)")
    if size != expected:
        raise SystemExit(f"error: packed size {size} != expected {expected}")


if __name__ == "__main__":
    main()
