#!/usr/bin/env python3
"""Writing and reading the packed weight files the engine maps.

The reader in runtime/model/WeightStore.cpp defines this format; everything
here follows it. A file is a sixteen-byte header and then sections, each
starting on a 16 KiB boundary, and it must be consumed exactly - a file with
a byte left over is rejected on load, which is what makes a missing or
mis-sized section fail loudly rather than quietly.

Two layouts carry Q4 weights and they are not interchangeable:

  * A matmul operand is tiled. Within a tile of StorageN output rows the
    order is [group][row][64 codes], which is the addressing the Metal tiles
    use, and its scales and biases follow as [tile][group][row].

  * A lookup table is plain row-major, with its weights, scales and biases in
    three separately aligned sections. Token and n-gram embeddings are
    gathered a row at a time, never multiplied as a tile, so they are stored
    the way they are read.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

ALIGNMENT = 16 * 1024
GROUP = 64
FINE_GROUP = 32
STORAGE_N = 256
EXPERT_STORAGE_N = 128
BF16 = 2


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def q4_bytes(out_size: int, in_size: int, group: int = GROUP) -> int:
    """Half a byte per weight, plus a bf16 scale and bias per group."""
    elements = out_size * in_size
    return elements // 2 + 2 * (elements // group) * BF16


class WeightFile:
    """Mirrors WeightFile: header, aligned sections, and an exact finish."""

    def __init__(self, path: Path, magic: bytes, layer: int, kind: int):
        if len(magic) != 8:
            raise ValueError("a packed file's magic is eight bytes")
        self.path = Path(path)
        self._header = struct.pack("<8sII", magic, layer, kind)
        self._sections: list[bytes] = []

    def section(self, payload: bytes, expected: int | None = None) -> "WeightFile":
        if expected is not None and len(payload) != expected:
            raise ValueError(
                f"section is {len(payload)} bytes where the layout wants {expected}"
            )
        if not payload:
            raise ValueError("a packed section is never empty")
        self._sections.append(payload)
        return self

    def finish(self) -> int:
        out = bytearray(self._header)
        for payload in self._sections:
            out.extend(b"\0" * (align(len(out)) - len(out)))
            out.extend(payload)
        out.extend(b"\0" * (align(len(out)) - len(out)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(out)
        return len(out)


class StreamingWeightFile:
    """The same format, written as it goes.

    A layer is a gibibyte or two and fits in memory comfortably. The
    per-layer embedding does not: its table alone is 29.8 GiB, and the source
    it is quantized from is larger still. This writes each section straight
    out, padding to alignment as it crosses a boundary, so a section can be
    appended in pieces.
    """

    def __init__(self, path: Path, magic: bytes, layer: int, kind: int):
        if len(magic) != 8:
            raise ValueError("a packed file's magic is eight bytes")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")
        self._handle.write(struct.pack("<8sII", magic, layer, kind))
        self._offset = 16
        self._open_section = False

    def begin(self) -> "StreamingWeightFile":
        """Start a section, padding out to its alignment first."""
        padding = align(self._offset) - self._offset
        if padding:
            self._handle.write(b"\0" * padding)
            self._offset += padding
        self._open_section = True
        return self

    def write(self, payload: bytes) -> "StreamingWeightFile":
        if not self._open_section:
            raise ValueError("write outside a section; call begin() first")
        self._handle.write(payload)
        self._offset += len(payload)
        return self

    def section(self, payload: bytes) -> "StreamingWeightFile":
        if not payload:
            raise ValueError("a packed section is never empty")
        return self.begin().write(payload)

    def reserve(self, size: int) -> int:
        """Start a section of size bytes that write_at fills in later; returns its offset.

        Sections built side by side, a piece of each at a time, then need no
        scratch files.
        """
        if size <= 0:
            raise ValueError("a packed section is never empty")
        self.begin()
        offset = self._offset
        self._handle.seek(size, os.SEEK_CUR)
        self._offset += size
        return offset

    def write_at(self, offset: int, payload: bytes) -> "StreamingWeightFile":
        """Write into a reserved section."""
        self._handle.flush()
        os.pwrite(self._handle.fileno(), payload, offset)
        return self

    def finish(self) -> int:
        padding = align(self._offset) - self._offset
        if padding:
            self._handle.write(b"\0" * padding)
            self._offset += padding
        # A reserved section at the end is not written by the handle itself.
        self._handle.truncate(self._offset)
        self._handle.close()
        return self._offset


def read_sections(path: Path, sizes: list[int]):
    """The inverse: pull each section back out at the offset the layout implies."""
    raw = Path(path).read_bytes()
    magic, layer, kind = struct.unpack("<8sII", raw[:16])
    out, offset = [], 16
    for size in sizes:
        offset = align(offset)
        out.append(raw[offset : offset + size])
        offset += size
    if align(offset) != len(raw):
        raise ValueError(f"{Path(path).name}: consumed {align(offset)} of {len(raw)}")
    return magic, layer, kind, out


def tile_q4(codes, scales, biases, storage_n: int = STORAGE_N, group: int = GROUP):
    """Matmul operand: [tile][group][row][group codes], scales after it."""
    out, inp = codes.shape
    if out % storage_n:
        raise ValueError(
            f"output {out} is not a whole number of {storage_n}-wide tiles"
        )
    groups = inp // group
    ordered = (
        codes.reshape(out // storage_n, storage_n, groups, group)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    packed = (ordered[0::2] | (ordered[1::2] << 4)).astype(np.uint8).tobytes()

    def parameters(values):
        return (
            values.reshape(out // storage_n, storage_n, groups)
            .transpose(0, 2, 1)
            .reshape(-1)
            .tobytes()
        )

    return packed + parameters(scales) + parameters(biases)


def plain_q4(codes, scales, biases):
    """Lookup table: row-major, three separately aligned runs."""
    flat = codes.reshape(-1)
    weights = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()
    return weights, scales.tobytes(), biases.tobytes()


def pad_rows(codes, scales, biases, rows: int):
    """Grow a concatenation up to a tile boundary with zeros."""
    have = codes.shape[0]
    if have == rows:
        return codes, scales, biases
    if have > rows:
        raise ValueError(f"{have} rows do not fit in {rows}")
    groups = scales.shape[1]
    return (
        np.vstack([codes, np.zeros((rows - have, codes.shape[1]), np.uint8)]),
        np.vstack([scales, np.zeros((rows - have, groups), np.uint16)]),
        np.vstack([biases, np.zeros((rows - have, groups), np.uint16)]),
    )
