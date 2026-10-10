"""Packing the vision tower: section order, sizes, and padding.

The engine consumes vision/model.bin exactly (runtime/model/QwenVision.cpp):
a 16-byte header, then 333 sections on 16 KiB boundaries, and finish() rejects
a byte left over. These tests pin pack_vision to that contract with synthetic
tensors, for both language widths (2560 Flash-Next, 5120 27B), without
needing the multi-gigabyte upstream checkpoint.
"""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "pack_vision", REPO_ROOT / "models" / "qwen4exp" / "tools" / "pack_vision.py"
)
pack_vision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pack_vision)

_gspec = importlib.util.spec_from_file_location(
    "convert_qwen4exp_gguf",
    REPO_ROOT / "models" / "qwen4exp" / "tools" / "convert_qwen4exp_gguf.py",
)
gguf_converter = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(gguf_converter)


class FakeSource:
    """Deterministic bf16 tensors in the Swift checkpoint's shapes."""

    SHAPES = {}
    _P = "model.visual"
    SHAPES[f"{_P}.patch_embed.proj.weight"] = [1152, 3, 2, 16, 16]
    SHAPES[f"{_P}.patch_embed.proj.bias"] = [1152]
    SHAPES[f"{_P}.pos_embed.weight"] = [48 * 48, 1152]
    for _block in range(27):
        _b = f"{_P}.blocks.{_block}"
        SHAPES.update(
            {
                f"{_b}.norm1.weight": [1152],
                f"{_b}.norm1.bias": [1152],
                f"{_b}.attn.qkv.weight": [3456, 1152],
                f"{_b}.attn.qkv.bias": [3456],
                f"{_b}.attn.proj.weight": [1152, 1152],
                f"{_b}.attn.proj.bias": [1152],
                f"{_b}.norm2.weight": [1152],
                f"{_b}.norm2.bias": [1152],
                f"{_b}.mlp.linear_fc1.weight": [4304, 1152],
                f"{_b}.mlp.linear_fc1.bias": [4304],
                f"{_b}.mlp.linear_fc2.weight": [1152, 4304],
                f"{_b}.mlp.linear_fc2.bias": [1152],
            }
        )
    SHAPES.update(
        {
            f"{_P}.merger.norm.weight": [1152],
            f"{_P}.merger.norm.bias": [1152],
            f"{_P}.merger.linear_fc1.weight": [4608, 4608],
            f"{_P}.merger.linear_fc1.bias": [4608],
            f"{_P}.merger.linear_fc2.weight": [2560, 4608],
            f"{_P}.merger.linear_fc2.bias": [2560],
        }
    )

    def raw(self, name):
        shape = self.SHAPES[name]
        count = int(np.prod(shape))
        payload = (np.arange(count, dtype=np.uint32) * 2654435761 % 65536).astype(
            np.uint16
        )
        return payload.tobytes(), shape, "BF16"


def walk(data):
    sizes = pack_vision.section_sizes(2560)
    assert len(sizes) == 333
    offset, blobs = 16, []
    for size in sizes:
        start = (offset + 16383) & ~16383
        blobs.append(data[start : start + size])
        offset = start + size
    assert offset <= len(data) and len(data) - ((offset + 16383) & ~16383) == 0
    return blobs


class PackVisionTest(unittest.TestCase):
    def test_size_formula_matches_production_27b_file(self):
        # The shipped 27B vision/model.bin is 930,250,752 bytes and loads in
        # production, so the size walk must reproduce it exactly for the
        # 5120-wide tower.
        self.assertEqual(pack_vision.expected_file_size(5120), 930250752)
        self.assertEqual(len(pack_vision.section_sizes(2560)), 333)

    def test_pack_order_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pack_vision.pack_vision(FakeSource(), 2560, out)
            data = (out / "model.bin").read_bytes()
        self.assertEqual(len(data), pack_vision.expected_file_size(2560))
        magic, layer, kind = struct.unpack("<8sII", data[:16])
        self.assertEqual((magic, layer, kind), (b"MDFV0001", 27, 0))
        blobs = walk(data)
        source = FakeSource()
        expect = []
        prefix = "model.visual"
        expect.append(source.raw(f"{prefix}.patch_embed.proj.weight")[0])
        expect.append(source.raw(f"{prefix}.patch_embed.proj.bias")[0])
        expect.append(source.raw(f"{prefix}.pos_embed.weight")[0])
        for block in range(27):
            base = f"{prefix}.blocks.{block}"
            for tail in (
                "norm1.weight",
                "norm1.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "norm2.weight",
                "norm2.bias",
            ):
                expect.append(source.raw(f"{base}.{tail}")[0])
            rows = np.frombuffer(
                source.raw(f"{base}.mlp.linear_fc1.weight")[0], np.uint16
            ).reshape(4304, 1152)
            padded = np.zeros((4352, 1152), np.uint16)
            padded[:4304] = rows
            expect.append(padded.tobytes())
            bias = np.frombuffer(
                source.raw(f"{base}.mlp.linear_fc1.bias")[0], np.uint16
            )
            padded_bias = np.zeros(4352, np.uint16)
            padded_bias[:4304] = bias
            expect.append(padded_bias.tobytes())
            down = np.frombuffer(
                source.raw(f"{base}.mlp.linear_fc2.weight")[0], np.uint16
            ).reshape(1152, 4304)
            padded_down = np.zeros((1152, 4352), np.uint16)
            padded_down[:, :4304] = down
            expect.append(padded_down.tobytes())
            expect.append(source.raw(f"{base}.mlp.linear_fc2.bias")[0])
        for tail in (
            "merger.norm.weight",
            "merger.norm.bias",
            "merger.linear_fc1.weight",
            "merger.linear_fc1.bias",
            "merger.linear_fc2.weight",
            "merger.linear_fc2.bias",
        ):
            expect.append(source.raw(f"{prefix}.{tail}")[0])
        self.assertEqual(len(expect), 333)
        for index, (got, want) in enumerate(zip(blobs, expect)):
            self.assertEqual(got, want, f"section {index}")

    def test_attach_vision_bin_links_and_keeps(self):
        tower = struct.pack("<8sII", b"MDFV0001", 27, 0) + b"\0" * 100
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "tower.bin"
            src.write_bytes(tower)
            package = root / "prepared"
            gguf_converter.attach_vision_bin(src, package)
            dst = package / "vision" / "model.bin"
            self.assertEqual(dst.read_bytes(), tower)
            self.assertEqual(src.stat().st_ino, dst.stat().st_ino)
            # Re-running prepare never clobbers a present tower (replace
            # the link with a standalone file first: writing through the
            # hardlink would rewrite the source too).
            dst.unlink()
            dst.write_bytes(b"keep me")
            gguf_converter.attach_vision_bin(src, package)
            self.assertEqual(dst.read_bytes(), b"keep me")

    def test_attach_vision_bin_rejects_non_tower(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "layer-0.bin"
            bad.write_bytes(struct.pack("<8sII", b"MDFN0031", 0, 0))
            with self.assertRaisesRegex(SystemExit, "not a packed vision tower"):
                gguf_converter.attach_vision_bin(bad, root / "prepared")

    def test_pack_5120_wide_tower(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertRaisesRegex(
                SystemExit,
                "merger.linear_fc2.weight",
                pack_vision.pack_vision,
                FakeSource(),
                5120,
                out,
            )


if __name__ == "__main__":
    unittest.main()
