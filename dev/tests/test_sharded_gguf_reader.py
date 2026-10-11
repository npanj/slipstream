"""Q5_K (ggml type 13) support in the sharded GGUF reader.

UD-Q4_K_M and Q4_K_M files quantize some projections (e.g. ffn_down) with
Q5_K, which the reader used to refuse with "Unsupported ggml type 13". The
kernel below is pinned against libggml-base's dequantize_row_q5_K (bit-exact
on a full 89M-element tensor), but the committed test only needs a hand-built
block so it runs in milliseconds with no model download.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.tools.sharded_gguf_reader import MultiShardGgufReader


def _write_str(buf, text):
    raw = text.encode()
    buf += struct.pack("<Q", len(raw)) + raw


def write_gguf(path, tensors):
    """Minimal GGUF: no KV pairs, 32-byte-aligned data section.

    tensors: list of (name, dims, ggml_type, payload) tuples.
    """
    buf = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), 0)
    running = 0
    for name, dims, ggml_type, payload in tensors:
        _write_str(buf, name)
        buf += struct.pack("<I", len(dims))
        buf += struct.pack("<%dQ" % len(dims), *dims)
        buf += struct.pack("<IQ", ggml_type, running)
        running += len(payload)
    data_start = (len(buf) + 31) & ~31
    buf += b"\0" * (data_start - len(buf))
    for _, _, _, payload in tensors:
        buf += payload
    Path(path).write_bytes(bytes(buf))


def write_single_tensor_gguf(path, name, dims, ggml_type, payload):
    write_gguf(path, [(name, dims, ggml_type, payload)])


def read_one(path, name):
    reader = MultiShardGgufReader(path)
    try:
        return reader.read_tensor(name)
    finally:
        reader.close()


class Q5KReaderTest(unittest.TestCase):
    def test_q5_k_block_decodes(self):
        # d = 1.0, dmin = 0.0; scales give (sc=1, m=0) then (sc=2, m=1).
        block = bytearray(176)
        struct.pack_into("<HHHH", block, 0, 0x3C00, 0x0000, 0, 0)
        block[4] = 1
        block[5] = 2
        block[9] = 1
        # qh all zero: every high bit is 0.
        # qs byte 0: low nibble 3, high nibble 5.
        block[48] = 0x53
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "q5k.gguf")
            write_single_tensor_gguf(
                path, "blk.0.w", (256,), 13, bytes(block)
            )
            reader = MultiShardGgufReader(path)
            try:
                got = reader.read_tensor("blk.0.w")
            finally:
                reader.close()
        self.assertEqual(got.shape, (256,))
        self.assertEqual(got.dtype, np.float32)
        # First 32 weights use pair 0 (1 * low - 0); next 32 use pair 1.
        self.assertAlmostEqual(float(got[0]), 3.0)
        self.assertAlmostEqual(float(got[32]), 2 * 5 - 1 * 0)
        self.assertAlmostEqual(float(got[1]), 0.0)
        self.assertTrue(bool(np.isfinite(got).all()))

    def test_iq4_nl_grid_decodes(self):
        # d = 1.0; low nibble 0 -> -127, high nibble 15 -> +113.
        block = bytearray(18)
        struct.pack_into("<H", block, 0, 0x3C00)
        block[2] = 0xF0
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nl.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (32,), 20, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertEqual(got.shape, (32,))
        self.assertAlmostEqual(float(got[0]), -127.0)
        self.assertAlmostEqual(float(got[16]), 113.0)
        self.assertTrue(bool(np.isfinite(got).all()))

    def test_iq4_xs_scales_decode(self):
        # d = 1.0; scales give ls = 33 for the first 32-group (dl = 1.0):
        # scales_l[0] low nibble 1, scales_h bits 0-1 = 2.
        block = bytearray(136)
        struct.pack_into("<H", block, 0, 0x3C00)
        struct.pack_into("<H", block, 2, 2)
        block[4] = 0x01
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "xs.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (256,), 23, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertEqual(got.shape, (256,))
        # Codes are 0 -> grid -127; everything else in group 0 uses ls=33.
        self.assertAlmostEqual(float(got[0]), -127.0)
        self.assertTrue(bool(np.isfinite(got).all()))

    def test_q3_k_decodes(self):
        # d = 1.0, zero scales -> every scale is -32; qs = 0, hm = 0 gives
        # q = 0 -> dl * (0 - 4) = (-32) * (-4) = 128.
        block = bytearray(110)
        struct.pack_into("<H", block, 108, 0x3C00)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "q3.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (256,), 11, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertEqual(got.shape, (256,))
        self.assertAlmostEqual(float(got[0]), 128.0)
        self.assertAlmostEqual(float(got[255]), 128.0)
        self.assertTrue(bool(np.isfinite(got).all()))

    def test_q3_k_high_bits_decode(self):
        # Same, but hmask all set -> subtract 0, q = 0 -> 0.0.
        block = bytearray(110)
        struct.pack_into("<H", block, 108, 0x3C00)
        for i in range(32):
            block[i] = 0xFF
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "q3h.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (256,), 11, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertAlmostEqual(float(got[0]), 0.0)
        self.assertAlmostEqual(float(got[255]), 0.0)

    def test_iq3_s_decodes(self):
        # d = 1.0, zero scales -> db = 1.0; qs/qh zero -> grid row 0
        # (0x01010101) -> values [1,1,1,1]; signs zero -> positive.
        block = bytearray(110)
        struct.pack_into("<H", block, 0, 0x3C00)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "iq3.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (256,), 21, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertEqual(got.shape, (256,))
        self.assertAlmostEqual(float(got[0]), 1.0)
        self.assertAlmostEqual(float(got[255]), 1.0)
        self.assertTrue(bool(np.isfinite(got).all()))

    def test_iq3_s_signs_decode(self):
        # All signs set -> every value negated.
        block = bytearray(110)
        struct.pack_into("<H", block, 0, 0x3C00)
        for i in range(74, 106):
            block[i] = 0xFF
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "iq3s.gguf")
            write_single_tensor_gguf(path, "blk.0.w", (256,), 21, bytes(block))
            got = read_one(path, "blk.0.w")
        self.assertAlmostEqual(float(got[0]), -1.0)
        self.assertAlmostEqual(float(got[255]), -1.0)

    def test_q5_k_high_bits_decode(self):
        # Same scales, but qh byte 0 = 0x03: bit0 -> low group of 64-block 0,
        # bit1 -> high group. qs byte 0 low nibble 0, high nibble 0.
        block = bytearray(176)
        struct.pack_into("<HHHH", block, 0, 0x3C00, 0x0000, 0, 0)
        block[4] = 1
        block[5] = 2
        block[9] = 1
        block[16] = 0x03
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "q5k.gguf")
            write_single_tensor_gguf(
                path, "blk.0.w", (256,), 13, bytes(block)
            )
            reader = MultiShardGgufReader(path)
            try:
                got = reader.read_tensor("blk.0.w")
            finally:
                reader.close()
        # q5 = 16 in both groups: 1*16-0 and 2*16-0.
        self.assertAlmostEqual(float(got[0]), 16.0)
        self.assertAlmostEqual(float(got[32]), 32.0)


if __name__ == "__main__":
    unittest.main()
