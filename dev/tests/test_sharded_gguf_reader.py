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


def write_single_tensor_gguf(path, name, dims, ggml_type, payload):
    """Minimal GGUF: no KV pairs, one tensor, 32-byte-aligned data."""
    buf = bytearray(b"GGUF") + struct.pack("<IQQ", 3, 1, 0)
    _write_str(buf, name)
    buf += struct.pack("<I", len(dims))
    buf += struct.pack("<%dQ" % len(dims), *dims)
    buf += struct.pack("<IQ", ggml_type, 0)
    # the offset above is relative to the aligned data start
    data_start = (len(buf) + 31) & ~31
    buf += b"\0" * (data_start - len(buf))
    buf += payload
    Path(path).write_bytes(bytes(buf))


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
