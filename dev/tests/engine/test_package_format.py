"""The packed layouts must survive a round trip and reject a bad section."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.package_format import (  # noqa: E402
    ALIGNMENT,
    EXPERT_STORAGE_N,
    StreamingWeightFile,
    WeightFile,
    align,
    pad_rows,
    plain_q4,
    q4_bytes,
    read_sections,
    tile_q4,
)


class PackageFormatTests(unittest.TestCase):
    def test_sections_land_on_alignment_and_survive_a_round_trip(self):
        payloads = [b"a" * 100, b"b" * (ALIGNMENT + 7), b"c" * 3]
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "layer-0.bin"
            written = WeightFile(path, b"MDFL0006", 5, 1)
            for payload in payloads:
                written.section(payload)
            total = written.finish()
            self.assertEqual(total % ALIGNMENT, 0)

            magic, layer, kind, back = read_sections(path, [len(p) for p in payloads])
            self.assertEqual((magic, layer, kind), (b"MDFL0006", 5, 1))
            self.assertEqual(back, payloads)

    def test_a_mis_sized_section_is_refused_at_write_time(self):
        with tempfile.TemporaryDirectory() as scratch:
            written = WeightFile(Path(scratch) / "x.bin", b"MDFL0006", 0, 0)
            with self.assertRaises(ValueError):
                written.section(b"short", expected=64)
            with self.assertRaises(ValueError):
                written.section(b"")

    def test_a_truncated_file_is_refused_at_read_time(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "layer-0.bin"
            WeightFile(path, b"MDFL0006", 0, 0).section(b"z" * 64).finish()
            # Claiming a larger section leaves the file with bytes unconsumed,
            # which is exactly what the engine rejects on load.
            with self.assertRaises(ValueError):
                read_sections(path, [64, 64])

    def test_q4_bytes_matches_what_tiling_produces(self):
        rng = np.random.default_rng(3)
        out, inp = 512, 128
        codes = rng.integers(0, 16, size=(out, inp), dtype=np.uint8)
        scales = rng.integers(0, 65535, size=(out, inp // 64), dtype=np.uint16)
        biases = rng.integers(0, 65535, size=(out, inp // 64), dtype=np.uint16)
        self.assertEqual(len(tile_q4(codes, scales, biases)), q4_bytes(out, inp))

    def test_tiling_orders_by_group_then_row(self):
        # Two tiles of 128, one group of 64: row r of tile t must land at
        # (t * 128 + r) * 32 bytes, which is what the kernel addresses.
        out, inp = 256, 64
        codes = np.zeros((out, inp), dtype=np.uint8)
        codes[129, :] = 0xF
        scales = np.zeros((out, 1), dtype=np.uint16)
        biases = np.zeros((out, 1), dtype=np.uint16)
        packed = tile_q4(codes, scales, biases, storage_n=EXPERT_STORAGE_N)
        marked = np.frombuffer(packed[: out * inp // 2], dtype=np.uint8)
        hot = np.nonzero(marked)[0]
        self.assertEqual(hot.min() // 32, 129)
        self.assertEqual(hot.max() // 32, 129)

    def test_a_lookup_table_stays_row_major(self):
        codes = np.arange(8, dtype=np.uint8).reshape(2, 4) % 16
        scales = np.zeros((2, 1), dtype=np.uint16)
        biases = np.zeros((2, 1), dtype=np.uint16)
        weights, _, _ = plain_q4(codes, scales, biases)
        self.assertEqual(list(weights), [0x10, 0x32, 0x54, 0x76])

    def test_padding_grows_to_a_tile_and_refuses_to_shrink(self):
        codes = np.ones((100, 64), dtype=np.uint8)
        scales = np.ones((100, 1), dtype=np.uint16)
        biases = np.ones((100, 1), dtype=np.uint16)
        grown = pad_rows(codes, scales, biases, 128)
        self.assertEqual(grown[0].shape, (128, 64))
        self.assertTrue(np.all(grown[0][100:] == 0))
        with self.assertRaises(ValueError):
            pad_rows(codes, scales, biases, 64)

    def test_streaming_matches_writing_it_all_at_once(self):
        payloads = [b"a" * 100, b"b" * (ALIGNMENT + 7), b"c" * 3]
        with tempfile.TemporaryDirectory() as scratch:
            whole = Path(scratch) / "whole.bin"
            written = WeightFile(whole, b"MDFN0004", 2, 3)
            for payload in payloads:
                written.section(payload)
            written.finish()

            # The large section arrives in pieces, as the table does.
            streamed = Path(scratch) / "streamed.bin"
            out = StreamingWeightFile(streamed, b"MDFN0004", 2, 3)
            out.section(payloads[0])
            out.begin()
            for start in range(0, len(payloads[1]), 512):
                out.write(payloads[1][start : start + 512])
            out.section(payloads[2])
            out.finish()

            self.assertEqual(whole.read_bytes(), streamed.read_bytes())

    def test_reserved_sections_filled_side_by_side_match_writing_it_all_at_once(self):
        payloads = [b"a" * (ALIGNMENT + 7), b"b" * 300, b"c" * 300, b"d" * 5]
        with tempfile.TemporaryDirectory() as scratch:
            whole = Path(scratch) / "whole.bin"
            written = WeightFile(whole, b"MDFN0004", 2, 3)
            for payload in payloads:
                written.section(payload)
            written.finish()

            # Codes, scales and biases arrive a chunk of each at a time, as the table's do.
            reserved = Path(scratch) / "reserved.bin"
            out = StreamingWeightFile(reserved, b"MDFN0004", 2, 3)
            cursors = [out.reserve(len(payload)) for payload in payloads[:3]]
            for start in range(0, len(payloads[0]), 100):
                for index, payload in enumerate(payloads[:3]):
                    piece = payload[start : start + 100]
                    if piece:
                        out.write_at(cursors[index], piece)
                        cursors[index] += len(piece)
            out.section(payloads[3])
            out.finish()

            self.assertEqual(whole.read_bytes(), reserved.read_bytes())

    def test_a_reserved_section_at_the_end_still_sizes_the_file(self):
        with tempfile.TemporaryDirectory() as scratch:
            whole = Path(scratch) / "whole.bin"
            written = WeightFile(whole, b"MDFN0004", 0, 0)
            written.section(b"\0" * 300)
            written.finish()
            reserved = Path(scratch) / "reserved.bin"
            out = StreamingWeightFile(reserved, b"MDFN0004", 0, 0)
            out.reserve(300)
            out.finish()
            self.assertEqual(whole.read_bytes(), reserved.read_bytes())

    def test_streaming_refuses_a_write_outside_a_section(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = StreamingWeightFile(Path(scratch) / "x.bin", b"MDFN0004", 0, 0)
            with self.assertRaises(ValueError):
                out.write(b"loose")

    def test_align_is_idempotent(self):
        self.assertEqual(align(0), 0)
        self.assertEqual(align(1), ALIGNMENT)
        self.assertEqual(align(ALIGNMENT), ALIGNMENT)


if __name__ == "__main__":
    unittest.main()
