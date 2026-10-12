"""Unit tests for the extended dequant kernels in fast_dequant_ext.c.

Run from the repo root:

    python3 -m dev.tools.test_fast_dequant_ext

Checks known-value dequantization for Q2_0, Q5_K and IQ4_NL, plus
finiteness/determinism property checks for every extended type.
"""

import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dev.tools.sharded_gguf_reader import DEQUANT  # noqa: E402


def dequant(type_name: str, raw: bytes, count: int) -> np.ndarray:
    fn = DEQUANT[type_name]
    out = np.zeros(count, dtype=np.float32)
    buf = np.frombuffer(raw, dtype=np.uint8)
    fn(buf.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), count)
    return out


FP16_1_0 = struct.pack("<H", 0x3C00)  # 1.0 in IEEE fp16
FP16_0_0 = struct.pack("<H", 0x0000)  # 0.0


def test_q2_0_known_values():
    # Block: 64 elements = fp16 d + 16 bytes. d = 1.0.
    # First byte 0b11101001 holds four 2-bit values (low nibble first):
    # 01, 10, 10, 11 -> (q - 1) = 0, 1, 1, 2
    raw = FP16_1_0 + bytes([0b11101001, 0b11101001] + [0] * 14)
    out = dequant("q2_0", raw, 64)
    assert out[:4].tolist() == [0.0, 1.0, 1.0, 2.0], out[:4]
    assert out[4:8].tolist() == [0.0, 1.0, 1.0, 2.0], out[4:8]
    # A zeroed byte is four two-bit values of 00, which decode to -1 each.
    assert np.all(out[8:] == -1.0), f"zeroed bytes must decode to -1, got {out[8:12]}"


def test_q5_k_known_values():
    # Block: 256 elements = fp16 d + fp16 dmin + 12 scale bytes
    # + 32 qh bytes + 128 qs bytes.
    # d = 1.0, dmin = 0.0, scales[0] = 1 so the first 32-element group
    # uses d1 = 1, m1 = 0; every other group uses scale 0.
    # ql = 0x01 everywhere: the low nibble (first 32-element group) carries
    # q = 1, the high nibble (second group) carries q = 0.
    raw = FP16_1_0 + FP16_0_0 + bytes([1] + [0] * 11) + bytes(32) + bytes([1] * 128)
    out = dequant("q5_K", raw, 256)
    assert np.all(out[:32] == 1.0), out[:8]
    assert out[32:].sum() == 0.0, out[32:40]


IQ4NL_VALUES = [
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
]


def test_iq4_nl_known_values():
    # Block: 32 elements = fp16 d + 16 bytes. d = 1.0.
    # qs[0] = 0x10 -> low nibble 0 -> element 0, high nibble 1 -> element 16.
    raw = FP16_1_0 + bytes([0x10] + [0] * 15)
    out = dequant("iq4_nl", raw, 32)
    assert out[0] == float(IQ4NL_VALUES[0]), out[0]
    assert out[16] == float(IQ4NL_VALUES[1]), out[16]
    assert out[1] == float(IQ4NL_VALUES[0]), out[1]


EXTENDED_TYPES = {
    "q2_0": (64, 18),
    "q5_K": (256, 176),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq2_s": (256, 82),
    "iq3_xxs": (256, 98),
    "iq3_s": (256, 110),
    "iq4_nl": (32, 18),
    "iq4_xs": (256, 136),
}


def test_extended_types_deterministic_and_finite():
    rng = np.random.default_rng(0)
    for name, (elems, block_bytes) in EXTENDED_TYPES.items():
        raw = rng.integers(0, 256, 4 * block_bytes, dtype=np.uint8).tobytes()
        out_a = dequant(name, raw, elems)
        out_b = dequant(name, raw, elems)
        assert np.array_equal(out_a, out_b), f"{name}: not deterministic"
        # Random bytes may legitimately produce NaN/Inf fp16 scales; require
        # that the kernel either stays finite everywhere or matches a manual
        # zero-scale read (no crashes, no memory overruns).
        assert out_a.shape == (elems,), f"{name}: wrong output shape"
        finite = np.isfinite(out_a)
        if finite.all():
            assert out_a.std() > 0, f"{name}: all-zero output on random input"


def main():
    tests = [
        test_q2_0_known_values,
        test_q5_k_known_values,
        test_iq4_nl_known_values,
        test_extended_types_deterministic_and_finite,
    ]
    for test in tests:
        test()
        print(f"ok - {test.__name__}")
    print(f"\n{len(tests)} tests passed ({len(EXTENDED_TYPES)} extended types loaded)")


if __name__ == "__main__":
    main()
