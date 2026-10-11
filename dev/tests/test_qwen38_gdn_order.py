"""GDN value-head order for GGUF-prepared Qwen3.8 packages.

Split GGUF exports (``attn_qkv`` / ``attn_gate`` / ``ssm_beta`` /
``ssm_alpha``) store the 48 value heads grouped as three blocks of sixteen.
The engine (and Swift's fused weights) interleave them round-robin: engine
head ``h`` is GGUF head ``(h % 3) * 16 + h // 3``. Preparing a package without
this reorder serves a model whose trunk diverges at prefill position 0 while
every tensor still matches its source at ~0.996 correlation, so the mapping
below is pinned against the measured cross-check, not derived from it.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen38.tools.convert_qwen38_gguf import (
    GDN_HEAD_ORDER,
    GDN_VALUE_HEADS,
    interleave_gdn_columns,
    interleave_gdn_heads,
)


def head_table(head_dim, cols):
    """One constant value per head block, so order is unambiguous."""
    return (
        np.arange(GDN_VALUE_HEADS, dtype=np.float32)[:, None, None]
        * np.ones((GDN_VALUE_HEADS, head_dim, cols), dtype=np.float32)
    ).reshape(GDN_VALUE_HEADS * head_dim, cols)


class GdnHeadOrderTest(unittest.TestCase):
    def test_order_is_the_measured_permutation(self):
        self.assertEqual(GDN_VALUE_HEADS, 48)
        self.assertEqual(
            list(GDN_HEAD_ORDER), [(h % 3) * 16 + h // 3 for h in range(48)]
        )
        self.assertEqual(sorted(GDN_HEAD_ORDER), list(range(48)))

    def test_rows_land_in_engine_order(self):
        engine = interleave_gdn_heads(head_table(128, 4))
        grouped = head_table(128, 4)
        for engine_head in range(GDN_VALUE_HEADS):
            gguf_head = GDN_HEAD_ORDER[engine_head]
            self.assertTrue(
                np.array_equal(
                    engine[engine_head * 128 : (engine_head + 1) * 128],
                    grouped[gguf_head * 128 : (gguf_head + 1) * 128],
                )
            )

    def test_columns_land_in_engine_order(self):
        table = np.repeat(head_table(128, 1), 3, axis=0)
        engine = interleave_gdn_columns(table)
        for engine_head in range(GDN_VALUE_HEADS):
            gguf_head = GDN_HEAD_ORDER[engine_head]
            self.assertTrue(
                np.array_equal(
                    engine[:, engine_head * 128 : (engine_head + 1) * 128],
                    table[:, gguf_head * 128 : (gguf_head + 1) * 128],
                )
            )

    def test_per_head_vectors_pass_through_in_order(self):
        vec = np.arange(GDN_VALUE_HEADS, dtype=np.float32)
        reordered = interleave_gdn_heads(vec, head_dim=1)
        for engine_head in range(GDN_VALUE_HEADS):
            self.assertEqual(
                reordered[engine_head], vec[GDN_HEAD_ORDER[engine_head]]
            )


if __name__ == "__main__":
    unittest.main()
