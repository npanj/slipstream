"""Preparing a GGUF model in place: its source is freed as parts are written, and resumes."""

import concurrent.futures
import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools.sharded_gguf_reader import MultiShardGgufReader
from models.qwen4exp.tools import convert_qwen4exp_gguf as converter

# Three file-system blocks per tensor, so freeing one always frees whole blocks.
ELEMENTS = 3 * 4096 // 4


def write_gguf(path: Path, names: list[str]) -> None:
    """A GGUF of float32 vectors, the n-th filled with n."""
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(names), 0)
    for index, name in enumerate(names):
        encoded = name.encode()
        header += struct.pack("<Q", len(encoded)) + encoded
        header += struct.pack("<IQIQ", 1, ELEMENTS, 0, index * ELEMENTS * 4)
    header += b"\0" * (-len(header) % 32)
    with path.open("wb") as out:
        out.write(header)
        for index in range(len(names)):
            out.write(struct.pack(f"<{ELEMENTS}f", *[float(index)] * ELEMENTS))


def fake_part(model_dir, *arguments):
    """Stands in for a converter: reads the tensors its part owns, writes it, says what it read."""
    out = Path(next(a for a in arguments if str(a).endswith(".bin")))
    reader = MultiShardGgufReader(model_dir)
    total = sum(
        float(reader.read_tensor(name)[0])
        for name in reader.tensors
        if converter.tensor_owner(name) == out.name
    )
    converter._partial(out).write_text(str(total))
    converter._commit(out)
    reader.close()
    return f"{out.name} finished", reader.read_names


class InPlacePreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        layers = [f"blk.{i}.attn_qkv.weight" for i in range(converter.LAYOUT["layers"])]
        write_gguf(
            self.root / "Model-00001-of-00002.gguf", layers[:30] + ["token_embd.weight"]
        )
        write_gguf(
            self.root / "Model-00002-of-00002.gguf",
            layers[30:]
            + ["output.weight", "per_layer_token_embd.weight", "blk.1.ple_key.weight"],
        )
        self.shards = sorted(self.root.glob("*.gguf"))
        self.calls = []
        self.failing = set()

        def part(function_name):
            def run(model_dir, *arguments):
                name = Path(next(a for a in arguments if str(a).endswith(".bin"))).name
                self.calls.append(name)
                if name in self.failing:
                    raise RuntimeError(f"{name} crashed")
                return fake_part(model_dir, *arguments)

            run.__name__ = function_name
            return run

        for function in (
            "convert_single_layer",
            "convert_head",
            "convert_embedding",
            "convert_ngram",
        ):
            self.enterContext(mock.patch.object(converter, function, part(function)))
        # Threads instead of processes, so the fakes above are what runs.
        self.enterContext(
            mock.patch.object(
                converter.concurrent.futures,
                "ProcessPoolExecutor",
                concurrent.futures.ThreadPoolExecutor,
            )
        )
        self.enterContext(mock.patch.object(converter, "fetch_tokenizer"))
        self.enterContext(mock.patch.object(converter, "write_placeholder_draft"))
        self.enterContext(
            mock.patch.object(
                converter,
                "write_manifest",
                lambda output: (Path(output) / "manifest.json").write_text("{}"),
            )
        )

    def prepare(self, consume_source=True):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            converter.prepare_gguf_model(
                self.root,
                self.root / "prepared",
                workers=2,
                consume_source=consume_source,
            )
        return output.getvalue()

    def journal(self):
        lines = (
            (self.root / "prepared" / converter.SourceJournal.NAME)
            .read_text()
            .splitlines()
        )
        return [json.loads(line) for line in lines]

    def test_every_tensor_of_the_model_has_one_part(self):
        self.assertEqual(converter.tensor_owner("blk.0.attn_qkv.weight"), "layer-0.bin")
        self.assertEqual(converter.tensor_owner("blk.1.ple_key.weight"), "ngram.bin")
        self.assertEqual(
            converter.tensor_owner("blk.1.hc_attn_norm.weight"), "layer-1.bin"
        )
        self.assertEqual(
            converter.tensor_owner("blk.48.nextn.enorm.weight"), "mtp-combiner.bin"
        )
        self.assertEqual(
            converter.tensor_owner("blk.48.attn_q.weight"), "mtp-layer.bin"
        )
        self.assertEqual(converter.tensor_owner("output_hc_down.weight"), "head.bin")
        self.assertEqual(converter.tensor_owner("token_embd.weight"), "embedding.bin")
        self.assertEqual(
            converter.tensor_owner("per_layer_token_embd.weight"), "ngram.bin"
        )
        self.assertIsNone(converter.tensor_owner("rope_freqs.weight"))

    def test_kept_source_stays_whole(self):
        before = [shard.read_bytes() for shard in self.shards]
        self.prepare(consume_source=False)
        self.assertEqual([shard.read_bytes() for shard in self.shards], before)
        target = self.root / "prepared/target"
        self.assertEqual((target / "layer-31.bin").read_text(), "1.0")
        self.assertTrue((self.root / "prepared/manifest.json").is_file())
        self.assertFalse(
            (self.root / "prepared" / converter.SourceJournal.NAME).exists()
        )
        self.assertEqual(list(target.glob("*" + converter.PARTIAL_SUFFIX)), [])

    def test_consumed_source_is_freed_part_by_part_then_deleted(self):
        if not converter.can_free_in_place(self.root):
            self.skipTest("this file system cannot free parts of a file")
        allocated = sum(shard.stat().st_blocks for shard in self.shards)
        self.failing = {"head.bin"}
        with self.assertRaisesRegex(RuntimeError, "head.bin crashed"):
            self.prepare()
        freed = {line["freed"] for line in self.journal()[1:]}
        self.assertNotIn("output.weight", freed)
        for name in freed:
            self.assertTrue(
                (self.root / "prepared/target" / converter.tensor_owner(name)).is_file()
            )
        # What is left reads back as it was; what was freed reads as zeros.
        reader = MultiShardGgufReader(self.root)
        self.assertEqual(float(reader.read_tensor("output.weight")[0]), 18.0)
        if "blk.5.attn_qkv.weight" in freed:
            # Whole blocks inside the tensor go; its edges share blocks with neighbours.
            self.assertEqual(
                float(reader.read_tensor("blk.5.attn_qkv.weight")[ELEMENTS // 2]), 0.0
            )
        reader.close()

        # By the time the package is complete, the shards hold next to nothing.
        left = []
        manifest = converter.write_manifest

        def measure(output):
            left.append(sum(shard.stat().st_blocks for shard in self.shards))
            manifest(output)

        self.failing, self.calls = set(), []
        with mock.patch.object(converter, "write_manifest", measure):
            output = self.prepare()
        self.assertEqual(self.calls, ["head.bin"])
        self.assertEqual(output.count("[DONE]"), 51)
        # Only the headers and the blocks two tensors share stay, one per tensor at most.
        self.assertLessEqual(left[0] * 512, (53 + len(self.shards)) * 4096)
        self.assertLess(left[0], allocated / 2)
        self.assertGreater(allocated * 512, 20 * ELEMENTS * 4)
        self.assertEqual(list(self.root.glob("*.gguf")), [])
        self.assertTrue((self.root / "prepared/manifest.json").is_file())
        self.assertEqual((self.root / "prepared/target/head.bin").read_text(), "18.0")
        self.assertFalse(
            (self.root / "prepared" / converter.SourceJournal.NAME).exists()
        )

    def test_a_freed_tensor_whose_part_was_lost_stops_the_resume(self):
        if not converter.can_free_in_place(self.root):
            self.skipTest("this file system cannot free parts of a file")
        self.failing = {"head.bin"}
        with self.assertRaises(RuntimeError):
            self.prepare()
        (self.root / "prepared/target/layer-5.bin").unlink()
        with self.assertRaisesRegex(SystemExit, "layer-5.bin still needs"):
            self.prepare()
        self.assertEqual(sorted(self.root.glob("*.gguf")), self.shards)

    def test_new_downloads_start_afresh(self):
        if not converter.can_free_in_place(self.root):
            self.skipTest("this file system cannot free parts of a file")
        self.failing = {"head.bin"}
        with self.assertRaises(RuntimeError):
            self.prepare()
        (self.root / "prepared/target/layer-5.bin").unlink()
        # Downloaded again: same names and sizes, but new files.
        for shard in self.shards:
            data = shard.read_bytes()
            shard.unlink()
            shard.write_bytes(data)
        self.failing, self.calls = set(), []
        self.prepare(consume_source=False)
        # Everything is converted again (ngram.bin, as before, is reused when present).
        self.assertLessEqual(
            {f"layer-{i}.bin" for i in range(48)} | {"head.bin"}, set(self.calls)
        )

    def test_too_little_room_fails_before_converting(self):
        with (
            mock.patch.object(
                converter.shutil, "disk_usage", return_value=mock.Mock(free=0)
            ),
            self.assertRaisesRegex(SystemExit, "free disk space"),
        ):
            self.prepare(consume_source=False)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
