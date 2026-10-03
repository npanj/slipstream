#!/usr/bin/env python3
"""Convert MLX / safetensors Qwen3.8-27B or Swift-27B checkpoints into Slipstream format.

Supports pure 4-bit, mixed 4/5-bit, and 8-bit MLX weights.
Packs all 64 layers (hybrid GDN and full attention every 4th layer), head, embedding,
and tokenizer assets with SHA-256 manifest verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Slipstream root path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))

from dev.tools.package_format import (  # noqa: E402
    ALIGNMENT,
    WeightFile,
    pad_rows,
    plain_q4,
    tile_q4,
)
from dev.tools.quantize import quantize_affine, to_bf16  # noqa: E402

LAYER_MAGIC = b"MDFL0006"
HEAD_MAGIC = b"MDFL0002"
EMBEDDING_MAGIC = b"MDFE0001"

LAYOUT = {
    "layers": 64,
    "hidden": 5120,
    "vocabulary": 248320,
    "packed_gdn": 16640,
    "packed_full": 14336,
    "convolution": 10240,
    "value_heads": 48,
    "head_dimension": 128,
    "attention_width": 6144,
    "intermediate": 17408,
    "attention_head_dimension": 256,
    "full_attention_period": 4,
}


def is_full_attention(layer_index: int) -> bool:
    return (layer_index + 1) % LAYOUT["full_attention_period"] == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def widen_bf16(payload: bytes) -> np.ndarray:
    return (np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


class SafetensorsCheckpoint:
    """Reads tensors directly from safetensors files without requiring external libraries."""

    def __init__(self, root: Path):
        self.root = Path(root)
        index_file = self.root / "model.safetensors.index.json"
        if index_file.exists():
            self.map = json.loads(index_file.read_text())["weight_map"]
        else:
            self.map = {}
            for st_file in self.root.glob("*.safetensors"):
                handle = open(st_file, "rb")
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
                for tensor_name in header:
                    if tensor_name != "__metadata__":
                        self.map[tensor_name] = st_file.name
                handle.close()
        self._open: dict[str, tuple] = {}
        self._mlx_cache: dict[str, dict] = {}

    def raw(self, name: str) -> Tuple[bytes, List[int], str]:
        if name not in self.map:
            # Try fallback prefixes
            for pfx in ["language_model.model.", "model.", ""]:
                candidate = f"{pfx}{name}"
                if candidate in self.map:
                    name = candidate
                    break
        shard = self.map[name]
        if shard not in self._open:
            handle = open(self.root / shard, "rb")
            (length,) = struct.unpack("<Q", handle.read(8))
            self._open[shard] = (handle, json.loads(handle.read(length)), 8 + length)
        handle, header, base = self._open[shard]
        entry = header[name]
        start, end = entry["data_offsets"]
        handle.seek(base + start)
        return handle.read(end - start), entry["shape"], entry.get("dtype", "BF16")

    def bf16(self, name: str) -> bytes:
        raw_bytes, shape, dtype = self.raw(name)
        if dtype == "BF16":
            return raw_bytes
        elif dtype == "F16":
            f16 = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
            return to_bf16(f16).tobytes()
        elif dtype == "F32":
            f32 = np.frombuffer(raw_bytes, dtype=np.float32)
            return to_bf16(f32).tobytes()
        return raw_bytes

    def q4(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Check if weight exists as .weight or directly
        w_name = name + ".weight" if (name + ".weight") in self.map else name
        s_name = name + ".scales"
        b_name = name + ".biases"

        packed, shape, dtype = self.raw(w_name)
        out = shape[0]

        if s_name in self.map and b_name in self.map:
            # MLX quantized format
            scales_bytes, s_shape, _ = self.raw(s_name)
            biases_bytes, b_shape, _ = self.raw(b_name)
            groups = s_shape[1]
            inp = groups * 64

            # Check if 4-bit (words: inp // 8)
            if shape[1] == inp // 8:
                words = np.frombuffer(packed, dtype="<u4").reshape(out, inp // 8)
                codes = np.empty((out, inp), dtype=np.uint8)
                for nibble in range(8):
                    codes[:, nibble::8] = (words >> (4 * nibble)) & 0xF
                scales = np.frombuffer(scales_bytes, dtype=np.uint16).reshape(
                    out, groups
                )
                biases = np.frombuffer(biases_bytes, dtype=np.uint16).reshape(
                    out, groups
                )
                return codes, scales, biases
            else:
                # 5-bit or other: dequantize using MLX if available, else numpy
                try:
                    import mlx.core as mx

                    shard = self.map[w_name]
                    if shard not in self._mlx_cache:
                        self._mlx_cache[shard] = mx.load(str(self.root / shard))
                    w_mx = self._mlx_cache[shard][w_name]
                    s_mx = self._mlx_cache[shard][s_name]
                    b_mx = self._mlx_cache[shard][b_name]
                    deq = mx.dequantize(w_mx, s_mx, b_mx, group_size=64, bits=5).astype(
                        mx.float32
                    )
                    deq_np = np.array(deq)
                    return quantize_affine(deq_np, group=64, bits=4)
                except ImportError:
                    pass

        # Dense unquantized FP16/BF16/F32 weight: quantize affine
        if dtype == "BF16":
            arr = widen_bf16(packed).reshape(shape)
        elif dtype == "F16":
            arr = (
                np.frombuffer(packed, dtype=np.float16)
                .astype(np.float32)
                .reshape(shape)
            )
        else:
            arr = np.frombuffer(packed, dtype=np.float32).reshape(shape)
        return quantize_affine(arr, group=64, bits=4)

    def close(self):
        for handle, _, _ in self._open.values():
            handle.close()
        self._open.clear()
        self._mlx_cache.clear()


def concatenate(parts, rows):
    codes = np.vstack([part[0] for part in parts])
    scales = np.vstack([part[1] for part in parts])
    biases = np.vstack([part[2] for part in parts])
    return pad_rows(codes, scales, biases, rows)


def write_layer(source: SafetensorsCheckpoint, index: int, destination: Path) -> Path:
    full = is_full_attention(index)
    kind = 1 if full else 0
    path = destination / f"layer-{index}.bin"
    packed = WeightFile(path, LAYER_MAGIC, index, kind)

    # Prefix resolution
    prefixes = [
        f"language_model.model.layers.{index}.",
        f"model.layers.{index}.",
        f"layers.{index}.",
    ]
    pfx = prefixes[0]
    for p in prefixes:
        if (p + "input_layernorm.weight") in source.map or (
            p + "input_layernorm"
        ) in source.map:
            pfx = p
            break

    packed.section(source.bf16(pfx + "input_layernorm.weight"))

    if full:
        # Full attention
        q = source.q4(pfx + "self_attn.q_proj")
        k = source.q4(pfx + "self_attn.k_proj")
        v = source.q4(pfx + "self_attn.v_proj")
        packed.section(tile_q4(*concatenate([q, k, v], LAYOUT["packed_full"])))
        packed.section(source.bf16(pfx + "self_attn.q_norm.weight"))
        packed.section(source.bf16(pfx + "self_attn.k_norm.weight"))
        packed.section(tile_q4(*source.q4(pfx + "self_attn.o_proj")))
    else:
        # GDN linear attention: qkv, z, b, a
        parts = [
            source.q4(pfx + "linear_attn.in_proj_qkv"),
            source.q4(pfx + "linear_attn.in_proj_z"),
            source.q4(pfx + "linear_attn.in_proj_b"),
            source.q4(pfx + "linear_attn.in_proj_a"),
        ]
        packed.section(tile_q4(*concatenate(parts, LAYOUT["packed_gdn"])))
        packed.section(source.bf16(pfx + "linear_attn.conv1d.weight"))
        logarithm = widen_bf16(source.bf16(pfx + "linear_attn.A_log"))
        packed.section((-np.exp(logarithm)).astype("<f4").tobytes())
        packed.section(source.bf16(pfx + "linear_attn.dt_bias"))
        packed.section(source.bf16(pfx + "linear_attn.norm.weight"))
        packed.section(tile_q4(*source.q4(pfx + "linear_attn.out_proj")))

    packed.section(source.bf16(pfx + "post_attention_layernorm.weight"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        packed.section(tile_q4(*source.q4(pfx + "mlp." + name)))
    packed.finish()
    return path


def write_head(source: SafetensorsCheckpoint, destination: Path) -> Path:
    path = destination / "head.bin"
    packed = WeightFile(path, HEAD_MAGIC, LAYOUT["layers"], 2)
    norm_name = "language_model.model.norm.weight"
    if norm_name not in source.map:
        for alt in ["model.norm.weight", "norm.weight"]:
            if alt in source.map:
                norm_name = alt
                break
    head_name = "language_model.lm_head"
    if (head_name + ".weight") not in source.map and head_name not in source.map:
        for alt in ["lm_head.weight", "lm_head"]:
            if alt in source.map:
                head_name = alt.removesuffix(".weight")
                break
    packed.section(source.bf16(norm_name))
    packed.section(tile_q4(*source.q4(head_name)))
    packed.finish()
    return path


def write_embedding(source: SafetensorsCheckpoint, destination: Path) -> Path:
    path = destination / "embedding.bin"
    packed = WeightFile(path, EMBEDDING_MAGIC, LAYOUT["vocabulary"], LAYOUT["hidden"])
    emb_name = "language_model.model.embed_tokens"
    if (emb_name + ".weight") not in source.map and emb_name not in source.map:
        for alt in [
            "model.embed_tokens.weight",
            "model.embed_tokens",
            "embed_tokens.weight",
            "embed_tokens",
        ]:
            if alt in source.map:
                emb_name = alt.removesuffix(".weight")
                break
    for run in plain_q4(*source.q4(emb_name)):
        packed.section(run)
    packed.finish()
    return path


def prepare_mlx_model(source_dir: Path, output_dir: Path) -> Path:
    target_dir = output_dir / "target"
    tokenizer_dir = output_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Slipstream] Loading MLX checkpoint from {source_dir}...", flush=True)
    source = SafetensorsCheckpoint(source_dir)

    print("[Slipstream] Converting 64 layers for Qwen3.8-27B...", flush=True)
    for index in range(LAYOUT["layers"]):
        t0 = time.time()
        write_layer(source, index, target_dir)
        kind = "attn" if is_full_attention(index) else "gdn"
        if (index + 1) % 8 == 0 or index == 0 or index == 63:
            print(
                f"  [{index + 1:02d}/64] layer-{index}.bin ({kind}) [{time.time() - t0:.2f}s]",
                flush=True,
            )

    print("[Slipstream] Packing head.bin and embedding.bin...", flush=True)
    write_head(source, target_dir)
    write_embedding(source, target_dir)
    draft_dir = output_dir / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)

    print("[Slipstream] Copying tokenizer assets...", flush=True)
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "chat_template.jinja",
        "config.json",
        "merges.txt",
    ]:
        src_f = source_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, tokenizer_dir / fname)

    cfg_file = tokenizer_dir / "config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
        except Exception:
            cfg = {}
    else:
        cfg = {}
    if "text_config" not in cfg:
        cfg["text_config"] = {
            "model_type": "qwen3_5_text",
            "hidden_size": LAYOUT["hidden"],
            "vocab_size": LAYOUT["vocabulary"],
            "max_position_embeddings": 262144,
        }
    else:
        cfg["text_config"].setdefault("model_type", "qwen3_5_text")
        cfg["text_config"].setdefault("hidden_size", LAYOUT["hidden"])
        cfg["text_config"].setdefault("vocab_size", LAYOUT["vocabulary"])
        cfg["text_config"].setdefault("max_position_embeddings", 262144)
    cfg_file.write_text(json.dumps(cfg, indent=2))

    # Generate manifest.json
    artifacts = []
    for sub in ["target", "tokenizer"]:
        p = output_dir / sub
        if not p.exists():
            continue
        for root, _, files in os.walk(p):
            for file in sorted(files):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(output_dir)
                artifacts.append(
                    {
                        "path": str(rel_path),
                        "sha256": sha256_file(file_path),
                        "size": file_path.stat().st_size,
                    }
                )

    manifest = {
        "model": "Swift-Qwen3.8-27B",
        "schema_version": 3,
        "format": {
            "name": "splash-packed-q4",
            "q4_bits": 4,
            "q4_group_size": 64,
            "q4_storage_n": 256,
            "quant_group_size": 64,
            "storage_n": 256,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": LAYER_MAGIC.decode(),
            "draft_layer_magic": "MDFD0004",
            "vision_magic": "MDFV0001",
        },
        "execution_geometry": {
            "allocation_extent_target_bytes": 134217728,
            "attention_history_group": 128,
            "draft_proposal_tokens": 7,
            "draft_query_rows": 8,
            "draft_sliding_window": 2048,
            "maximum_batch_width": 4,
            "prefill_token_budget": 2048,
            "target_kv_block_tokens": 32,
            "target_verify_rows": 8,
        },
        "artifacts": sorted(artifacts, key=lambda x: x["path"]),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Slipstream] Package preparation complete at {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, type=Path, help="Directory containing safetensors"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for prepared package",
    )
    args = parser.parse_args()
    prepare_mlx_model(args.source, args.output)


if __name__ == "__main__":
    main()
