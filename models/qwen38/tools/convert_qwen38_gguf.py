#!/usr/bin/env python3
"""Convert Qwen3.8-27B or Swift-27B sharded GGUF models directly into Slipstream format.

Converts all 64 layers (hybrid GDN and full attention every 4th layer), head, embedding,
and tokenizer assets from GGUF weights (Q4_0, Q8_0, Q4_K, etc.) using SIMD dequantization.
Supports low-memory streaming and in-place conversion with --consume-source (F_PUNCHHOLE).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Slipstream path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))

from dev.tools.quantize import to_bf16, from_bf16, quantize_affine
from dev.tools.package_format import (
    ALIGNMENT, BF16, GROUP, STORAGE_N,
    StreamingWeightFile, WeightFile, pad_rows, plain_q4, tile_q4, q4_bytes
)
from dev.tools.sharded_gguf_reader import MultiShardGgufReader

LAYER_Q4_MAGIC = b"MDFL0006"
HEAD_Q4_MAGIC = b"MDFL0002"
EMBEDDING_Q4_MAGIC = b"MDFE0001"

LAYER_Q8_MAGIC = b"MDFL0008"
HEAD_Q8_MAGIC = b"MDFL0008"
EMBEDDING_Q8_MAGIC = b"MDFE0008"

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

F_PUNCHHOLE = 99
OUTPUT_GROWTH = 1.05
IN_FLIGHT_BYTES_PER_WORKER = 3 << 29

def is_full_attention(layer_index: int) -> bool:
    return (layer_index + 1) % LAYOUT["full_attention_period"] == 0

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _punch_hole(path: Path, offset: int, length: int) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        block = os.fstatvfs(descriptor).f_frsize
        start = -(-offset // block) * block
        end = (offset + length) // block * block
        if end > start:
            fcntl.fcntl(
                descriptor, F_PUNCHHOLE, struct.pack("IIqq", 0, 0, start, end - start)
            )
    finally:
        os.close(descriptor)

def can_free_in_place(directory: Path) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".punch-probe-") as probe:
            block = os.fstatvfs(probe.fileno()).f_frsize
            probe.write(b"\1" * block * 4)
            probe.flush()
            os.fsync(probe.fileno())
            before = os.fstat(probe.fileno()).st_blocks
            _punch_hole(Path(probe.name), block, 2 * block)
            return os.fstat(probe.fileno()).st_blocks < before
    except OSError:
        return False

def tensor_owner(name: str) -> str | None:
    if name == "token_embd.weight":
        return "embedding.bin"
    if name.startswith("output"):
        return "head.bin"
    match = re.match(r"blk\.(\d+)\.", name)
    if match:
        layer = int(match.group(1))
        if layer < LAYOUT["layers"]:
            return f"layer-{layer}.bin"
    return None

def concatenate(parts, rows):
    codes = np.vstack([part[0] for part in parts])
    scales = np.vstack([part[1] for part in parts])
    biases = np.vstack([part[2] for part in parts])
    return pad_rows(codes, scales, biases, rows)

def convert_single_layer(model_dir: str, layer_idx: int, out_file: str) -> Tuple[str, Set[str]]:
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    full = is_full_attention(layer_idx)
    kind = 1 if full else 0
    prefix = f"blk.{layer_idx}"
    packed = WeightFile(out_path, LAYER_Q4_MAGIC, layer_idx, kind)

    # 1. Input Layernorm
    norm_key = f"{prefix}.attn_norm.weight"
    if not reader.has(norm_key):
        norm_key = f"{prefix}.input_layernorm.weight"
    norm = reader.read_tensor(norm_key)
    packed.section(to_bf16(norm).tobytes())

    # 2. Attention or GDN Mixer
    if full:
        if reader.has(f"{prefix}.attn_qkv.weight"):
            qkv = reader.read_tensor(f"{prefix}.attn_qkv.weight")
            q4_qkv = quantize_affine(qkv, group=64, bits=4)
            packed.section(tile_q4(*pad_rows(*q4_qkv, LAYOUT["packed_full"])))
        else:
            q = quantize_affine(reader.read_tensor(f"{prefix}.attn_q.weight"), group=64, bits=4)
            k = quantize_affine(reader.read_tensor(f"{prefix}.attn_k.weight"), group=64, bits=4)
            v = quantize_affine(reader.read_tensor(f"{prefix}.attn_v.weight"), group=64, bits=4)
            packed.section(tile_q4(*concatenate([q, k, v], LAYOUT["packed_full"])))
        
        q_norm = reader.read_tensor(f"{prefix}.attn_q_norm.weight") if reader.has(f"{prefix}.attn_q_norm.weight") else np.zeros(LAYOUT["attention_head_dimension"], dtype=np.float32)
        k_norm = reader.read_tensor(f"{prefix}.attn_k_norm.weight") if reader.has(f"{prefix}.attn_k_norm.weight") else np.zeros(LAYOUT["attention_head_dimension"], dtype=np.float32)
        packed.section(to_bf16(q_norm).tobytes())
        packed.section(to_bf16(k_norm).tobytes())

        o_proj = quantize_affine(reader.read_tensor(f"{prefix}.attn_output.weight"), group=64, bits=4)
        packed.section(tile_q4(*o_proj))
    else:
        # GDN Mixer
        if reader.has(f"{prefix}.ssm_in.weight"):
            ssm_in = quantize_affine(reader.read_tensor(f"{prefix}.ssm_in.weight"), group=64, bits=4)
            packed.section(tile_q4(*pad_rows(*ssm_in, LAYOUT["packed_gdn"])))
        else:
            # Fallback to individual projections
            qkv_key = f"{prefix}.ssm_in_qkv.weight" if reader.has(f"{prefix}.ssm_in_qkv.weight") else f"{prefix}.linear_attn.in_proj_qkv"
            z_key = f"{prefix}.ssm_in_z.weight" if reader.has(f"{prefix}.ssm_in_z.weight") else f"{prefix}.linear_attn.in_proj_z"
            b_key = f"{prefix}.ssm_in_b.weight" if reader.has(f"{prefix}.ssm_in_b.weight") else f"{prefix}.linear_attn.in_proj_b"
            a_key = f"{prefix}.ssm_in_a.weight" if reader.has(f"{prefix}.ssm_in_a.weight") else f"{prefix}.linear_attn.in_proj_a"
            qkv = quantize_affine(reader.read_tensor(qkv_key), group=64, bits=4)
            z = quantize_affine(reader.read_tensor(z_key), group=64, bits=4)
            b = quantize_affine(reader.read_tensor(b_key), group=64, bits=4)
            a = quantize_affine(reader.read_tensor(a_key), group=64, bits=4)
            packed.section(tile_q4(*concatenate([qkv, z, b, a], LAYOUT["packed_gdn"])))

        conv_key = f"{prefix}.ssm_conv1d.weight" if reader.has(f"{prefix}.ssm_conv1d.weight") else f"{prefix}.linear_attn.conv1d.weight"
        packed.section(to_bf16(reader.read_tensor(conv_key)).tobytes())

        a_key = f"{prefix}.ssm_a" if reader.has(f"{prefix}.ssm_a") else f"{prefix}.linear_attn.A_log"
        a_log = reader.read_tensor(a_key)
        packed.section((-np.exp(a_log)).astype("<f4").tobytes())

        dt_key = f"{prefix}.ssm_dt" if reader.has(f"{prefix}.ssm_dt") else f"{prefix}.linear_attn.dt_bias"
        packed.section(to_bf16(reader.read_tensor(dt_key)).tobytes())

        norm_key = f"{prefix}.ssm_norm.weight" if reader.has(f"{prefix}.ssm_norm.weight") else f"{prefix}.linear_attn.norm.weight"
        packed.section(to_bf16(reader.read_tensor(norm_key)).tobytes())

        out_key = f"{prefix}.ssm_out.weight" if reader.has(f"{prefix}.ssm_out.weight") else f"{prefix}.linear_attn.out_proj"
        out_proj = quantize_affine(reader.read_tensor(out_key), group=64, bits=4)
        packed.section(tile_q4(*out_proj))

    # 3. Post Attention Layernorm
    post_norm_key = f"{prefix}.ffn_norm.weight" if reader.has(f"{prefix}.ffn_norm.weight") else f"{prefix}.post_attention_layernorm.weight"
    packed.section(to_bf16(reader.read_tensor(post_norm_key)).tobytes())

    # 4. Dense FFN
    gate_key = f"{prefix}.ffn_gate.weight" if reader.has(f"{prefix}.ffn_gate.weight") else f"{prefix}.mlp.gate_proj"
    up_key = f"{prefix}.ffn_up.weight" if reader.has(f"{prefix}.ffn_up.weight") else f"{prefix}.mlp.up_proj"
    down_key = f"{prefix}.ffn_down.weight" if reader.has(f"{prefix}.ffn_down.weight") else f"{prefix}.mlp.down_proj"

    gate = quantize_affine(reader.read_tensor(gate_key), group=64, bits=4)
    up = quantize_affine(reader.read_tensor(up_key), group=64, bits=4)
    down = quantize_affine(reader.read_tensor(down_key), group=64, bits=4)

    packed.section(tile_q4(*gate))
    packed.section(tile_q4(*up))
    packed.section(tile_q4(*down))

    packed.finish()
    read_names = set(reader.read_names)
    reader.close()
    return f"layer-{layer_idx}.bin", read_names

def convert_head(model_dir: str, out_file: str) -> Tuple[str, Set[str]]:
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()
    packed = WeightFile(out_path, HEAD_Q4_MAGIC, LAYOUT["layers"], 2)

    norm_key = "output_norm.weight" if reader.has("output_norm.weight") else "language_model.model.norm.weight"
    packed.section(to_bf16(reader.read_tensor(norm_key)).tobytes())

    out_key = "output.weight" if reader.has("output.weight") else "language_model.lm_head"
    head = quantize_affine(reader.read_tensor(out_key), group=64, bits=4)
    packed.section(tile_q4(*head))
    packed.finish()
    read_names = set(reader.read_names)
    reader.close()
    return "head.bin", read_names

def convert_embedding(model_dir: str, out_file: str) -> Tuple[str, Set[str]]:
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()
    packed = WeightFile(out_path, EMBEDDING_Q4_MAGIC, LAYOUT["vocabulary"], LAYOUT["hidden"])
    emb_key = "token_embd.weight" if reader.has("token_embd.weight") else "language_model.model.embed_tokens"
    emb = quantize_affine(reader.read_tensor(emb_key), group=64, bits=4)
    for run in plain_q4(*emb):
        packed.section(run)
    packed.finish()
    read_names = set(reader.read_names)
    reader.close()
    return "embedding.bin", read_names

def prepare_gguf_model(model_dir: Path, output_dir: Path, workers: int = 4, consume_source: bool = False) -> Path:
    target_dir = output_dir / "target"
    tokenizer_dir = output_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_dir)
    base_dir = model_dir if model_dir.is_dir() else model_dir.parent
    reader = MultiShardGgufReader(model_dir)
    sources = [Path(shard["path"]) for shard in reader.shards]
    own = {path for path in sources if path.resolve().is_relative_to(base_dir.resolve())}
    in_place = consume_source and can_free_in_place(base_dir)

    parts = {
        f"layer-{i}.bin": (convert_single_layer, str(model_dir), i)
        for i in range(LAYOUT["layers"])
    }
    parts["head.bin"] = (convert_head, str(model_dir))
    parts["embedding.bin"] = (convert_embedding, str(model_dir))

    def release_source(part: str, read: Set[str]):
        if not in_place:
            return
        for name in read:
            if reader.has(name):
                shard_path, offset, length = reader.extent(name)
                if Path(shard_path) in own:
                    _punch_hole(Path(shard_path), offset, length)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fn, *args, str(target_dir / name)): name
            for name, (fn, *args) in parts.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            res, read = future.result()
            print(f"  [DONE] {res}", flush=True)
            if in_place:
                release_source(name, read)

    reader.close()

    if consume_source:
        for path in sorted(own):
            if path.exists():
                path.unlink()
    draft_dir = output_dir / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer files if present
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "chat_template.jinja", "config.json"]:
        src_f = base_dir / fname
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
                artifacts.append({
                    "path": str(rel_path),
                    "sha256": sha256_file(file_path),
                    "size": file_path.stat().st_size,
                })

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
            "target_layer_magic": LAYER_Q4_MAGIC.decode(),
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
    print(f"[Slipstream] GGUF preparation complete at {output_dir}", flush=True)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing GGUF shards")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for prepared package")
    parser.add_argument("--workers", type=int, default=4, help="Worker count")
    parser.add_argument("--consume-source", action="store_true", help="Free GGUF source blocks in place")
    args = parser.parse_args()
    prepare_gguf_model(args.model_dir, args.output, workers=args.workers, consume_source=args.consume_source)

if __name__ == "__main__":
    main()
