#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next sharded GGUF models directly into Slipstream format.

Converts all 48 layers, MTP draft layer & combiner, head, embedding, and n-gram table
directly from quantized GGUF weights (Q4_0, Q4_1, Q8_0, Q4_K, etc.) using native
C-accelerated dequantization and parallel worker processes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Slipstream path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))

from dev.tools.quantize import to_bf16, from_bf16, quantize_affine, pack_nibbles
from dev.tools.package_format import (
    ALIGNMENT, BF16, GROUP, FINE_GROUP, STORAGE_N, EXPERT_STORAGE_N,
    StreamingWeightFile, WeightFile, pad_rows, tile_q4, q4_bytes
)
from dev.tools.sharded_gguf_reader import MultiShardGgufReader
from models.qwen4exp.tools.convert_qwen4exp import (
    quantized_q8, quantized_q8_tile, quantized_tile, q8_bytes,
    LAYER_MAGIC, NGRAM_MAGIC, HEAD_MAGIC, EMBEDDING_MAGIC, DRAFT_MAGIC,
    write_manifest, write_placeholder_draft
)

MTP_COMBINER_MAGIC = b"MDFN0035"
HYPER_DOWN_PADDED = 512

NGRAM_VOCABULARY = 320_001_536
NGRAM_HEAD_DIMENSION = 160
NGRAM_SHARDS = 128
NGRAM_CHUNK_ROWS = 1 << 19
WORKER_MEMORY_BYTES = 12 << 30

# The GGUF carries no HF tokenizer files; the base model's match its vocabulary
# and chat template exactly. config.json is written by write_manifest.
TOKENIZER_REPO = "Qwen/Qwen3.8-Flash-Next"
TOKENIZER_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json")

LAYOUT = {
    "layers": 48,
    "hidden": 2560,
    "vocabulary": 248320,
    "packed_gdn": 16640,
    "packed_full": 13312,
    "attention_width": 6144,
    "experts": 512,
    "expert_intermediate": 640,
    "convolution": 10240,
    "value_heads": 48,
    "head_dimension": 128,
}

def is_full_attention(layer: int) -> bool:
    return (layer + 1) % 4 == 0

def convert_single_layer(model_dir: str, sidecar_path: str | None, layer_idx: int, out_file: str) -> str:
    reader = MultiShardGgufReader(model_dir, sidecars=sidecar_path)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    full = is_full_attention(layer_idx)
    kind = 1 if full else 0
    prefix = f"blk.{layer_idx}"
    packed = WeightFile(out_path, LAYER_MAGIC, layer_idx, kind)

    # 1. Attention Hyper-connection (norm stored as 1.0 + weight in GGUF)
    norm = reader.read_tensor(f"{prefix}.hc_attn_norm.weight") - 1.0
    packed.section(to_bf16(norm).tobytes())

    mix_down = reader.read_tensor(f"{prefix}.hc_attn_down.weight")
    packed.section(quantized_q8_tile(mix_down, pad_to=HYPER_DOWN_PADDED))

    mix_up = reader.read_tensor(f"{prefix}.hc_attn_up.weight")
    packed.section(quantized_q8_tile(mix_up))

    inject = reader.read_tensor(f"{prefix}.hc_attn_inject.weight")
    packed.section(to_bf16(inject).tobytes())

    # 2. Mixer (GDN or Full Attention)
    if full:
        q = reader.read_tensor(f"{prefix}.attn_q.weight")
        k = reader.read_tensor(f"{prefix}.attn_k.weight")
        v = reader.read_tensor(f"{prefix}.attn_v.weight")
        stacked_qkv = np.vstack([q, k, v])
        packed.section(quantized_q8_tile(stacked_qkv, pad_to=LAYOUT["packed_full"]))

        # Attention RMS norms already include 1.0 + weight in GGUF
        q_norm = reader.read_tensor(f"{prefix}.attn_q_norm.weight")
        k_norm = reader.read_tensor(f"{prefix}.attn_k_norm.weight")
        packed.section(to_bf16(q_norm).tobytes())
        packed.section(to_bf16(k_norm).tobytes())

        out_proj = reader.read_tensor(f"{prefix}.attn_output.weight")
        packed.section(quantized_q8_tile(out_proj))

        idx_q = reader.read_tensor(f"{prefix}.indexer.q_proj.weight")
        idx_k = reader.read_tensor(f"{prefix}.indexer.k_proj.weight")
        idx_qk = np.vstack([idx_q, idx_k])
        packed.section(quantized_tile(idx_qk, storage_n=EXPERT_STORAGE_N))

        # Indexer RMS norms in GGUF have +1.0 added; Splash kernel expects raw weight
        idx_q_norm = reader.read_tensor(f"{prefix}.indexer.q_norm.weight") - 1.0
        idx_k_norm = reader.read_tensor(f"{prefix}.indexer.k_norm.weight") - 1.0
        packed.section(to_bf16(idx_q_norm).tobytes())
        packed.section(to_bf16(idx_k_norm).tobytes())
    else:
        # GDN value heads in GGUF are permuted (3, 16) instead of canonical (16, 3)
        qkv = reader.read_tensor(f"{prefix}.attn_qkv.weight")
        qkv_v = qkv[4096:].reshape(3, 16, 128, 2560).transpose(1, 0, 2, 3).reshape(6144, 2560)
        qkv = np.vstack([qkv[:4096], qkv_v])

        gate = reader.read_tensor(f"{prefix}.attn_gate.weight")
        gate = gate.reshape(3, 16, 128, 2560).transpose(1, 0, 2, 3).reshape(6144, 2560)

        beta = reader.read_tensor(f"{prefix}.ssm_beta.weight")
        beta = beta.reshape(3, 16, 2560).transpose(1, 0, 2).reshape(48, 2560)

        alpha = reader.read_tensor(f"{prefix}.ssm_alpha.weight")
        alpha = alpha.reshape(3, 16, 2560).transpose(1, 0, 2).reshape(48, 2560)

        stacked = np.vstack([qkv, gate, beta, alpha])
        packed.section(quantized_q8_tile(stacked, pad_to=LAYOUT["packed_gdn"]))

        conv1d = reader.read_tensor(f"{prefix}.ssm_conv1d.weight")
        conv_v = conv1d[4096:].reshape(3, 16, 128, 4).transpose(1, 0, 2, 3).reshape(6144, 4)
        conv1d = np.vstack([conv1d[:4096], conv_v])
        packed.section(to_bf16(conv1d.flatten()).tobytes())

        ssm_a = reader.read_tensor(f"{prefix}.ssm_a")
        ssm_a = ssm_a.reshape(3, 16).T.flatten()
        packed.section(ssm_a.astype("<f4").tobytes())

        ssm_dt = reader.read_tensor(f"{prefix}.ssm_dt.bias")
        ssm_dt = ssm_dt.reshape(3, 16).T.flatten()
        packed.section(to_bf16(ssm_dt).tobytes())

        ssm_norm = reader.read_tensor(f"{prefix}.ssm_norm.weight")
        packed.section(to_bf16(ssm_norm).tobytes())

        out_proj = reader.read_tensor(f"{prefix}.ssm_out.weight")
        out_proj = out_proj.reshape(2560, 3, 16, 128).transpose(0, 2, 1, 3).reshape(2560, 6144)
        packed.section(quantized_q8_tile(out_proj))

    # 3. MLP Hyper-connection (norm stored as 1.0 + weight in GGUF)
    mlp_norm = reader.read_tensor(f"{prefix}.hc_ffn_norm.weight") - 1.0
    packed.section(to_bf16(mlp_norm).tobytes())

    mlp_mix_down = reader.read_tensor(f"{prefix}.hc_ffn_down.weight")
    packed.section(quantized_q8_tile(mlp_mix_down, pad_to=HYPER_DOWN_PADDED))

    mlp_mix_up = reader.read_tensor(f"{prefix}.hc_ffn_up.weight")
    packed.section(quantized_q8_tile(mlp_mix_up))

    mlp_inject = reader.read_tensor(f"{prefix}.hc_ffn_inject.weight")
    packed.section(to_bf16(mlp_inject).tobytes())

    # 4. MoE Experts (Router uses [quant group][row] layout via quantized_q8)
    router = reader.read_tensor(f"{prefix}.ffn_gate_inp.weight")
    packed.section(quantized_q8(router))

    # 512 experts gate, up, down, one expert at a time: whole, each tensor is 3.4 GB as float32
    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_gate_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_up_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_down_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    # Shared expert
    shexp_gate = reader.read_tensor(f"{prefix}.ffn_gate_shexp.weight")
    packed.section(quantized_tile(shexp_gate, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_up = reader.read_tensor(f"{prefix}.ffn_up_shexp.weight")
    packed.section(quantized_tile(shexp_up, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_down = reader.read_tensor(f"{prefix}.ffn_down_shexp.weight")
    packed.section(quantized_tile(shexp_down, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_gate_inp = reader.read_tensor(f"{prefix}.ffn_gate_inp_shexp.weight")
    padded = np.zeros((STORAGE_N, shexp_gate_inp.shape[0]), dtype=np.float32)
    padded[0] = shexp_gate_inp
    packed.section(quantized_q8(padded))

    total_bytes = packed.finish()
    reader.close()
    return f"Layer {layer_idx:2d} finished ({total_bytes / (1024*1024):.1f} MB)"

def convert_mtp_layer(model_dir: str, sidecar_path: str, out_file: str) -> str:
    reader = MultiShardGgufReader(model_dir, sidecars=sidecar_path)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    prefix = "blk.48"
    packed = WeightFile(out_path, LAYER_MAGIC, 48, 1)

    # 1. Attention Hyper-connection
    norm = reader.read_tensor(f"{prefix}.hc_attn_norm.weight") - 1.0
    packed.section(to_bf16(norm).tobytes())

    mix_down = reader.read_tensor(f"{prefix}.hc_attn_down.weight")
    packed.section(quantized_q8_tile(mix_down, pad_to=HYPER_DOWN_PADDED))

    mix_up = reader.read_tensor(f"{prefix}.hc_attn_up.weight")
    packed.section(quantized_q8_tile(mix_up))

    inject = reader.read_tensor(f"{prefix}.hc_attn_inject.weight")
    packed.section(to_bf16(inject).tobytes())

    # 2. Full Attention
    q = reader.read_tensor(f"{prefix}.attn_q.weight")
    k = reader.read_tensor(f"{prefix}.attn_k.weight")
    v = reader.read_tensor(f"{prefix}.attn_v.weight")
    stacked_qkv = np.vstack([q, k, v])
    packed.section(quantized_q8_tile(stacked_qkv, pad_to=LAYOUT["packed_full"]))

    # Attention RMS norms already include 1.0 + weight in GGUF
    q_norm = reader.read_tensor(f"{prefix}.attn_q_norm.weight")
    k_norm = reader.read_tensor(f"{prefix}.attn_k_norm.weight")
    packed.section(to_bf16(q_norm).tobytes())
    packed.section(to_bf16(k_norm).tobytes())

    out_proj = reader.read_tensor(f"{prefix}.attn_output.weight")
    packed.section(quantized_q8_tile(out_proj))

    idx_q = reader.read_tensor(f"{prefix}.indexer.q_proj.weight")
    idx_k = reader.read_tensor(f"{prefix}.indexer.k_proj.weight")
    idx_qk = np.vstack([idx_q, idx_k])
    packed.section(quantized_tile(idx_qk, storage_n=EXPERT_STORAGE_N))

    # Indexer RMS norms in GGUF have +1.0 added; Splash kernel expects raw weight
    idx_q_norm = reader.read_tensor(f"{prefix}.indexer.q_norm.weight") - 1.0
    idx_k_norm = reader.read_tensor(f"{prefix}.indexer.k_norm.weight") - 1.0
    packed.section(to_bf16(idx_q_norm).tobytes())
    packed.section(to_bf16(idx_k_norm).tobytes())

    # 3. MLP Hyper-connection (norm stored as 1.0 + weight in GGUF)
    mlp_norm = reader.read_tensor(f"{prefix}.hc_ffn_norm.weight") - 1.0
    packed.section(to_bf16(mlp_norm).tobytes())

    mlp_mix_down = reader.read_tensor(f"{prefix}.hc_ffn_down.weight")
    packed.section(quantized_q8_tile(mlp_mix_down, pad_to=HYPER_DOWN_PADDED))

    mlp_mix_up = reader.read_tensor(f"{prefix}.hc_ffn_up.weight")
    packed.section(quantized_q8_tile(mlp_mix_up))

    mlp_inject = reader.read_tensor(f"{prefix}.hc_ffn_inject.weight")
    packed.section(to_bf16(mlp_inject).tobytes())

    # 4. MoE Experts (Router uses [quant group][row] layout via quantized_q8)
    router = reader.read_tensor(f"{prefix}.ffn_gate_inp.weight")
    packed.section(quantized_q8(router))

    # 512 experts gate, up, down
    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_gate_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_up_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    packed.section(b"".join(
        quantized_tile(reader.read_expert(f"{prefix}.ffn_down_exps.weight", i),
                       storage_n=EXPERT_STORAGE_N, group=GROUP)
        for i in range(512)
    ))

    # Shared expert
    shexp_gate = reader.read_tensor(f"{prefix}.ffn_gate_shexp.weight")
    packed.section(quantized_tile(shexp_gate, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_up = reader.read_tensor(f"{prefix}.ffn_up_shexp.weight")
    packed.section(quantized_tile(shexp_up, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_down = reader.read_tensor(f"{prefix}.ffn_down_shexp.weight")
    packed.section(quantized_tile(shexp_down, storage_n=EXPERT_STORAGE_N, group=GROUP))

    shexp_gate_inp = reader.read_tensor(f"{prefix}.ffn_gate_inp_shexp.weight")
    padded = np.zeros((STORAGE_N, shexp_gate_inp.shape[0]), dtype=np.float32)
    padded[0] = shexp_gate_inp
    packed.section(quantized_q8(padded))

    total_bytes = packed.finish()
    reader.close()
    return f"MTP Layer (48) finished ({total_bytes / (1024*1024):.1f} MB)"

def convert_mtp_combiner(model_dir: str, sidecar_path: str, out_file: str) -> str:
    reader = MultiShardGgufReader(model_dir, sidecars=sidecar_path)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    combiner = WeightFile(out_path, MTP_COMBINER_MAGIC, 48, 3)

    enorm = reader.read_tensor("blk.48.nextn.enorm.weight") - 1.0
    combiner.section(to_bf16(enorm).tobytes())

    hnorm = reader.read_tensor("blk.48.nextn.hnorm.weight") - 1.0
    combiner.section(to_bf16(hnorm).tobytes())

    fc_emb = reader.read_tensor("blk.48.nextn.fc_embd.weight")
    combiner.section(quantized_tile(fc_emb))

    fc_hid = reader.read_tensor("blk.48.nextn.fc_hidden.weight")
    combiner.section(quantized_tile(fc_hid))

    hc_norm = reader.read_tensor("blk.48.nextn.hc_norm.weight") - 1.0
    combiner.section(to_bf16(hc_norm).tobytes())

    hc_down = reader.read_tensor("blk.48.nextn.hc_down.weight")
    combiner.section(quantized_q8_tile(hc_down, pad_to=HYPER_DOWN_PADDED))

    hc_up = reader.read_tensor("blk.48.nextn.hc_up.weight")
    combiner.section(quantized_q8_tile(hc_up))

    total = combiner.finish()
    reader.close()
    return f"MTP Combiner finished ({total / (1024*1024):.1f} MB)"

def convert_head(model_dir: str, out_file: str) -> str:
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    packed = WeightFile(out_path, HEAD_MAGIC, 48, 2)

    norm = reader.read_tensor("output_hc_norm.weight") - 1.0
    packed.section(to_bf16(norm).tobytes())

    mix_down = reader.read_tensor("output_hc_down.weight")
    packed.section(quantized_q8_tile(mix_down, pad_to=HYPER_DOWN_PADDED))

    mix_up = reader.read_tensor("output_hc_up.weight")
    packed.section(quantized_q8_tile(mix_up))

    # Final norm (zero placeholder if absent in GGUF)
    if reader.has("output_norm.weight"):
        packed.section(to_bf16(reader.read_tensor("output_norm.weight")).tobytes())
    else:
        packed.section(b"\0" * (LAYOUT["hidden"] * BF16))

    head = reader.read_tensor("output.weight")
    packed.section(quantized_q8_tile(head))
    packed.section(quantized_tile(head))

    total = packed.finish()
    reader.close()
    return f"Head finished ({total / (1024*1024):.1f} MB)"

def convert_embedding(model_dir: str, out_file: str) -> str:
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    packed = WeightFile(out_path, EMBEDDING_MAGIC, LAYOUT["vocabulary"], LAYOUT["hidden"])
    embed = reader.read_tensor("token_embd.weight")
    rows, width = embed.shape
    groups = embed.reshape(rows, width // GROUP, GROUP)
    low = groups.min(axis=-1)
    spread = groups.max(axis=-1) - low
    scale_bits = to_bf16(np.where(spread > 0, spread / 255.0, 1.0))
    bias_bits = to_bf16(low)
    codes = np.clip(
        np.rint(
            (groups - from_bf16(bias_bits)[..., None])
            / from_bf16(scale_bits)[..., None]
        ),
        0,
        255,
    ).astype(np.uint8)
    packed.section(codes.tobytes())
    packed.section(scale_bits.tobytes())
    packed.section(bias_bits.tobytes())

    total = packed.finish()
    reader.close()
    return f"Embedding finished ({total / (1024*1024):.1f} MB)"

def convert_ngram(model_dir: str, out_file: str) -> str:
    """The per-layer n-gram table and its PLE projections, in write_per_layer_embedding's order.

    llama.cpp concatenates the 128 checkpoint shards into one table of 320M rows
    of 160. Dequantized whole it would be ~190 GiB, so it is requantized a chunk
    of rows at a time: codes stream straight into the file, while scales and
    biases (3 GiB each) go to scratch files beside it and are appended after.
    """
    reader = MultiShardGgufReader(model_dir)
    out_path = Path(out_file)
    if out_path.exists():
        out_path.unlink()

    table = "per_layer_token_embd.weight"
    width, rows = reader.tensors[table]["dims"]
    if (width, rows) != (NGRAM_HEAD_DIMENSION, NGRAM_VOCABULARY):
        raise ValueError(f"{table} is {rows}x{width}, expected {NGRAM_VOCABULARY}x{NGRAM_HEAD_DIMENSION}")

    packed = StreamingWeightFile(out_path, NGRAM_MAGIC, NGRAM_SHARDS, width)
    scratch = [out_path.with_name(out_path.name + f".{part}.tmp") for part in ("scales", "biases")]
    try:
        with open(scratch[0], "wb") as scales, open(scratch[1], "wb") as biases:
            packed.begin()
            for start in range(0, rows, NGRAM_CHUNK_ROWS):
                values = reader.read_rows(table, start, min(NGRAM_CHUNK_ROWS, rows - start))
                codes, scale_bits, bias_bits = quantize_affine(values, group=FINE_GROUP)
                packed.write(pack_nibbles(codes).tobytes())
                scales.write(scale_bits.tobytes())
                biases.write(bias_bits.tobytes())
        for path in scratch:
            packed.begin()
            with open(path, "rb") as part:
                while chunk := part.read(1 << 26):
                    packed.write(chunk)
    finally:
        for path in scratch:
            path.unlink(missing_ok=True)

    # The hash constants travel as GGUF metadata, already little-endian 64-bit.
    for key in ("head_offsets", "head_vocab_sizes", "layer_multipliers"):
        packed.section(reader.metadata[f"qwen4exp.ple.{key}"])

    prefix = "blk.1"
    packed.section(quantized_tile(reader.read_tensor(f"{prefix}.ple_key.weight")))
    packed.section(quantized_tile(reader.read_tensor(f"{prefix}.ple_value.weight")))
    # PLE norms stored as 1.0 + weight in GGUF
    for norm in ("key", "query", "conv"):
        weight = reader.read_tensor(f"{prefix}.ple_norm_{norm}.weight") - 1.0
        packed.section(to_bf16(weight).tobytes())
    packed.section(to_bf16(reader.read_tensor(f"{prefix}.ple_conv1d.weight")).tobytes())

    total = packed.finish()
    reader.close()
    return f"N-gram table finished ({total / (1024*1024):.1f} MB)"

def fetch_tokenizer(tokenizer_dir: Path) -> None:
    """Download the base model's tokenizer; its vocabulary and chat template match the GGUF's."""
    from huggingface_hub import hf_hub_download

    for name in TOKENIZER_FILES:
        if not (tokenizer_dir / name).exists():
            shutil.copyfile(
                hf_hub_download(TOKENIZER_REPO, name, revision=TOKENIZER_REVISION),
                tokenizer_dir / name,
            )
    print(f"Fetched tokenizer from {TOKENIZER_REPO}")

def default_workers() -> int:
    """Workers that fit in memory: a layer peaks near 4 GiB, the head near 13 GiB."""
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return max(1, min(8, memory // WORKER_MEMORY_BYTES))

def prepare_gguf_model(
    model_dir: str | Path,
    output_dir: str | Path,
    sidecar_path: str | Path | None = None,
    reference_dir: str | Path | None = None,
    workers: int | None = None
) -> None:
    if workers is None:
        workers = default_workers()
    model_dir = Path(model_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    target_dir = output_dir / "target"
    draft_dir = output_dir / "draft"
    tokenizer_dir = output_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if sidecar_path is None:
        candidates = [
            model_dir / "MTP/mtp-shared-Q4_K_M.gguf",
            model_dir / "mtp-shared-Q4_K_M.gguf",
            Path.home() / "models/qwen38-flash-next-mtp/mtp-shared-Q4_K_M.gguf",
            Path.home() / "models/qwen38-flash-next-v3/MTP/mtp-shared-Q4_K_M.gguf",
        ]
        for candidate in candidates:
            if candidate.exists():
                sidecar_path = candidate
                break

    # A previously prepared package saves converting the n-gram table; without
    # one everything is built from the GGUF. The output itself never counts.
    if reference_dir is None:
        for candidate in [
            Path.home() / "models/qwen38-flash-next-v3/prepared",
            Path.home() / "models/qwen38-flash-next-splash",
        ]:
            if candidate.resolve() != output_dir and (candidate / "target/ngram.bin").exists():
                reference_dir = candidate
                break

    start_time = time.perf_counter()
    print(f"=== Fast GGUF Ingestion for Qwen3.8-Flash-Next ===")
    print(f"Source: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Sidecar: {sidecar_path}")
    print(f"Reference: {reference_dir}")
    print(f"Workers: {workers}")

    # Copy / hardlink n-gram table if available, else convert it with the layers
    ngram_dst = target_dir / "ngram.bin"
    convert_ngram_table = False
    if not ngram_dst.exists():
        if reference_dir and (Path(reference_dir) / "target/ngram.bin").exists():
            ngram_src = Path(reference_dir) / "target/ngram.bin"
            try:
                os.link(ngram_src, ngram_dst)
                print(f"Linked ngram.bin from {ngram_src} (0 extra bytes)")
            except Exception:
                shutil.copyfile(ngram_src, ngram_dst)
                print(f"Copied ngram.bin from {ngram_src}")
        else:
            convert_ngram_table = True

    # Copy tokenizer & draft vocab if available
    if reference_dir:
        ref_path = Path(reference_dir)
        vocab_src = ref_path / "target/draft-vocab.bin"
        if vocab_src.exists() and not (target_dir / "draft-vocab.bin").exists():
            shutil.copyfile(vocab_src, target_dir / "draft-vocab.bin")
            print("Copied draft-vocab.bin")

        for f in (ref_path / "tokenizer").glob("*"):
            if not (tokenizer_dir / f.name).exists():
                shutil.copyfile(f, tokenizer_dir / f.name)
        print("Copied tokenizer files")

        for f in (ref_path / "draft").glob("*.bin"):
            if not (draft_dir / f.name).exists():
                shutil.copyfile(f, draft_dir / f.name)
        print("Copied draft files")

    if not all((tokenizer_dir / name).exists() for name in TOKENIZER_FILES):
        fetch_tokenizer(tokenizer_dir)
    # This model has no DFlash 2 draft; the package format still expects one.
    if not (draft_dir / "model.bin").exists():
        write_placeholder_draft(draft_dir)
        print("Wrote placeholder draft")

    # Parallel conversion tasks
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Layers 0..47
        for i in range(48):
            out_file = str(target_dir / f"layer-{i}.bin")
            tasks.append(executor.submit(convert_single_layer, str(model_dir), str(sidecar_path), i, out_file))

        # MTP layer & combiner
        if sidecar_path and Path(sidecar_path).exists():
            mtp_layer_file = str(target_dir / "mtp-layer.bin")
            mtp_comb_file = str(target_dir / "mtp-combiner.bin")
            tasks.append(executor.submit(convert_mtp_layer, str(model_dir), str(sidecar_path), mtp_layer_file))
            tasks.append(executor.submit(convert_mtp_combiner, str(model_dir), str(sidecar_path), mtp_comb_file))
        elif reference_dir:
            ref_path = Path(reference_dir)
            for mtp_name in ["mtp-layer.bin", "mtp-combiner.bin"]:
                src = ref_path / f"target/{mtp_name}"
                dst = target_dir / mtp_name
                if src.exists() and not dst.exists():
                    try:
                        os.link(src, dst)
                        print(f"Linked {mtp_name} from {src} (0 extra bytes)")
                    except Exception:
                        shutil.copyfile(src, dst)
                        print(f"Copied {mtp_name} from {src}")

        # Head & Embedding
        head_file = str(target_dir / "head.bin")
        emb_file = str(target_dir / "embedding.bin")
        tasks.append(executor.submit(convert_head, str(model_dir), head_file))
        tasks.append(executor.submit(convert_embedding, str(model_dir), emb_file))
        if convert_ngram_table:
            tasks.append(executor.submit(convert_ngram, str(model_dir), str(ngram_dst)))

        for future in concurrent.futures.as_completed(tasks):
            try:
                res = future.result()
                print(f"  [DONE] {res}")
            except Exception as e:
                print(f"  [ERROR] Task failed: {e}")
                raise e

    # Last, so a package with a manifest is a complete one: the launcher
    # treats manifest.json and layer-0.bin as "already prepared".
    write_manifest(output_dir)
    print("Generated manifest.json")

    elapsed = time.perf_counter() - start_time
    print(f"=== Successfully prepared Slipstream model in {elapsed:.1f}s ===")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory containing GGUF shards")
    parser.add_argument("--output", required=True, help="Destination directory for prepared package")
    parser.add_argument("--sidecar", default=None, help="Path to MTP draft sidecar GGUF")
    parser.add_argument("--reference", default=None, help="Previously prepared package to reuse ngram.bin, tokenizer and draft from")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: by installed memory, at most 8)")
    args = parser.parse_args()

    prepare_gguf_model(args.model_dir, args.output, args.sidecar, args.reference, workers=args.workers)

if __name__ == "__main__":
    main()
