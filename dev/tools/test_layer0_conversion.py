import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev.tools.package_format import (
    EXPERT_STORAGE_N,
    GROUP,
    STORAGE_N,
    WeightFile,
    tile_q4,
)
from dev.tools.quantize import from_bf16, quantize_affine, to_bf16
from models.qwen4exp.tools.convert_qwen4exp import quantized_q8_tile

LAYER_MAGIC = b"MDFN0031"
HYPER_DOWN_PADDED = 512

LAYOUT = {
    "hidden": 2560,
    "packed_gdn": 16640,
    "packed_full": 13312,
    "attention_width": 6144,
    "experts": 512,
    "expert_intermediate": 640,
    "convolution": 10240,
    "value_heads": 48,
    "head_dimension": 128,
}


class GgufShardReader:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = open(self.path, "rb")
        magic = self.fd.read(4)
        assert magic == b"GGUF", f"Not GGUF: {path}"
        ver, n_tensors, n_kv = struct.unpack("<IQQ", self.fd.read(20))
        self.kv = {}
        for _ in range(n_kv):
            klen = struct.unpack("<Q", self.fd.read(8))[0]
            self.fd.read(klen)  # the key; only its value's type matters here
            vtype = struct.unpack("<I", self.fd.read(4))[0]
            if vtype in (0, 1, 7):
                self.fd.seek(1, 1)
            elif vtype in (2, 3):
                self.fd.seek(2, 1)
            elif vtype in (4, 5, 6):
                self.fd.seek(4, 1)
            elif vtype in (10, 11, 12):
                self.fd.seek(8, 1)
            elif vtype == 8:
                slen = struct.unpack("<Q", self.fd.read(8))[0]
                self.fd.seek(slen, 1)
            elif vtype == 9:
                etype, count = struct.unpack("<IQ", self.fd.read(12))
                sz = (
                    1
                    if etype in (0, 1, 7)
                    else 2
                    if etype in (2, 3)
                    else 4
                    if etype in (4, 5, 6)
                    else 8
                )
                if etype == 8:
                    for _ in range(count):
                        slen = struct.unpack("<Q", self.fd.read(8))[0]
                        self.fd.seek(slen, 1)
                else:
                    self.fd.seek(count * sz, 1)
        self.tensors = {}
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", self.fd.read(8))[0]
            name = self.fd.read(nlen).decode("latin1")
            ndims = struct.unpack("<I", self.fd.read(4))[0]
            dims = struct.unpack(f"<{ndims}Q", self.fd.read(ndims * 8))
            ttype, toffset = struct.unpack("<IQ", self.fd.read(12))
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": toffset}
        pos = self.fd.tell()
        align = 32
        pad = (align - pos % align) % align
        self.data_offset = pos + pad

    def read_tensor(self, name):
        meta = self.tensors[name]
        offset = self.data_offset + meta["offset"]
        dims = meta["dims"]
        ttype = meta["type"]
        self.fd.seek(offset)

        # Dequantize or read into float32 numpy array
        if ttype == 0:  # F32
            count = int(np.prod(dims))
            raw = self.fd.read(count * 4)
            arr = np.frombuffer(raw, dtype="<f4")
            # GGUF dims: [ne0, ne1, ...] -> numpy shape [..., ne1, ne0]
            shape = tuple(reversed(dims))
            return arr.reshape(shape)
        elif ttype == 1:  # F16
            count = int(np.prod(dims))
            raw = self.fd.read(count * 2)
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            shape = tuple(reversed(dims))
            return arr.reshape(shape)
        elif ttype == 30:  # BF16
            count = int(np.prod(dims))
            raw = self.fd.read(count * 2)
            u16 = np.frombuffer(raw, dtype="<u2")
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
            shape = tuple(reversed(dims))
            return arr.reshape(shape)
        elif ttype == 8:  # Q8_0
            # Block of 32: 2 bytes f16 scale + 32 bytes i8
            ne0 = dims[0]
            rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
            blocks_per_row = ne0 // 32
            row_bytes = blocks_per_row * 34
            raw = self.fd.read(rows * row_bytes)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks_per_row, 34)
            d = arr[:, :, :2].copy().view("<f2").astype(np.float32)  # (rows, blocks, 1)
            q = arr[:, :, 2:].view(np.int8).astype(np.float32)  # (rows, blocks, 32)
            vals = (q * d).reshape(tuple(reversed(dims)))
            return vals
        elif ttype == 2:  # Q4_0
            # Block of 32: 2 bytes f16 scale + 16 bytes nibbles
            ne0 = dims[0]
            rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
            blocks_per_row = ne0 // 32
            row_bytes = blocks_per_row * 18
            raw = self.fd.read(rows * row_bytes)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks_per_row, 18)
            d = arr[:, :, :2].copy().view("<f2").astype(np.float32)  # (rows, blocks, 1)
            nibbles = arr[:, :, 2:]  # (rows, blocks, 16)
            q0 = (nibbles & 0x0F).astype(np.float32) - 8.0
            q1 = (nibbles >> 4).astype(np.float32) - 8.0
            # Interleave elements: 0..15 in q0, 16..31 in q1
            q = np.concatenate([q0, q1], axis=-1)  # (rows, blocks, 32)
            vals = (q * d).reshape(tuple(reversed(dims)))
            return vals
        elif ttype == 3:  # Q4_1
            # Block of 32: 2 bytes f16 scale d + 2 bytes f16 min m + 16 bytes nibbles
            ne0 = dims[0]
            rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
            blocks_per_row = ne0 // 32
            row_bytes = blocks_per_row * 20
            raw = self.fd.read(rows * row_bytes)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks_per_row, 20)
            d = arr[:, :, :2].copy().view("<f2").astype(np.float32)  # (rows, blocks, 1)
            m = (
                arr[:, :, 2:4].copy().view("<f2").astype(np.float32)
            )  # (rows, blocks, 1)
            nibbles = arr[:, :, 4:]  # (rows, blocks, 16)
            q0 = (nibbles & 0x0F).astype(np.float32)
            q1 = (nibbles >> 4).astype(np.float32)
            q = np.concatenate([q0, q1], axis=-1)  # (rows, blocks, 32)
            vals = (q * d + m).reshape(tuple(reversed(dims)))
            return vals
        else:
            raise ValueError(f"Unsupported type {ttype} for {name}")


def write_test_layer0():
    shard1_path = "/Users/nitin/models/qwen38-flash-next-v3/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf"
    print(f"Reading {shard1_path}...")
    reader = GgufShardReader(shard1_path)

    out_path = Path("/tmp/test_layer-0.bin")
    if out_path.exists():
        out_path.unlink()
    packed = WeightFile(out_path, LAYER_MAGIC, 0, 0)

    # 1. Attention Hyper-connection
    print("Writing attention hyper-connection...")
    norm = reader.read_tensor("blk.0.hc_attn_norm.weight")
    packed.section(to_bf16(norm).tobytes())

    mix_down = reader.read_tensor("blk.0.hc_attn_down.weight")
    packed.section(quantized_q8_tile(mix_down, pad_to=HYPER_DOWN_PADDED))

    mix_up = reader.read_tensor("blk.0.hc_attn_up.weight")
    packed.section(quantized_q8_tile(mix_up))

    inject = reader.read_tensor("blk.0.hc_attn_inject.weight")
    packed.section(to_bf16(inject).tobytes())

    # 2. GDN Mixer
    print("Writing GDN mixer...")
    qkv = reader.read_tensor("blk.0.attn_qkv.weight")  # (10240, 2560)
    gate = reader.read_tensor("blk.0.attn_gate.weight")  # (6144, 2560)
    beta = reader.read_tensor("blk.0.ssm_beta.weight")  # (48, 2560)
    alpha = reader.read_tensor("blk.0.ssm_alpha.weight")  # (48, 2560)
    parts = [qkv, gate, beta, alpha]
    stacked = np.vstack(parts)
    packed.section(quantized_q8_tile(stacked, pad_to=LAYOUT["packed_gdn"]))

    conv1d = reader.read_tensor("blk.0.ssm_conv1d.weight")
    packed.section(to_bf16(conv1d).tobytes())

    ssm_a = reader.read_tensor("blk.0.ssm_a")
    packed.section(ssm_a.astype("<f4").tobytes())

    dt_bias = reader.read_tensor("blk.0.ssm_dt.bias")
    packed.section(to_bf16(dt_bias).tobytes())

    ssm_norm = reader.read_tensor("blk.0.ssm_norm.weight")
    packed.section(to_bf16(ssm_norm).tobytes())

    out_proj = reader.read_tensor("blk.0.ssm_out.weight")
    packed.section(quantized_q8_tile(out_proj))

    # 3. MLP Hyper-connection
    print("Writing MLP hyper-connection...")
    mlp_norm = reader.read_tensor("blk.0.hc_ffn_norm.weight")
    packed.section(to_bf16(mlp_norm).tobytes())

    mlp_mix_down = reader.read_tensor("blk.0.hc_ffn_down.weight")
    packed.section(quantized_q8_tile(mlp_mix_down, pad_to=HYPER_DOWN_PADDED))

    mlp_mix_up = reader.read_tensor("blk.0.hc_ffn_up.weight")
    packed.section(quantized_q8_tile(mlp_mix_up))

    mlp_inject = reader.read_tensor("blk.0.hc_ffn_inject.weight")
    packed.section(to_bf16(mlp_inject).tobytes())

    # 4. MoE Experts
    print("Writing MoE experts...")
    router = reader.read_tensor("blk.0.ffn_gate_inp.weight")  # (512, 2560)
    # router in convert_qwen4exp.py: quantized_q8(router)
    out, inp = router.shape
    groups = inp // GROUP
    blocks = router.reshape(out, groups, GROUP)
    low = blocks.min(axis=2)
    high = blocks.max(axis=2)
    anchor_low = np.abs(low) > np.abs(high)
    bias = np.where(anchor_low, low, high)
    other = np.where(anchor_low, high, low)
    scale = (other - bias) / 255.0
    scale_bits = to_bf16(scale)
    bias_bits = to_bf16(bias)
    stored_scale = from_bf16(scale_bits)[..., None]
    stored_bias = from_bf16(bias_bits)[..., None]
    safe = np.where(stored_scale == 0, 1.0, stored_scale)
    codes = np.clip(np.rint((blocks - stored_bias) / safe), 0, 255).astype(np.uint8)
    packed.section(codes.tobytes() + scale_bits.tobytes() + bias_bits.tobytes())

    # 512 experts gate, up, down
    print("  Packing 512 expert gate projections...")
    gate_exps = reader.read_tensor("blk.0.ffn_gate_exps.weight")  # (512, 640, 2560)
    packed.section(
        b"".join(
            tile_q4(
                *quantize_affine(gate_exps[i], group=GROUP),
                storage_n=EXPERT_STORAGE_N,
                group=GROUP,
            )
            for i in range(512)
        )
    )

    print("  Packing 512 expert up projections...")
    up_exps = reader.read_tensor("blk.0.ffn_up_exps.weight")  # (512, 640, 2560)
    packed.section(
        b"".join(
            tile_q4(
                *quantize_affine(up_exps[i], group=GROUP),
                storage_n=EXPERT_STORAGE_N,
                group=GROUP,
            )
            for i in range(512)
        )
    )

    print("  Packing 512 expert down projections...")
    down_exps = reader.read_tensor("blk.0.ffn_down_exps.weight")  # (512, 2560, 640)
    packed.section(
        b"".join(
            tile_q4(
                *quantize_affine(down_exps[i], group=GROUP),
                storage_n=EXPERT_STORAGE_N,
                group=GROUP,
            )
            for i in range(512)
        )
    )

    print("  Packing shared expert...")
    shexp_gate = reader.read_tensor("blk.0.ffn_gate_shexp.weight")  # (640, 2560)
    packed.section(
        tile_q4(
            *quantize_affine(shexp_gate, group=GROUP),
            storage_n=EXPERT_STORAGE_N,
            group=GROUP,
        )
    )

    shexp_up = reader.read_tensor("blk.0.ffn_up_shexp.weight")  # (640, 2560)
    packed.section(
        tile_q4(
            *quantize_affine(shexp_up, group=GROUP),
            storage_n=EXPERT_STORAGE_N,
            group=GROUP,
        )
    )

    shexp_down = reader.read_tensor("blk.0.ffn_down_shexp.weight")  # (2560, 640)
    packed.section(
        tile_q4(
            *quantize_affine(shexp_down, group=GROUP),
            storage_n=EXPERT_STORAGE_N,
            group=GROUP,
        )
    )

    shexp_gate_inp = reader.read_tensor("blk.0.ffn_gate_inp_shexp.weight")  # (2560,)
    padded = np.zeros((STORAGE_N, shexp_gate_inp.shape[0]), dtype=np.float32)
    padded[0] = shexp_gate_inp
    out, inp = padded.shape
    groups = inp // GROUP
    blocks = padded.reshape(out, groups, GROUP)
    low = blocks.min(axis=2)
    high = blocks.max(axis=2)
    anchor_low = np.abs(low) > np.abs(high)
    bias = np.where(anchor_low, low, high)
    other = np.where(anchor_low, high, low)
    scale = (other - bias) / 255.0
    scale_bits = to_bf16(scale)
    bias_bits = to_bf16(bias)
    stored_scale = from_bf16(scale_bits)[..., None]
    stored_bias = from_bf16(bias_bits)[..., None]
    safe = np.where(stored_scale == 0, 1.0, stored_scale)
    codes = np.clip(np.rint((blocks - stored_bias) / safe), 0, 255).astype(np.uint8)
    packed.section(codes.tobytes() + scale_bits.tobytes() + bias_bits.tobytes())

    total_bytes = packed.finish()
    print(f"Finished /tmp/test_layer-0.bin: {total_bytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    write_test_layer0()
