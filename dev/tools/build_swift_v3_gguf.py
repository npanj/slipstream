#!/usr/bin/env python3
"""Build Swift-Qwen3.8-Flash-Next-V3 by splicing Q8_0 resident tensors into Swift Q4_0 base.

Recipe:
- Base: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF (Q4_0, 3 shards, ~93.7 GiB)
  - 512 routed MoE experts stay at Q4_0 (streamed from SSD).
  - PLE table stays at Q4_0.
- Donor: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF (Q8_0, 686 tensors, ~4.77 GiB)
  - output.weight -> Q8_0
  - token_embd.weight -> Q8_0
  - attention projections (attn_qkv, attn_gate, attn_output) -> Q8_0
  - hyper-connection mixers (hc_*) -> Q8_0
  - linear attention projections (ssm_out) -> Q8_0
  - shared experts (shexp) -> Q8_0

Memory / Disk Strategy:
- Streams Q8_0 donor tensors via HTTP range requests directly (no need to download 175 GB Q8_0).
- Processes base shards 1, 2, 3 sequentially.
- Deletes each temporary raw base shard after writing its spliced V3 shard.
- Peak temp disk overhead <= 42 GiB; final package size ~95.5 GiB.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path.home() / "models"
OUT_DIR = MODELS_DIR / "swift-qwen38-flash-next-v3"
STAGING_DIR = MODELS_DIR / "staging-swift-v3"
LOG_FILE = MODELS_DIR / "logs/build-swift-v3.log"

REPO_ID = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF"

GROUPS = {
    "output": r"^output\.weight$",
    "token_embd": r"^token_embd\.weight$",
    "attn": r"^blk\.\d+\.attn_(q|k|v|qkv|gate|output)\.weight$",
    "hc": r"^blk\.\d+\.hc_\w+\.weight$",
    "ssm_out": r"^blk\.\d+\.ssm_out\.weight$",
    "shexp": r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$",
}

QUANT = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 32, 18),
}

SCALAR = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def tensor_nbytes(dims, type_id):
    n = 1
    for d in dims:
        n *= d
    _, blk, size = QUANT[type_id]
    if n % blk:
        raise ValueError(f"{n} elements is not a multiple of block {blk}")
    return n // blk * size


class Header:
    def __init__(self, path, buf):
        self.path = path
        self.buf = buf
        self.o = 0
        if self._raw(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            self.kv[k] = self._value(self._u32())
        self.tensors = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            type_pos = self.o
            type_id = self._u32()
            off_pos = self.o
            rel = self._u64()
            self.tensors.append(
                {
                    "name": name,
                    "dims": dims,
                    "type": type_id,
                    "rel": rel,
                    "type_pos": type_pos,
                    "off_pos": off_pos,
                    "nbytes": tensor_nbytes(dims, type_id),
                }
            )
        self.align = self.kv.get("general.alignment", 32)
        self.header_end = self.o
        self.data_start = (self.o + self.align - 1) // self.align * self.align
        for t in self.tensors:
            t["file_offset"] = self.data_start + t["rel"]

    def _raw(self, n):
        if self.o + n > len(self.buf):
            raise EOFError(f"{self.path}: header truncated")
        v = self.buf[self.o : self.o + n]
        self.o += n
        return v

    def _u32(self):
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._raw(8))[0]

    def _str(self):
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._u32(), self._u64()
            if et in (8, 9):
                return [self._value(et) for _ in range(n)]
            self._raw(n * struct.calcsize(SCALAR[et]))
            return f"<array {n} x t{et}>"
        f = SCALAR[t]
        return struct.unpack("<" + f, self._raw(struct.calcsize(f)))[0]


def read_header(path, probe=1 << 20):
    size = os.path.getsize(path)
    while True:
        with open(path, "rb") as f:
            buf = f.read(min(probe, size))
        try:
            return Header(path, buf)
        except EOFError:
            if probe >= size:
                raise
            probe *= 4


def copy_range(src, src_off, n, dst, chunk=32 << 20):
    with open(src, "rb") as f:
        f.seek(src_off)
        left = n
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise EOFError(f"{src}: short read at {src_off}, {left} bytes left")
            dst.write(b)
            left -= len(b)


def fetch_header_remote(url: str, probe_bytes: int = 30000000) -> Header:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{probe_bytes}", "User-Agent": "hf"}
    )
    with urllib.request.urlopen(req) as resp:
        buf = resp.read()
    return Header(url, buf)


def download_tensor_range(url: str, offset: int, nbytes: int, dst_path: Path):
    if dst_path.exists() and dst_path.stat().st_size == nbytes:
        return
    end_byte = offset + nbytes - 1
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={offset}-{end_byte}", "User-Agent": "hf"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) != nbytes:
                    raise IOError(f"Expected {nbytes} bytes but got {len(data)}")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dst_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    f.write(data)
                tmp_path.rename(dst_path)
                return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def build_donor_index() -> dict[str, dict]:
    log("1/4: Indexing remote Q8_0 donor tensors across shards 1, 3, 4, 5...")
    donor_map = {}
    for s_idx in [1, 3, 4, 5]:
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/Q8_0/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-0000{s_idx}-of-00005.gguf"
        h = fetch_header_remote(url)
        for t in h.tensors:
            name = t["name"]
            hit = next(
                (g for g, pat in GROUPS.items() if re.fullmatch(pat, name)), None
            )
            if hit:
                donor_map[name] = {
                    "url": url,
                    "offset": t["file_offset"],
                    "nbytes": t["nbytes"],
                    "type": t["type"],
                    "dims": t["dims"],
                    "group": hit,
                }
    log(
        f"  Indexed {len(donor_map)} resident donor tensors (~{sum(d['nbytes'] for d in donor_map.values()) / 1024**3:.2f} GiB)"
    )
    return donor_map


def process_shard(shard_num: int, total_shards: int, donor_map: dict[str, dict]):
    shard_str = f"{shard_num:05d}-of-{total_shards:05d}"
    out_name = f"Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-{shard_str}.gguf"
    out_path = OUT_DIR / out_name

    if out_path.exists() and out_path.stat().st_size > 10 * 1024**3:
        log(
            f"Shard {shard_num}/{total_shards} already completed at {out_path} ({out_path.stat().st_size / 1024**3:.2f} GiB). Skipping."
        )
        return

    base_fn = f"Q4_0/Swift-1.5-Qwen3.8-Flash-Next-Q4_0-{shard_str}.gguf"
    log(f"\n--- Processing Shard {shard_num}/{total_shards} ---")
    log(f"  Downloading raw base shard {base_fn}...")
    t0 = time.time()
    raw_path_str = hf_hub_download(
        repo_id=REPO_ID,
        filename=base_fn,
        local_dir=str(STAGING_DIR / "base"),
        local_dir_use_symlinks=False,
    )
    raw_path = Path(raw_path_str)
    dl_elapsed = time.time() - t0
    log(
        f"  Downloaded base shard ({raw_path.stat().st_size / 1024**3:.2f} GiB in {dl_elapsed:.1f}s)"
    )

    # Read base header
    log("  Parsing base shard header...")
    h = read_header(str(raw_path))

    # Identify tensors to substitute
    subs = {}
    download_tasks = []
    tensor_staging = STAGING_DIR / f"tensors_s{shard_num}"
    tensor_staging.mkdir(parents=True, exist_ok=True)

    for t in h.tensors:
        name = t["name"]
        if name in donor_map:
            d = donor_map[name]
            dst_t_path = tensor_staging / f"{name}.bin"
            subs[name] = {
                "src": dst_t_path,
                "off": 0,
                "type": d["type"],
                "nbytes": d["nbytes"],
                "url": d["url"],
                "donor_off": d["offset"],
            }
            download_tasks.append((d["url"], d["offset"], d["nbytes"], dst_t_path))

    log(
        f"  Found {len(subs)} tensors in this shard to upgrade to Q8_0 (~{sum(s['nbytes'] for s in subs.values()) / 1024**2:.1f} MB)"
    )

    # Download donor tensors concurrently
    if download_tasks:
        log(
            f"  Streaming {len(download_tasks)} Q8_0 donor tensors via HTTP range requests..."
        )
        t_dl0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [
                ex.submit(download_tensor_range, url, off, nb, p)
                for url, off, nb, p in download_tasks
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()
        log(f"  All donor tensors downloaded in {time.time() - t_dl0:.1f}s")

    # Compute new layout & patch header
    hdr = bytearray(h.buf[: h.data_start])
    rel = 0
    layout = []

    for t in h.tensors:
        name = t["name"]
        s = subs.get(name)
        new_type = s["type"] if s else t["type"]
        new_nb = s["nbytes"] if s else t["nbytes"]

        struct.pack_into("<I", hdr, t["type_pos"], new_type)
        struct.pack_into("<Q", hdr, t["off_pos"], rel)

        src_file = s["src"] if s else raw_path
        src_offset = s["off"] if s else t["file_offset"]

        layout.append((name, src_file, src_offset, new_nb, rel))
        rel += (new_nb + h.align - 1) // h.align * h.align

    # Write output shard
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tmp")
    log(f"  Writing spliced shard {out_name} (target size: {rel / 1024**3:.2f} GiB)...")
    t_wr0 = time.time()
    with open(tmp_out, "wb") as out_f:
        out_f.write(hdr)
        for idx, (tname, s_path, s_off, nb, r_off) in enumerate(layout):
            copy_range(str(s_path), s_off, nb, out_f)
            pad = (h.align - (nb % h.align)) % h.align
            if pad:
                out_f.write(b"\x00" * pad)

    tmp_out.rename(out_path)
    wr_elapsed = time.time() - t_wr0
    log(
        f"  Shard {shard_num} written successfully in {wr_elapsed:.1f}s ({out_path.stat().st_size / 1024**3:.2f} GiB)"
    )

    # Clean up temporary raw base shard and downloaded tensors
    log("  Reclaiming temporary staging files for this shard...")
    try:
        raw_path.unlink()
    except Exception:
        pass
    for _, _, _, p in download_tasks:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tensor_staging.rmdir()
    except Exception:
        pass
    log(f"  Shard {shard_num} finished. Space reclaimed.")


def main():
    log("=== STARTING SWIFT-QWEN3.8-FLASH-NEXT-V3 BUILD PIPELINE ===")
    total_start = time.time()

    # Preflight check
    log(f"Target directory: {OUT_DIR}")
    donor_map = build_donor_index()

    # Process all 3 shards
    for s_idx in [1, 2, 3]:
        process_shard(s_idx, 3, donor_map)

    # Postflight verification
    log("\n=== POSTFLIGHT VERIFICATION ===")
    shards = sorted(list(OUT_DIR.glob("Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-*.gguf")))
    log(f"Found {len(shards)} generated shards:")
    total_sz = 0
    for p in shards:
        sz = p.stat().st_size
        total_sz += sz
        h = read_header(str(p))
        q8_count = sum(1 for t in h.tensors if t["type"] == 8)
        q4_count = sum(1 for t in h.tensors if t["type"] in (2, 3, 20))
        log(
            f"  {p.name}: {sz / 1024**3:.2f} GiB | {len(h.tensors)} tensors ({q8_count} Q8_0, {q4_count} Q4)"
        )

    log(f"Total package size: {total_sz / 1024**3:.2f} GiB")
    log(f"=== BUILD FINISHED in {(time.time() - total_start) / 60:.1f} min ===")


if __name__ == "__main__":
    sys.exit(main())
