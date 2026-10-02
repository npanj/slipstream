import os
import struct
import ctypes
import subprocess
import tempfile
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEQUANT_SOURCE = Path(__file__).with_name("fast_dequant.c")
DEQUANT_LIB = REPO_ROOT / "build/libslipstream-dequant.dylib"
DEQUANT_TYPES = ("q4_0", "q4_1", "q5_0", "q8_0", "q4_K", "q6_K")
# ggml type id -> (elements per block, bytes per block)
GGML_BLOCKS = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    8: (32, 34),  # Q8_0
    12: (256, 144),  # Q4_K
    14: (256, 210),  # Q6_K
    30: (1, 2),  # BF16
}


def _build_dequant_lib():
    """Compile fast_dequant.c into build/ when missing or older than the source."""
    if DEQUANT_LIB.exists() and DEQUANT_LIB.stat().st_mtime >= DEQUANT_SOURCE.stat().st_mtime:
        return DEQUANT_LIB
    DEQUANT_LIB.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the target and rename, so concurrent workers never load a partial file.
    fd, tmp = tempfile.mkstemp(dir=DEQUANT_LIB.parent, suffix=".dylib")
    os.close(fd)
    try:
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "clang", "-O3", "-shared", "-fPIC",
             "-o", tmp, str(DEQUANT_SOURCE)],
            check=True,
        )
        os.replace(tmp, DEQUANT_LIB)
    except (OSError, subprocess.CalledProcessError) as error:
        os.unlink(tmp)
        raise RuntimeError(
            f"Cannot build {DEQUANT_LIB} from {DEQUANT_SOURCE}: {error}. "
            "Install the Xcode command line tools, or set SLIPSTREAM_GGML_LIB "
            "to a libggml-base dylib."
        ) from None
    return DEQUANT_LIB


def _load_dequantizers():
    """Return {type: fn(src, dst, count)} from libggml-base or the bundled kernels."""
    argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    if ggml_path := os.environ.get("SLIPSTREAM_GGML_LIB"):
        if not os.path.exists(ggml_path):
            raise RuntimeError(f"SLIPSTREAM_GGML_LIB points to a missing file: {ggml_path}")
        lib = ctypes.CDLL(ggml_path)
        names = {t: f"dequantize_row_{t}" for t in DEQUANT_TYPES}
    else:
        lib = ctypes.CDLL(str(_build_dequant_lib()))
        names = {t: f"dequant_{t.lower()}" for t in DEQUANT_TYPES}
    functions = {}
    for ggml_type, name in names.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None
        functions[ggml_type] = fn
    return functions


DEQUANT = _load_dequantizers()

class MultiShardGgufReader:
    def __init__(self, model_dir_or_files, sidecars=None):
        if isinstance(model_dir_or_files, (str, Path)):
            p = Path(model_dir_or_files)
            if p.is_dir():
                shard_paths = sorted(p.glob("*.gguf"))
            else:
                shard_paths = [p]
        else:
            shard_paths = [Path(f) for f in model_dir_or_files]

        if sidecars:
            if isinstance(sidecars, (str, Path)):
                sidecars = [sidecars]
            for s in sidecars:
                sp = Path(s)
                if sp.exists() and sp not in shard_paths:
                    shard_paths.append(sp)

        self.shards = []
        self.tensors = {}
        self.metadata = {}

        for shard_idx, shard_path in enumerate(shard_paths):
            fd = open(shard_path, "rb")
            magic = fd.read(4)
            if magic != b"GGUF":
                fd.close()
                continue
            ver, n_tensors, n_kv = struct.unpack("<IQQ", fd.read(20))
            
            # Read KV metadata (on shard 0 or all shards)
            for _ in range(n_kv):
                klen = struct.unpack("<Q", fd.read(8))[0]
                k = fd.read(klen).decode("latin1", errors="replace")
                vtype = struct.unpack("<I", fd.read(4))[0]
                if vtype in (0, 1, 7):
                    val = fd.read(1)
                elif vtype in (2, 3):
                    val = fd.read(2)
                elif vtype in (4, 5, 6):
                    val = fd.read(4)
                elif vtype in (10, 11, 12):
                    val = fd.read(8)
                elif vtype == 8:
                    slen = struct.unpack("<Q", fd.read(8))[0]
                    val = fd.read(slen).decode("latin1", errors="replace")
                elif vtype == 9:
                    etype, count = struct.unpack("<IQ", fd.read(12))
                    sz = 1 if etype in (0,1,7) else 2 if etype in (2,3) else 4 if etype in (4,5,6) else 8
                    if etype == 8:
                        val = []
                        for _ in range(count):
                            slen = struct.unpack("<Q", fd.read(8))[0]
                            val.append(fd.read(slen).decode("latin1", errors="replace"))
                    else:
                        val = fd.read(count * sz)
                self.metadata[k] = val

            # Read tensor metadata
            for _ in range(n_tensors):
                nlen = struct.unpack("<Q", fd.read(8))[0]
                name = fd.read(nlen).decode("latin1", errors="replace")
                ndims = struct.unpack("<I", fd.read(4))[0]
                dims = struct.unpack(f"<{ndims}Q", fd.read(ndims * 8))
                ttype, toffset = struct.unpack("<IQ", fd.read(12))
                self.tensors[name] = {
                    "shard_idx": shard_idx,
                    "dims": dims,
                    "type": ttype,
                    "offset": toffset
                }

            pos = fd.tell()
            align = 32
            pad = (align - pos % align) % align
            data_offset = pos + pad

            self.shards.append({
                "path": shard_path,
                "fd": fd,
                "data_offset": data_offset
            })

    def has(self, name):
        return name in self.tensors

    def read_tensor(self, name):
        meta = self._meta(name)
        shape = tuple(reversed(meta["dims"]))
        return self._decode(name, 0, shape)

    def read_rows(self, name, start, count):
        """Rows [start, start + count) of a tensor viewed as 2-D, without reading the rest.

        A 3-D tensor such as ffn_gate_exps flattens to [experts * rows, width], so
        expert e is rows [e * rows, (e + 1) * rows).
        """
        meta = self._meta(name)
        width, rows = meta["dims"][0], int(np.prod(meta["dims"][1:]))
        if start < 0 or start + count > rows:
            raise IndexError(f"rows {start}..{start + count} outside {name} ({rows} rows)")
        block, size = GGML_BLOCKS[meta["type"]]
        if width % block:
            raise ValueError(f"{name} rows do not split into whole {block}-element blocks")
        return self._decode(name, start * (width // block) * size, (count, width))

    def read_expert(self, name, expert):
        """One expert's [rows, width] matrix of a [experts, rows, width] tensor."""
        rows = self._meta(name)["dims"][1]
        return self.read_rows(name, expert * rows, rows)

    def _meta(self, name):
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found in GGUF shards")
        return self.tensors[name]

    def _decode(self, name, skip, shape):
        """Read and dequantize the elements of shape starting skip bytes into name."""
        meta = self.tensors[name]
        shard = self.shards[meta["shard_idx"]]
        fd = shard["fd"]
        ttype = meta["type"]
        count = int(np.prod(shape))

        fd.seek(shard["data_offset"] + meta["offset"] + skip)
        if ttype == 0: # F32
            return np.frombuffer(fd.read(count * 4), dtype="<f4").reshape(shape)
        elif ttype == 1: # F16
            return np.frombuffer(fd.read(count * 2), dtype="<f2").astype(np.float32).reshape(shape)
        elif ttype == 30: # BF16
            u16 = np.frombuffer(fd.read(count * 2), dtype="<u2")
            return ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(shape)

        out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.c_void_p)

        if ttype == 2: # Q4_0
            raw = fd.read(count // 32 * 18)
            DEQUANT["q4_0"](raw, ptr, count)
        elif ttype == 3: # Q4_1
            raw = fd.read(count // 32 * 20)
            DEQUANT["q4_1"](raw, ptr, count)
        elif ttype == 6: # Q5_0
            raw = fd.read(count // 32 * 22)
            DEQUANT["q5_0"](raw, ptr, count)
        elif ttype == 8: # Q8_0
            raw = fd.read(count // 32 * 34)
            DEQUANT["q8_0"](raw, ptr, count)
        elif ttype == 12: # Q4_K
            raw = fd.read(count // 256 * 144)
            DEQUANT["q4_K"](raw, ptr, count)
        elif ttype == 14: # Q6_K
            raw = fd.read(count // 256 * 210)
            DEQUANT["q6_K"](raw, ptr, count)
        else:
            raise ValueError(f"Unsupported ggml type {ttype} for tensor {name}")

        return out

    def close(self):
        for s in self.shards:
            s["fd"].close()
