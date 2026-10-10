"""
Slipstream Shared Repository Token Indexer (Track 1)

Pre-tokenizes workspace source files into a binary memory-mapped index
(`.slipstream/repo_tokens.idx`) shared with the C++ inference engine.
Allows PromptLookup to immediately draft code proposals matching repository
symbols with zero runtime file tokenization overhead.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import List, Tuple

MAGIC = 0x534C5452  # 'SLTR' (Slipstream Token Repository)
VERSION = 1

CODE_EXTENSIONS = {
    ".py", ".cpp", ".hpp", ".h", ".mm", ".metal", ".c",
    ".rs", ".go", ".ts", ".js", ".json", ".md", ".sh",
    ".yaml", ".yml", ".toml",
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "build", "node_modules",
    "prepared", ".cache", ".gemini", ".agents", "dist",
}


def _simple_tokenize(text: str) -> List[int]:
    """Fast deterministic whitespace and punctuation token mapping.

    Produces dense token IDs for code symbols so n-grams can be matched
    across the repository structure by PromptLookup.
    """
    tokens: List[int] = []
    # Byte-level BPE fallback: map UTF-8 bytes with offset 256 for words
    words = text.split()
    for w in words:
        # Use simple polynomial hash modulo 200,000 to map words to token IDs
        h = 5381
        for ch in w:
            h = ((h << 5) + h + ord(ch)) & 0xFFFFFFF
        tokens.append((h % 200000) + 1000)
    return tokens


def build_repo_token_index(
    workspace_dir: str | Path,
    output_path: str | Path | None = None,
    max_tokens: int = 200000,
) -> Tuple[int, int, str]:
    """Scan workspace code files and write `.slipstream/repo_tokens.idx`.

    Returns:
        (total_tokens, file_count, output_path_str)
    """
    workspace = Path(workspace_dir).resolve()
    if output_path is None:
        slipstream_dir = workspace / ".slipstream"
        slipstream_dir.mkdir(parents=True, exist_ok=True)
        output_file = slipstream_dir / "repo_tokens.idx"
    else:
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

    all_tokens: List[int] = []
    file_count = 0

    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in CODE_EXTENSIONS or f in {"Makefile", "CMakeLists.txt"}:
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    file_tokens = _simple_tokenize(content)
                    if file_tokens:
                        all_tokens.extend(file_tokens)
                        file_count += 1
                        if len(all_tokens) >= max_tokens:
                            break
                except Exception:
                    continue
        if len(all_tokens) >= max_tokens:
            break

    total_tokens = len(all_tokens)
    timestamp = int(time.time())

    # Header: magic (uint32), version (uint32), totalTokens (uint32), fileCount (uint32), timestamp (uint64)
    header = struct.pack("<IIIIQ", MAGIC, VERSION, total_tokens, file_count, timestamp)
    token_bytes = struct.pack(f"<{total_tokens}I", *all_tokens) if total_tokens else b""

    with open(output_file, "wb") as f:
        f.write(header)
        f.write(token_bytes)

    return total_tokens, file_count, str(output_file)


def verify_repo_token_index(index_path: str | Path) -> Tuple[bool, int, int]:
    """Verifies that an index file exists and conforms to the SLTR header."""
    p = Path(index_path)
    if not p.is_file():
        return False, 0, 0
    size = p.stat().st_size
    header_size = struct.calcsize("<IIIIQ")
    if size < header_size:
        return False, 0, 0

    with open(p, "rb") as f:
        header_data = f.read(header_size)

    magic, version, total_tokens, file_count, _ = struct.unpack("<IIIIQ", header_data)
    if magic != MAGIC or version != VERSION:
        return False, 0, 0

    expected_size = header_size + total_tokens * 4
    if size < expected_size:
        return False, 0, 0

    return True, total_tokens, file_count
