"""Native in-process tools for the co-located Slipstream Fast Agent Harness."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class FileEditor:
    """Windowed file reader and surgical block patcher."""

    @staticmethod
    def view_file(path: str | Path, start_line: int = 1, line_count: int = 200) -> str:
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), start_idx + line_count)
        sliced = lines[start_idx:end_idx]
        output = [f"=== {target.name} (Lines {start_idx + 1}–{end_idx} of {len(lines)}) ==="]
        for idx, line in enumerate(sliced, start=start_idx + 1):
            output.append(f"{idx:4d} | {line}")
        return "".join(output)

    @staticmethod
    def verify_chunk(path: str | Path, target_chunk: str) -> dict[str, Any]:
        """Pre-flight chunk probe in < 0.1ms: returns exists, unique, and suggestions."""
        target = Path(path).resolve()
        if not target.is_file():
            return {"exists": False, "unique": False, "error": f"File not found: {path}"}
        content = target.read_text(encoding="utf-8", errors="replace")
        if target_chunk in content:
            count = content.count(target_chunk)
            return {"exists": True, "unique": (count == 1), "count": count, "error": None if count == 1 else "matches multiple locations"}
        norm_content = content.replace("\r\n", "\n")
        norm_target = target_chunk.replace("\r\n", "\n")
        if norm_target in norm_content:
            count = norm_content.count(norm_target)
            return {"exists": True, "unique": (count == 1), "count": count, "error": None if count == 1 else "matches multiple locations"}
        return {"exists": False, "unique": False, "error": "target_chunk not found in file"}

    @staticmethod
    def replace_file_content(
        path: str | Path, target_chunk: str, replacement_chunk: str
    ) -> bool:
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        content = target.read_text(encoding="utf-8", errors="replace")
        
        # 1. Exact match
        if target_chunk in content:
            if content.count(target_chunk) > 1:
                raise ValueError("target_chunk matches multiple locations; must be unique")
            updated = content.replace(target_chunk, replacement_chunk, 1)
            target.write_text(updated, encoding="utf-8")
            return True

        # 2. Line ending normalization match (\r\n -> \n)
        norm_content = content.replace("\r\n", "\n")
        norm_target = target_chunk.replace("\r\n", "\n")
        if norm_target in norm_content:
            if norm_content.count(norm_target) > 1:
                raise ValueError("target_chunk matches multiple locations; must be unique")
            updated = norm_content.replace(norm_target, replacement_chunk.replace("\r\n", "\n"), 1)
            target.write_text(updated, encoding="utf-8")
            return True

        # 3. Indentation & whitespace-insensitive match via line-by-line comparison
        target_lines = [line.strip() for line in norm_target.splitlines() if line.strip()]
        if target_lines:
            content_lines = norm_content.splitlines()
            best_idx = -1
            match_len = len(target_lines)
            for i in range(len(content_lines) - match_len + 1):
                window = [line.strip() for line in content_lines[i : i + match_len] if line.strip()]
                if window == target_lines:
                    if best_idx != -1:
                        raise ValueError("target_chunk matches multiple locations under whitespace normalization")
                    best_idx = i

            if best_idx != -1:
                orig_slice = "\n".join(content_lines[best_idx : best_idx + match_len])
                updated = norm_content.replace(orig_slice, replacement_chunk.replace("\r\n", "\n"), 1)
                target.write_text(updated, encoding="utf-8")
                return True

        # 4. RapidFuzz fuzzy anchor fallback if available
        try:
            import rapidfuzz.fuzz as fuzz

            target_clean = norm_target.strip()
            # Sliding window over content lines to find best match ratio
            content_lines = norm_content.splitlines()
            target_line_count = max(1, len(norm_target.splitlines()))
            best_ratio = 0.0
            best_window_idx = -1

            for i in range(max(1, len(content_lines) - target_line_count + 1)):
                candidate = "\n".join(content_lines[i : i + target_line_count]).strip()
                ratio = fuzz.ratio(target_clean, candidate)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_window_idx = i

            if best_ratio >= 92.0 and best_window_idx != -1:
                matched_block = "\n".join(content_lines[best_window_idx : best_window_idx + target_line_count])
                updated = norm_content.replace(matched_block, replacement_chunk.replace("\r\n", "\n"), 1)
                target.write_text(updated, encoding="utf-8")
                return True
        except ImportError:
            pass

        raise ValueError("target_chunk not found in file")


    @staticmethod
    def write_file(path: str | Path, content: str, overwrite: bool = True) -> bool:
        target = Path(path).resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True


class RipgrepSearcher:
    """Fast directory search using ripgrep."""

    @staticmethod
    def search(pattern: str, root_dir: str | Path = ".", max_matches: int = 50) -> list[str]:
        root = str(Path(root_dir).resolve())
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", str(max_matches), pattern, root]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return res.stdout.splitlines()[:max_matches]
        except FileNotFoundError:
            # Fallback to basic python search if rg not installed on system path
            matches = []
            for p in Path(root).rglob("*"):
                if p.is_file() and not p.name.startswith("."):
                    try:
                        for l_num, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                            if pattern in line:
                                matches.append(f"{p}:{l_num}:{line}")
                                if len(matches) >= max_matches:
                                    return matches
                    except Exception:
                        pass
            return matches


class WebResearcher:
    """Lean online research tools: DuckDuckGo search and trafilatura markdown fetching with caching."""

    @staticmethod
    def search(query: str, max_results: int = 5) -> str:
        """Perform headless web search via DuckDuckGo and return concise text snippets."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No search results found."
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                href = r.get("href", "")
                body = r.get("body", "")
                formatted.append(f"{i}. [{title}]({href})\n   {body}")
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Web search error: {e}"

    @staticmethod
    def _get_cache_db() -> Path:
        cache_dir = Path.home() / ".slipstream" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "web.db"
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS page_cache (url TEXT PRIMARY KEY, content TEXT, timestamp REAL)"
            )
        return db_path

    @staticmethod
    def fetch_url(url: str, max_chars: int = 6000) -> str:
        """Fetch URL and extract clean markdown content without HTML/JS bloat, backed by SQLite cache."""
        import sqlite3
        import time

        db_path = WebResearcher._get_cache_db()
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute("SELECT content, timestamp FROM page_cache WHERE url = ?", (url,))
                row = cur.fetchone()
                if row:
                    cached_text, ts = row
                    # 24-hour TTL
                    if (time.time() - ts) < 86400:
                        if len(cached_text) > max_chars:
                            return cached_text[:max_chars] + f"\n\n... [cached content truncated at {max_chars} chars] ..."
                        return cached_text
        except Exception:
            pass

        import urllib.request
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            
            # Use trafilatura if installed for high-quality clean markdown
            extracted = None
            try:
                import trafilatura
                extracted = trafilatura.extract(html, output_format="txt", include_links=True)
            except ImportError:
                pass
            
            if not extracted:
                import re
                cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
                cleaned = re.sub(r"<[^>]+>", " ", cleaned)
                extracted = re.sub(r"\s+", " ", cleaned).strip()

            # Store in SQLite cache
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO page_cache (url, content, timestamp) VALUES (?, ?, ?)",
                        (url, extracted, time.time()),
                    )
            except Exception:
                pass

            if len(extracted) > max_chars:
                return extracted[:max_chars] + f"\n\n... [content truncated at {max_chars} chars] ..."
            return extracted
        except Exception as e:
            return f"Error fetching URL {url}: {e}"


class RepoMapper:
    """Compact in-memory repository structure and symbol summarizer."""

    @staticmethod
    def _extract_symbols(file_path: Path, max_symbols: int = 4) -> list[str]:
        """Extract top-level definitions from source files in < 1ms."""
        symbols = []
        try:
            suffix = file_path.suffix
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                stripped = line.strip()
                if suffix == ".py":
                    if stripped.startswith("class ") or stripped.startswith("def "):
                        name = stripped.split("(")[0].split(":")[0]
                        symbols.append(name)
                elif suffix in (".cpp", ".hpp", ".h", ".mm"):
                    if (stripped.startswith("struct ") or stripped.startswith("class ")) and "{" in stripped or ";" in stripped:
                        name = stripped.split("{")[0].split(":")[0].strip()
                        symbols.append(name)
                if len(symbols) >= max_symbols:
                    break
        except Exception:
            pass
        return symbols

    @staticmethod
    def get_repo_map(root_dir: str | Path = ".", max_files: int = 40) -> str:
        """Generate a concise structural summary of the workspace with key symbols."""
        root = Path(root_dir).resolve()
        summary = [f"=== Workspace Structure: {root.name} ==="]
        file_count = 0
        
        # Traverse tracked / important source files
        ignore_dirs = {".git", ".venv", "build", "dist", "__pycache__", "node_modules", ".cache", ".idea", ".vscode"}
        for p in sorted(root.rglob("*")):
            if any(part in ignore_dirs for part in p.parts):
                continue
            if p.is_file() and p.suffix in (".py", ".cpp", ".hpp", ".h", ".mm", ".m", ".metal", ".sh", ".md", ".json"):
                file_count += 1
                if file_count <= max_files:
                    rel = p.relative_to(root)
                    syms = RepoMapper._extract_symbols(p)
                    sym_str = f" ({', '.join(syms)})" if syms else ""
                    summary.append(f"  {rel}{sym_str}")
        
        if file_count > max_files:
            summary.append(f"  ... and {file_count - max_files} more files")
        return "\n".join(summary)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command inside the persistent warm PTY environment (zsh). Preserves working directory and environment variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command line string to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View lines of a file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                    "start_line": {
                        "type": "integer",
                        "description": "1-based starting line number (default: 1).",
                        "default": 1,
                    },
                    "line_count": {
                        "type": "integer",
                        "description": "Number of lines to read (default: 200).",
                        "default": 200,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_file_content",
            "description": "Surgically replace a unique block of text in a file with new content. Features indentation-insensitive healing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "target_chunk": {
                        "type": "string",
                        "description": "Exact text chunk to replace (must match uniquely).",
                    },
                    "replacement_chunk": {
                        "type": "string",
                        "description": "Replacement text chunk.",
                    },
                },
                "required": ["path", "target_chunk", "replacement_chunk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path to create/overwrite."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search directory files for pattern using ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or substring to find."},
                    "root_dir": {
                        "type": "string",
                        "description": "Directory to search (default: .).",
                        "default": ".",
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum matches to return (default: 50).",
                        "default": 50,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a fast, headless web search via DuckDuckGo for documentation, APIs, or bug reports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of ranked search results to return (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_todo",
            "description": "Manage the multi-step execution plan / todo checklist. Call this to initialize, update, or check off tasks during complex coding workflows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "update", "get"],
                        "description": "Action to perform: 'set' replaces the checklist with new tasks, 'update' modifies a task's status, 'get' retrieves current checklist.",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of task descriptions when action is 'set'.",
                    },
                    "task_id": {
                        "type": "integer",
                        "description": "1-based task index when action is 'update'.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                        "description": "New status for task when action is 'update'.",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


class TodoManager:
    """Deterministic in-memory checklist state machine for tracking multi-step agent execution."""

    def __init__(self):
        self.tasks: list[dict[str, Any]] = []

    def set_tasks(self, descriptions: list[str]) -> list[dict[str, Any]]:
        self.tasks = [
            {"id": idx + 1, "task": desc.strip(), "status": "pending"}
            for idx, desc in enumerate(descriptions)
            if desc.strip()
        ]
        return self.tasks

    def update_task(self, task_id: int, status: str) -> dict[str, Any] | None:
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = status
                return t
        return None

    def get_next_pending_or_in_progress(self) -> dict[str, Any] | None:
        """Return the first in_progress task, or first pending task if none is in progress."""
        for t in self.tasks:
            if t["status"] == "in_progress":
                return t
        for t in self.tasks:
            if t["status"] == "pending":
                return t
        return None

    def get_tasks(self) -> list[dict[str, Any]]:
        return list(self.tasks)

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def active_summary(self) -> str:
        """Compact string representation for system/turn prompt injection (< 50 tokens)."""
        if not self.tasks:
            return ""
        lines = ["[Current Plan / Todo List]:"]
        for t in self.tasks:
            marker = "✔" if t["status"] == "completed" else ("❯" if t["status"] == "in_progress" else "○")
            lines.append(f"  {marker} {t['id']}. {t['task']} ({t['status']})")
        return "\n".join(lines)


def compress_tool_output(
    output: str, max_chars: int = 30000, preserve_tail_lines: int = 30
) -> str:
    """Compress bulky tool outputs with tail preservation.

    Preserves the last N lines (exit codes, test verdicts, summary lines)
    while truncating intermediate stdout/stderr to prevent KV-cache explosion.
    """
    if len(output) <= max_chars:
        return output
    lines = output.splitlines(keepends=True)
    if len(lines) <= preserve_tail_lines:
        return output[:max_chars] + "\n... [output truncated]"
    head_budget = max_chars // 2
    tail_lines = lines[-preserve_tail_lines:]
    tail_text = "".join(tail_lines)
    head_text = output[:head_budget]
    omitted = len(lines) - preserve_tail_lines
    return f"{head_text}\n... [{omitted} lines compressed] ...\n{tail_text}"
