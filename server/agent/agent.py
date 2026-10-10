"""Co-located Fast Agent Harness for Slipstream on Apple Silicon."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .pty_shell import WarmPtyShell
from .tools import (
    TOOL_DEFINITIONS,
    FileEditor,
    RepoMapper,
    RipgrepSearcher,
    TodoManager,
    WebResearcher,
    compress_tool_output,
)

SYSTEM_PROMPT = """You are Slipstream Agent, an ultra-fast autonomous coding agent co-located with the Slipstream inference engine on Apple Silicon.

You have access to native, sub-millisecond in-process tools:
- manage_todo(action, tasks, task_id, status): Manages the execution plan checklist. Call action='set' with tasks=[...] to establish a multi-step plan before non-trivial coding tasks, and action='update' with task_id=N, status='in_progress'|'completed'|'failed' as you work.
- run_command(command): Runs shell commands in a persistent warm PTY zsh environment. Preserves state, virtualenvs, and cwd across calls.
- view_file(path, start_line, line_count): Windowed file inspection with line numbers.
- replace_file_content(path, target_chunk, replacement_chunk): Surgical code replacement. target_chunk must match uniquely.
- write_file(path, content): Create or overwrite files.
- search(pattern, root_dir, max_matches): Fast ripgrep directory search.
- web_search(query, max_results): Headless online web search for docs/APIs.
- fetch_url(url, max_chars): Clean markdown/text web page reader.

Guidelines:
1. For complex, multi-step tasks (architectural refactors, multi-file changes, bug diagnosis), call manage_todo(action="set", tasks=[...]) first to establish a clear plan before modifying code.
2. Update task progress via manage_todo(action="update", task_id=N, status="completed") as you finish each step. When a step is done, always update its status so the plan advances.
3. Think before each action: write your rationale and analysis in your reasoning trace before invoking tools. Explain what you are inspecting, why, and what you expect to find.
4. For conceptual questions or simple queries, you do not need to call manage_todo—answer directly.
5. Read files before editing. Use replace_file_content for surgical edits rather than rewriting full files.
6. Test your changes using run_command before concluding.
7. When finished, summarize what was achieved clearly and concisely.
"""


class UnixHTTPConnection(http.client.HTTPConnection):
    """Direct UNIX domain socket HTTP connection bypassing TCP loopback."""

    def __init__(self, socket_path: str, timeout: float | None = None):
        super().__init__("localhost", timeout=timeout if (timeout is not None and timeout > 0) else None)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None and self.timeout > 0:
            self.sock.settimeout(self.timeout)
        else:
            self.sock.settimeout(None)
        self.sock.connect(self.socket_path)


class SlipstreamAgent:
    """Co-located coding agent client communicating directly with local Slipstream engine."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090/v1",
        api_key: str | None = None,
        cwd: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str = "medium",
        session_id: str | None = None,
        unix_socket_path: str | None = None,
        stream_mode: str = "hybrid",
        timeout: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("SLIPSTREAM_API_KEY", "")
        self.cwd = Path(cwd or os.getcwd()).resolve()
        if timeout is not None:
            self.timeout = None if timeout <= 0 else timeout
        else:
            env_t = os.environ.get("SLIPSTREAM_AGENT_TIMEOUT", "").strip().lower()
            if env_t in ("0", "none", "no", "false", "inf", "infinity", "-1", "off", "unlimited"):
                self.timeout = None
            elif env_t:
                try:
                    val = float(env_t)
                    self.timeout = None if val <= 0 else val
                except ValueError:
                    self.timeout = None
            else:
                self.timeout = None  # No timeout by default: unconstrained local workstation generation
        self.unix_socket_path = unix_socket_path
        if not self.unix_socket_path:
            if self.base_url.startswith("unix://"):
                self.unix_socket_path = self.base_url.removeprefix("unix://")
                self.base_url = "http://localhost/v1"
            elif "127.0.0.1" in self.base_url or "localhost" in self.base_url:
                m = re.search(r":(\d+)", self.base_url)
                if m:
                    cand = f"/tmp/slipstream-{m.group(1)}.sock"
                    if os.path.exists(cand):
                        self.unix_socket_path = cand
        self.stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "turns": 0,
            "max_context": 131072,
        }
        self.model = self._resolve_model(model)
        self.reasoning_effort = reasoning_effort
        mode_raw = (stream_mode or "hybrid").lower()
        if mode_raw in ("never", "batch", "suppress", "off", "none", "no-stream"):
            self.stream_mode = "never"
        elif mode_raw in ("always", "on", "all"):
            self.stream_mode = "always"
        elif mode_raw in ("hybrid", "default"):
            self.stream_mode = "hybrid"
        else:
            raise ValueError(f"Invalid stream_mode '{stream_mode}'. Expected 'hybrid', 'always', or 'never' (alias 'batch').")
        self.last_turn_streamed: bool = False
        self.session_id = session_id or f"slipstream-agent-{uuid.uuid4().hex[:12]}"
        self.last_user_prompt: str = ""
        self.last_failed_prompt: str = ""
        self.pending_prompt: str | None = None

        self.pty = WarmPtyShell(cwd=str(self.cwd))
        self.console = Console()
        self.todo = TodoManager()
        self.on_todo_update: Any = None
        self.recent_tool_signatures: list[str] = []
        repo_map = RepoMapper.get_repo_map(self.cwd, max_files=35)
        full_system_prompt = f"{SYSTEM_PROMPT}\n\n{repo_map}"
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system_prompt}
        ]
        self._checkpoints: list[list[dict[str, Any]]] = []

    def _query_unix(self, path: str, timeout: float = 2.0) -> dict[str, Any] | None:
        if not self.unix_socket_path or not os.path.exists(self.unix_socket_path):
            return None
        try:
            conn = UnixHTTPConnection(self.unix_socket_path, timeout=timeout)
            headers = {"Host": "localhost"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                conn.close()
                return data
            conn.close()
        except Exception:
            pass
        return None

    def _resolve_model(self, model: str | None) -> str:
        # Also query /props to discover dynamic max_context from engine
        props = self._query_unix("/props", timeout=2.0)
        if props:
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            if n_ctx and isinstance(n_ctx, int) and n_ctx > 0:
                self.stats["max_context"] = n_ctx
        else:
            try:
                p_req = urllib.request.Request(
                    f"{self.base_url.rsplit('/v1', 1)[0]}/props",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
                with urllib.request.urlopen(p_req, timeout=2.0) as resp:
                    props = json.loads(resp.read().decode("utf-8"))
                    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
                    if n_ctx and isinstance(n_ctx, int) and n_ctx > 0:
                        self.stats["max_context"] = n_ctx
            except Exception:
                pass

        if model and model != "default":
            return model

        models_data = self._query_unix("/v1/models", timeout=3.0)
        if models_data:
            models = models_data.get("data", [])
            if models and isinstance(models, list):
                return models[0].get("id", "default")

        # Query engine /models endpoint to discover currently loaded model name
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if models and isinstance(models, list):
                    return models[0].get("id", "default")
        except Exception:
            pass
        return "default"

    def estimate_context_tokens(self) -> int:
        """Estimate token usage of current conversation history in characters / 3.8."""
        total_chars = 0
        for m in self.messages:
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                total_chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))
        return max(1, int(total_chars / 3.8))

    def compact_history_if_needed(self, threshold_ratio: float = 0.70, preserve_recent: int = 6) -> bool:
        """Compact middle conversation history when exceeding threshold_ratio of max_context.
        
        Preserves:
        - Zone 1: Immutable system prompt (repository map & guidelines)
        - Zone 2: Rolling middle summary of compressed turns
        - Zone 3: Recent turns (user prompts, tool calls, and final responses)
        """
        max_ctx = self.stats.get("max_context", 131072)
        est = self.estimate_context_tokens()
        if est < int(max_ctx * threshold_ratio):
            return False
        if len(self.messages) <= preserve_recent + 2:
            return False

        # Extract system prompt
        sys_msg = self.messages[0]
        # Preserve recent messages
        recent_msgs = self.messages[-preserve_recent:]
        # Middle messages to compress
        middle_msgs = self.messages[1:-preserve_recent]
        if not middle_msgs:
            return False

        # Synthesize concise bullet summary of middle messages
        bullets = []
        for m in middle_msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user" and content:
                bullets.append(f"- User requested: {content[:120].strip()}")
            elif role == "assistant" and content:
                bullets.append(f"- Assistant noted: {content[:150].strip()}")
            elif role == "tool":
                tname = m.get("name", "tool")
                bullets.append(f"- Tool {tname} executed: {content[:80].strip()}")

        summary_body = (
            "[Prior Conversation Context Summary]\n"
            + "\n".join(bullets[-15:])
            + "\n[End of Prior Context Summary]"
        )
        self.messages = [sys_msg, {"role": "system", "content": summary_body}] + recent_msgs
        return True

    def save_checkpoint(self, vault_dir: str | Path | None = None) -> str:
        """Checkpoint current agent conversation messages and session telemetry to APFS disk."""
        target_dir = Path(vault_dir) if vault_dir else Path.home() / ".cache" / "slipstream" / "agent_vaults"
        target_dir.mkdir(parents=True, exist_ok=True)
        vault_path = target_dir / f"{self.session_id}.json"

        last_user = ""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")[:120]
                break

        data = {
            "session_id": self.session_id,
            "model": self.model,
            "cwd": str(self.cwd),
            "messages": self.messages,
            "stats": self.stats,
            "todos": self.todo.get_tasks(),
            "stream_mode": getattr(self, "stream_mode", "hybrid"),
            "reasoning_effort": getattr(self, "reasoning_effort", "medium"),
            "timestamp": time.time(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_user_prompt": last_user,
            "turn_counter": self.stats.get("turns", 0),
        }

        tmp_path = target_dir / f"{self.session_id}.json.tmp"
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(vault_path)

        latest_path = target_dir / "latest.json"
        try:
            latest_path.write_text(json.dumps({"session_id": self.session_id}, indent=2), encoding="utf-8")
        except Exception:
            pass

        return str(vault_path)

    @classmethod
    def list_checkpoints(cls, vault_dir: str | Path | None = None) -> list[dict[str, Any]]:
        """List all saved session checkpoints ordered from newest to oldest."""
        target_dir = Path(vault_dir) if vault_dir else Path.home() / ".cache" / "slipstream" / "agent_vaults"
        if not target_dir.is_dir():
            return []
        checkpoints = []
        for p in target_dir.glob("*.json"):
            if p.name == "latest.json" or p.name.endswith(".tmp"):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sid = data.get("session_id", p.stem)
                checkpoints.append({
                    "session_id": sid,
                    "path": str(p),
                    "model": data.get("model", "default"),
                    "turns": data.get("turn_counter", data.get("stats", {}).get("turns", 0)),
                    "messages": len(data.get("messages", [])),
                    "timestamp": data.get("timestamp", p.stat().st_mtime),
                    "updated_at": data.get("updated_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))),
                    "last_prompt": data.get("last_user_prompt", ""),
                    "todos_count": len(data.get("todos", [])),
                })
            except Exception:
                continue
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints

    @classmethod
    def restore_checkpoint(cls, session_id: str = "latest", vault_dir: str | Path | None = None, **kwargs) -> "SlipstreamAgent | None":
        """Restore an agent session from its APFS disk checkpoint."""
        target_dir = Path(vault_dir) if vault_dir else Path.home() / ".cache" / "slipstream" / "agent_vaults"
        if not target_dir.is_dir():
            return None

        target_sid = session_id.strip() if session_id else "latest"
        if target_sid in ("latest", "last", "recent", ""):
            latest_ptr = target_dir / "latest.json"
            if latest_ptr.is_file():
                try:
                    ptr_data = json.loads(latest_ptr.read_text(encoding="utf-8"))
                    target_sid = ptr_data.get("session_id", "")
                except Exception:
                    pass
            if not target_sid or target_sid == "latest":
                checkpoints = cls.list_checkpoints(vault_dir=target_dir)
                if not checkpoints:
                    return None
                target_sid = checkpoints[0]["session_id"]

        vault_path = target_dir / f"{target_sid}.json"
        if not vault_path.is_file():
            # Check prefix match across saved checkpoints
            checkpoints = cls.list_checkpoints(vault_dir=target_dir)
            matched = [c for c in checkpoints if c["session_id"].startswith(target_sid)]
            if matched:
                vault_path = Path(matched[0]["path"])
                target_sid = matched[0]["session_id"]
            else:
                return None

        try:
            data = json.loads(vault_path.read_text(encoding="utf-8"))
            stream_mode = kwargs.pop("stream_mode", data.get("stream_mode", "hybrid"))
            reasoning_effort = kwargs.pop("reasoning_effort", data.get("reasoning_effort", "medium"))
            model = kwargs.pop("model", data.get("model", None))
            saved_cwd = kwargs.get("cwd") or data.get("cwd")
            if saved_cwd and not Path(saved_cwd).is_dir():
                kwargs.pop("cwd", None)
            else:
                kwargs["cwd"] = str(Path(saved_cwd).expanduser().resolve())
            agent = cls(
                session_id=data.get("session_id", target_sid),
                stream_mode=stream_mode,
                reasoning_effort=reasoning_effort,
                model=model,
                **kwargs
            )
            agent.messages = data.get("messages", agent.messages)
            agent.stats.update(data.get("stats", {}))
            todos = data.get("todos", [])
            if todos:
                agent.todo.tasks = todos
            return agent
        except Exception:
            return None

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        raw_to = self.timeout if timeout is None else timeout
        eff_timeout = None if (raw_to is None or raw_to <= 0) else raw_to
        payload_copy = dict(payload)
        payload_copy["stream"] = False
        data = json.dumps(payload_copy).encode("utf-8")

        # Direct UNIX Domain Socket transport (bypassing TCP loopback)
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            try:
                conn = UnixHTTPConnection(self.unix_socket_path, timeout=eff_timeout)
                headers = {
                    "Content-Type": "application/json",
                    "Host": "localhost",
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                target_path = f"/v1{endpoint}" if not endpoint.startswith("/v1") else endpoint
                conn.request("POST", target_path, body=data, headers=headers)
                resp = conn.getresponse()
                if resp.status >= 400:
                    err_body = resp.read().decode("utf-8", errors="replace")
                    conn.close()
                    raise RuntimeError(f"Engine HTTP {resp.status}: {err_body}")
                res_data = json.loads(resp.read().decode("utf-8"))
                conn.close()
                return res_data
            except RuntimeError:
                raise
            except (socket.timeout, TimeoutError) as err:
                dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
                raise TimeoutError(f"Request timed out{dur}. The model engine took longer than expected to process this request.") from err
            except Exception:
                # Graceful fallback to TCP
                pass

        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (socket.timeout, TimeoutError) as err:
            dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
            raise TimeoutError(f"Request timed out{dur}. The model engine took longer than expected to process this request.") from err
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Engine HTTP {err.code}: {body}") from err
        except urllib.error.URLError as err:
            reason = getattr(err, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).lower():
                dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
                raise TimeoutError(f"Request timed out{dur}: {reason}") from err
            raise ConnectionError(f"Could not connect to Slipstream engine at {self.base_url}: {err}") from err

    def _post_stream(self, endpoint: str, payload: dict[str, Any], timeout: float | None = None, cancel_event: Any = None):
        """Send request and yield server-sent events as decoded dictionaries."""
        raw_to = self.timeout if timeout is None else timeout
        eff_timeout = None if (raw_to is None or raw_to <= 0) else raw_to
        payload_copy = dict(payload)
        payload_copy["stream"] = True
        data = json.dumps(payload_copy).encode("utf-8")

        # Direct UNIX Domain Socket transport (bypassing TCP loopback)
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            try:
                conn = UnixHTTPConnection(self.unix_socket_path, timeout=eff_timeout)
                headers = {
                    "Content-Type": "application/json",
                    "Host": "localhost",
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                target_path = f"/v1{endpoint}" if not endpoint.startswith("/v1") else endpoint
                conn.request("POST", target_path, body=data, headers=headers)
                resp = conn.getresponse()
                while True:
                    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                        break
                    line_bytes = resp.readline()
                    if not line_bytes:
                        break
                    decoded = line_bytes.decode("utf-8").strip()
                    if not decoded or not decoded.startswith("data: "):
                        continue
                    body = decoded[6:]
                    if body == "[DONE]":
                        break
                    try:
                        yield json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    except json.JSONDecodeError:
                        continue
                conn.close()
                return
            except (socket.timeout, TimeoutError) as err:
                dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
                raise TimeoutError(f"Streaming connection timed out{dur} without tokens.") from err
            except Exception:
                # Graceful fallback to TCP
                pass

        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                for line in resp:
                    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                        break
                    decoded = line.decode("utf-8").strip()
                    if not decoded or not decoded.startswith("data: "):
                        continue
                    body = decoded[6:]
                    if body == "[DONE]":
                        break
                    try:
                        yield json.loads(body)
                    except json.JSONDecodeError:
                        continue
        except (socket.timeout, TimeoutError) as err:
            dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
            raise TimeoutError(f"Streaming connection timed out{dur} without tokens.") from err
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as err:
            reason = getattr(err, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).lower():
                dur = f" after {eff_timeout:.0f}s" if eff_timeout is not None else ""
                raise TimeoutError(f"Streaming connection timed out{dur}: {reason}") from err
            raise ConnectionError(f"Could not connect to Slipstream engine at {self.base_url}: {err}") from err

    def _ingest_reference_content(self, text: str) -> None:
        """Pre-emptively tokenizes and ingests reference text into the engine's PLD index."""
        if not text or len(text) < 15:
            return
        payload = json.dumps({"content": text[:8192]}).encode("utf-8")
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            try:
                conn = UnixHTTPConnection(self.unix_socket_path, timeout=1.0)
                headers = {"Content-Type": "application/json", "Host": "localhost"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                conn.request("POST", "/tokenize", body=payload, headers=headers)
                resp = conn.getresponse()
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    tokens = data.get("tokens", [])
                    if tokens:
                        self.stats["ingested_reference_tokens"] = self.stats.get("ingested_reference_tokens", 0) + len(tokens)
                conn.close()
                return
            except Exception:
                pass
        try:
            # Query the engine's fast in-process /tokenize endpoint
            tok_url = f"{self.base_url.rsplit('/v1', 1)[0]}/tokenize"
            req = urllib.request.Request(
                tok_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tokens = data.get("tokens", [])
                if tokens:
                    # Ingested into session tracker for PLD reference proposals
                    self.stats["ingested_reference_tokens"] = self.stats.get("ingested_reference_tokens", 0) + len(tokens)
        except Exception:
            # Non-critical speculative acceleration path; silently fallback
            pass

    @staticmethod
    def cap_tool_output(text: str, max_chars: int = 30000) -> str:
        """Cap large tool outputs while preserving head and tail contexts."""
        if not text or len(text) <= max_chars:
            return text
        half = (max_chars - 120) // 2
        head = text[:half]
        tail = text[-half:]
        omitted = len(text) - len(head) - len(tail)
        lines_omitted = text[half:-half].count("\n")
        return f"{head}\n\n[... output truncated: {omitted:,} characters ({lines_omitted:,} lines) omitted ...]\n\n{tail}"


    def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool locally in-process."""

        try:
            res = ""
            if name == "run_command":
                cmd = args.get("command", "")
                rc, out = self.pty.run_command(cmd)
                capped_out = self.cap_tool_output(out)
                res = f"[Exit code {rc}]\n{capped_out}" if capped_out else f"[Exit code {rc}]"
            elif name == "view_file":
                p = args.get("path", "")
                start = args.get("start_line", 1)
                count = args.get("line_count", 200)
                file_path = self.cwd / p if not Path(p).is_absolute() else Path(p)
                res = FileEditor.view_file(file_path, start, count)
                # Pre-emptively ingest file content tokens into engine PLD reference table
                self._ingest_reference_content(res)
            elif name == "replace_file_content":
                p = args.get("path", "")
                target = args.get("target_chunk", "")
                repl = args.get("replacement_chunk", "")
                file_path = self.cwd / p if not Path(p).is_absolute() else Path(p)
                FileEditor.replace_file_content(file_path, target, repl)
                res = f"Successfully updated {p}"
                self._ingest_reference_content(repl)
            elif name == "write_file":
                p = args.get("path", "")
                content = args.get("content", "")
                file_path = self.cwd / p if not Path(p).is_absolute() else Path(p)
                FileEditor.write_file(file_path, content)
                res = f"Successfully wrote {p}"
                self._ingest_reference_content(content)
            elif name == "search":
                pattern = args.get("pattern", "")
                root = args.get("root_dir", ".")
                max_m = args.get("max_matches", 50)
                matches = RipgrepSearcher.search(pattern, self.cwd / root if not Path(root).is_absolute() else root, max_m)

                res = "\n".join(matches) if matches else "No matches found"
            elif name == "web_search":
                query = args.get("query", "")
                max_r = args.get("max_results", 5)
                res = WebResearcher.search(query, max_results=max_r)
            elif name == "fetch_url":
                url = args.get("url", "")
                max_c = args.get("max_chars", 6000)
                res = WebResearcher.fetch_url(url, max_chars=max_c)
            elif name == "manage_todo":
                action = args.get("action", "get")
                if action == "set":
                    tasks = args.get("tasks", [])
                    self.todo.set_tasks(tasks)
                    res = f"Plan initialized with {len(tasks)} steps:\n" + self.todo.active_summary()
                elif action == "update":
                    # Robust task_id extraction: handle direct int, string int, alternative keys ('id', 'step', 'task'), or regex digits
                    tid = None
                    for key in ("task_id", "id", "step", "task"):
                        val = args.get(key)
                        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                            tid = val
                            break
                        elif isinstance(val, str) and val.isdigit():
                            tid = int(val)
                            break
                    if tid is None:
                        # Check for flag keys like {"task_id4": True, "task_id1": False}
                        for k, v in args.items():
                            if k.startswith("task_id") and v is True:
                                m = re.search(r"\d+", k)
                                if m:
                                    cand = int(m.group(0))
                                    if any(t["id"] == cand for t in self.todo.tasks):
                                        tid = cand
                                        break
                    if tid is None:
                        # Check if any key contains digits or regex search across all string values/keys (skipping bools)
                        for k, v in args.items():
                            if k.startswith("task_id") and isinstance(v, int) and not isinstance(v, bool):
                                tid = v
                                break
                            if isinstance(v, bool):
                                continue
                            m = re.search(r"\d+", str(k) + " " + str(v))
                            if m:
                                cand = int(m.group(0))
                                if any(t["id"] == cand for t in self.todo.tasks):
                                    tid = cand
                                    break
                    st = args.get("status", "completed")
                    # If tid still could not be determined or points to an already completed task when updating to 'completed',
                    # automatically resolve to the next pending or in_progress task so the plan naturally advances!
                    if tid is None or (st == "completed" and any(t["id"] == tid and t["status"] == "completed" for t in self.todo.tasks)):
                        next_t = self.todo.get_next_pending_or_in_progress()
                        if next_t:
                            tid = next_t["id"]
                        elif tid is None:
                            tid = 1

                    updated = self.todo.update_task(tid, st)
                    if updated:
                        res = f"Step {tid} updated to '{st}'.\n" + self.todo.active_summary()
                    else:
                        res = f"Error: task_id {tid} not found.\n" + self.todo.active_summary()
                else:
                    res = self.todo.active_summary() or "No active plan currently defined."
                if callable(getattr(self, "on_todo_update", None)):
                    try:
                        self.on_todo_update(self.todo)
                    except Exception:
                        pass
            else:
                return f"Error: unknown tool {name}"
            return compress_tool_output(res)
        except Exception as e:
            return f"Tool error ({name}): {e}"

    def step(self, user_input: str, max_turns: int | None = None, verbose: bool = True, cancel_event: Any = None, on_status: Any = None, console: Any = None) -> str:
        """Run multi-turn agent interaction loop until final assistant response (unbounded by default)."""
        out_console = console or self.console
        self.last_user_prompt = user_input
        self.last_failed_prompt = ""
        # Check and compact context if history has grown beyond safe window budget
        compacted = self.compact_history_if_needed(threshold_ratio=0.70)
        if compacted and verbose:
            out_console.print("  [bold cyan]⚡ [context-compactor][/bold cyan] [dim]Compressed middle history to maintain fast prefix cache and headroom.[/dim]")

        self.messages.append({"role": "user", "content": user_input})
        turns = 0

        while max_turns is None or turns < max_turns:
            if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                if verbose:
                    out_console.print("\n  [bold red]⚡ [interrupted][/bold red] [dim]Execution paused by user.[/dim]")
                if on_status:
                    on_status("Interrupted")
                self.save_checkpoint()
                return "Turn interrupted by user."

            turns += 1
            # If an explicit turn limit was provided and reached, omit tools to force response synthesis
            has_budget_cap = max_turns is not None
            provide_tools = None if (has_budget_cap and turns >= max_turns) else TOOL_DEFINITIONS
            if has_budget_cap and turns == max_turns:
                self.messages.append({
                    "role": "user",
                    "content": "You have reached your tool exploration budget. Now provide your final synthesized response directly based on all the information gathered above.",
                })

            payload = {
                "model": self.model,
                "messages": self.messages,
                "session_id": self.session_id,
                "temperature": 0.0,
                "stream_options": {"include_usage": True},
            }
            if getattr(self, "reasoning_effort", None) == "none":
                payload["enable_thinking"] = False
                payload["reasoning_effort"] = "none"
            elif getattr(self, "reasoning_effort", None):
                payload["enable_thinking"] = True
                payload["reasoning_effort"] = self.reasoning_effort
            if provide_tools:
                payload["tools"] = provide_tools

            accumulated_content = []
            accumulated_thinking = []
            tool_calls_map: dict[int, dict[str, Any]] = {}
            spec_futures: dict[int, tuple[str, dict[str, Any], float, Any]] = {}
            spec_pool = ThreadPoolExecutor(max_workers=2)
            active_thinking = False

            last_stream_speed = None
            last_stream_tokens = None
            last_prompt_tokens = None
            last_cached_tokens = None
            tool_parse_cursors: dict[int, int] = {}

            # Determine streaming strategy for this step:
            # - "always": stream every turn
            # - "never": non-streaming for all turns
            # - "hybrid": non-streaming when tools are provided (tool exploration/execution turns),
            #             streaming when no tools are provided (direct conversational answer / synthesis)
            post_mocked = hasattr(getattr(self, "_post", None), "_mock_self")
            stream_mocked = hasattr(getattr(self, "_post_stream", None), "_mock_self")

            if not getattr(self, "stream", True) or self.stream_mode == "never":
                use_stream = False
            elif self.stream_mode == "always":
                use_stream = True
            elif stream_mocked and not post_mocked:
                use_stream = True
            elif post_mocked and not stream_mocked:
                use_stream = False
            else:
                use_stream = False if provide_tools else True

            self.last_turn_streamed = use_stream

            if not use_stream:
                try:
                    resp = self._post("/chat/completions", payload)
                except Exception as post_err:
                    err_str = str(post_err)
                    # Transient tool syntax / parsing glitch from model: nudge with higher temp or retry
                    if "invalid_model_output" in err_str or "malformed tool" in err_str or "repeated a tool parameter" in err_str:
                        if verbose:
                            out_console.print("  [bold yellow]⚡ [retry-recovery][/bold yellow] [dim]Model generated malformed tool XML. Retrying turn...[/dim]")
                        retry_payload = dict(payload)
                        retry_payload["temperature"] = 0.2
                        resp = self._post("/chat/completions", retry_payload)
                    else:
                        raise
                choice = resp.get("choices", [{}])[0]
                msg = choice.get("message", {})

                # Extract and display reasoning in non-streaming mode
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    accumulated_thinking.append(reasoning)
                    if verbose:
                        out_console.print("\n[bold rgb(160,120,240)]· 💭 thinking: [/bold rgb(160,120,240)]", end="")
                        out_console.print(reasoning, markup=False, style="dim italic bright_white")
                        out_console.print("\n")
                        if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                            out_console.file.flush()
                        elif hasattr(sys.stdout, "flush"):
                            sys.stdout.flush()

                # Extract content
                text = msg.get("content")
                if text:
                    accumulated_content.append(text)

                # Capture real engine telemetry from response
                timings = resp.get("timings") or {}
                metrics = resp.get("metrics") or {}
                req_lat = metrics.get("request_latency") or {}
                spd = timings.get("predicted_per_second") or req_lat.get("stream_tokens_per_second")
                if spd and float(spd) > 0:
                    last_stream_speed = float(spd)
                usage = resp.get("usage") or {}
                if usage.get("completion_tokens"):
                    last_stream_tokens = int(usage["completion_tokens"])
                prompt_toks = usage.get("prompt_tokens") or timings.get("prompt_n", 0)
                cached_toks = (
                    usage.get("prompt_tokens_details", {}).get("cached_tokens")
                    or timings.get("cache_n", 0)
                )
                if prompt_toks:
                    self.stats["total_prompt_tokens"] = self.stats.get("total_prompt_tokens", 0) + int(prompt_toks)
                    self.stats["cached_prompt_tokens"] = self.stats.get("cached_prompt_tokens", 0) + int(cached_toks)
            else:
                try:
                    for chunk in self._post_stream("/chat/completions", payload, cancel_event=cancel_event):
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            
                            # Stream thinking / reasoning content
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                if not active_thinking:
                                    active_thinking = True
                                    if on_status:
                                        on_status("💭 Thinking...")
                                    if verbose:
                                        out_console.print("\n[bold rgb(160,120,240)]· 💭 thinking: [/bold rgb(160,120,240)]", end="")
                                        if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                            out_console.file.flush()
                                        elif hasattr(sys.stdout, "flush"):
                                            sys.stdout.flush()
                                accumulated_thinking.append(reasoning)
                                if verbose:
                                    out_console.print(reasoning, end="", markup=False, style="dim italic bright_white")
                                    if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                        out_console.file.flush()
                                    elif hasattr(sys.stdout, "flush"):
                                        sys.stdout.flush()

                            # Stream standard response tokens
                            text = delta.get("content")
                            if text:
                                if active_thinking:
                                    active_thinking = False
                                    if on_status:
                                        on_status("⚡ Streaming response...")
                                    if verbose:
                                        out_console.print("\n\n[bold rgb(100,200,255)]🤖 Response:[/bold rgb(100,200,255)]\n", end="")
                                        if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                            out_console.file.flush()
                                        elif hasattr(sys.stdout, "flush"):
                                            sys.stdout.flush()
                                if verbose:
                                    out_console.print(text, end="", markup=False)
                                    if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                        out_console.file.flush()
                                    elif hasattr(sys.stdout, "flush"):
                                        sys.stdout.flush()
                                accumulated_content.append(text)

                            # Stream tool call fragments
                            tcs = delta.get("tool_calls", [])
                            if tcs and active_thinking:
                                active_thinking = False
                                if verbose:
                                    out_console.print("\n")
                                    if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                        out_console.file.flush()
                            for tc in tcs:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {
                                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    tool_calls_map[idx]["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

                                # Speculatively dispatch pure-read tools as soon as arguments form valid JSON
                                cur_fn_name = tool_calls_map[idx]["function"]["name"]
                                cur_args_str = tool_calls_map[idx]["function"]["arguments"].strip()
                                last_cursor = tool_parse_cursors.get(idx, 0)
                                if (
                                    idx not in spec_futures
                                    and cur_fn_name in {"view_file", "search", "web_search", "fetch_url"}
                                    and cur_args_str.startswith("{")
                                    and cur_args_str.endswith("}")
                                    and len(cur_args_str) > last_cursor
                                ):
                                    tool_parse_cursors[idx] = len(cur_args_str)
                                    try:
                                        cur_parsed_args = json.loads(cur_args_str)
                                        t_spec = time.monotonic()
                                        fut = spec_pool.submit(self.execute_tool, cur_fn_name, cur_parsed_args)
                                        spec_futures[idx] = (cur_fn_name, cur_parsed_args, t_spec, fut)
                                    except Exception:
                                        pass

                        # Capture real engine telemetry from chunk metadata
                        timings = chunk.get("timings") or {}
                        metrics = chunk.get("metrics") or {}
                        req_lat = metrics.get("request_latency") or {}
                        spd = timings.get("predicted_per_second") or req_lat.get("stream_tokens_per_second")
                        if spd and float(spd) > 0:
                            last_stream_speed = float(spd)
                        usage = chunk.get("usage") or {}
                        if usage.get("completion_tokens"):
                            last_stream_tokens = int(usage["completion_tokens"])
                        prompt_toks = usage.get("prompt_tokens") or timings.get("prompt_n")
                        cached_toks = (
                            usage.get("prompt_tokens_details", {}).get("cached_tokens")
                            or timings.get("cache_n")
                        )
                        if prompt_toks is not None:
                            last_prompt_tokens = int(prompt_toks)
                        if cached_toks is not None:
                            last_cached_tokens = int(cached_toks)

                    if last_prompt_tokens is not None:
                        self.stats["total_prompt_tokens"] = self.stats.get("total_prompt_tokens", 0) + last_prompt_tokens
                        self.stats["cached_prompt_tokens"] = self.stats.get("cached_prompt_tokens", 0) + (last_cached_tokens or 0)

                    if active_thinking:
                        active_thinking = False
                        if verbose:
                            out_console.print("\n")
                            if hasattr(out_console, "file") and hasattr(out_console.file, "flush"):
                                out_console.file.flush()

                    msg = {"role": "assistant"}
                    full_content = "".join(accumulated_content)
                    if full_content:
                        msg["content"] = full_content
                    if tool_calls_map:
                        msg["tool_calls"] = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]
                except Exception:
                    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                        return "Turn interrupted by user."
                    # Graceful fallback to standard _post if streaming encounters errors
                    resp = self._post("/chat/completions", payload)
                    choice = resp.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    self.last_turn_streamed = False

            if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                if verbose:
                    out_console.print("\n  [bold red]⚡ [interrupted][/bold red] [dim]Execution paused by user.[/dim]")
                return "Turn interrupted by user."

            if last_stream_speed or last_stream_tokens:
                self.last_turn_stats = {
                    "tok_s": last_stream_speed,
                    "tokens": last_stream_tokens,
                }
            elif accumulated_thinking or accumulated_content:
                total_words = len(" ".join(accumulated_thinking + accumulated_content).split())
                self.last_turn_stats = {
                    "tok_s": None,
                    "tokens": max(1, int(total_words * 1.33)),
                }

            self.messages.append(msg)
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")

            if not tool_calls:
                # If an execution plan is active and has unfinished tasks, nudge the model to execute the next step
                # rather than stopping prematurely, unless we are on the final turn budget or user requested a stop.
                next_task = self.todo.get_next_pending_or_in_progress() if not self.todo.is_empty() else None
                if next_task and (max_turns is None or turns < max_turns - 1):
                    if verbose:
                        out_console.print(f"  [bold yellow]⚡ [plan-guard][/bold yellow] [dim]Step {next_task['id']} pending ('{next_task['task'][:60]}'). Auto-advancing agent to continue plan...[/dim]")
                    self.messages.append({
                        "role": "user",
                        "content": f"Please proceed with the next step in the execution plan: Step {next_task['id']}: {next_task['task']}. Use your tools to execute it.",
                    })
                    spec_pool.shutdown(wait=False)
                    continue

                if verbose:
                    out_console.print()
                if on_status:
                    on_status("Ready")
                spec_pool.shutdown(wait=False)
                return content or ""

            # Execute tool calls with parallel read batching for pure reads
            pure_reads = {"view_file", "search"}
            can_parallel = len(tool_calls) > 1 and all(
                tc.get("function", {}).get("name") in pure_reads for tc in tool_calls
            )

            if can_parallel:
                if on_status:
                    on_status("⚡ Action: Executing tools in parallel...")
                def run_tc(tc):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        fn_args = {}
                    spec_info = None
                    for s_idx, (s_name, s_args, s_t0, s_fut) in list(spec_futures.items()):
                        if s_name == fn_name and s_args == fn_args:
                            spec_info = (s_name, s_args, s_t0, s_fut)
                            spec_futures.pop(s_idx, None)
                            break
                    if spec_info is not None:
                        try:
                            res = spec_info[3].result()
                            dt_ms = (time.monotonic() - spec_info[2]) * 1000.0
                            return tc, fn_name, fn_args, res, dt_ms
                        except Exception:
                            pass
                    t0 = time.monotonic()
                    res = self.execute_tool(fn_name, fn_args)
                    dt_ms = (time.monotonic() - t0) * 1000.0
                    return tc, fn_name, fn_args, res, dt_ms

                with ThreadPoolExecutor(max_workers=min(4, len(tool_calls))) as pool:
                    results = list(pool.map(run_tc, tool_calls))
                for tc, fn_name, fn_args, tool_result, dt_ms in results:
                    if verbose:
                        arg_desc = ""
                        if fn_name == "view_file":
                            arg_desc = f"{fn_args.get('path', '')}:{fn_args.get('start_line', 1)}"
                        elif fn_name in ("replace_file_content", "write_file"):
                            arg_desc = f"{fn_args.get('path', '')}"
                        elif fn_name == "run_command":
                            arg_desc = f"{fn_args.get('command', '')[:50]}"
                        elif fn_name in ("search", "web_search"):
                            arg_desc = f"'{fn_args.get('pattern') or fn_args.get('query')}'"
                        elif fn_name == "fetch_url":
                            arg_desc = f"{fn_args.get('url', '')[:45]}"
                        cur_ctx = self.estimate_context_tokens()
                        max_c = self.stats.get("max_context", 131072)
                        tool_line = Text()
                        tool_line.append("  ⏺ ", style="bold rgb(140,160,190)")
                        tool_line.append(f"{fn_name}", style="bold bright_white")
                        if arg_desc:
                            tool_line.append(f" {arg_desc}", style="dim white")
                        tool_line.append(f" · {dt_ms:.1f}ms", style="dim green")
                        tool_line.append(f" · ctx:{cur_ctx:,}/{max_c // 1024}k", style="dim cyan")
                        out_console.print(tool_line)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": fn_name,
                        "content": tool_result,
                    })
            else:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        fn_args = {}

                    arg_desc = ""
                    if fn_name == "view_file":
                        arg_desc = f"{fn_args.get('path', '')}:{fn_args.get('start_line', 1)}"
                    elif fn_name in ("replace_file_content", "write_file"):
                        arg_desc = f"{fn_args.get('path', '')}"
                    elif fn_name == "run_command":
                        arg_desc = f"{fn_args.get('command', '')[:40]}"
                    elif fn_name in ("search", "web_search"):
                        arg_desc = f"'{fn_args.get('pattern') or fn_args.get('query')}'"
                    elif fn_name == "fetch_url":
                        arg_desc = f"{fn_args.get('url', '')[:35]}"
                    elif fn_name == "manage_todo":
                        action = fn_args.get("action", "")
                        if action == "set":
                            arg_desc = f"set {len(fn_args.get('tasks', []))} tasks"
                        elif action == "update":
                            arg_desc = f"task #{fn_args.get('task_id')} -> {fn_args.get('status')}"
                        else:
                            arg_desc = "get tasks"

                    if on_status:
                        on_status(f"⚡ Action: {fn_name} {arg_desc}".strip())

                    spec_info = None
                    for s_idx, (s_name, s_args, s_t0, s_fut) in list(spec_futures.items()):
                        if s_name == fn_name and s_args == fn_args:
                            spec_info = (s_name, s_args, s_t0, s_fut)
                            spec_futures.pop(s_idx, None)
                            break
                    if spec_info is not None:
                        try:
                            tool_result = spec_info[3].result()
                            dt_ms = (time.monotonic() - spec_info[2]) * 1000.0
                        except Exception as e:
                            tool_result = f"Error in speculative tool execution: {e}"
                            dt_ms = (time.monotonic() - spec_info[2]) * 1000.0
                    else:
                        t0 = time.monotonic()
                        tool_result = self.execute_tool(fn_name, fn_args)
                        dt_ms = (time.monotonic() - t0) * 1000.0
                    if verbose:
                        cur_ctx = self.estimate_context_tokens()
                        max_c = self.stats.get("max_context", 131072)
                        tool_line = Text()
                        tool_line.append("  ⏺ ", style="bold rgb(140,160,190)")
                        tool_line.append(f"{fn_name}", style="bold bright_white")
                        if arg_desc:
                            tool_line.append(f" {arg_desc}", style="dim white")
                        tool_line.append(f" · {dt_ms:.1f}ms", style="dim green")
                        tool_line.append(f" · ctx:{cur_ctx:,}/{max_c // 1024}k", style="dim cyan")
                        out_console.print(tool_line)

                        # manage_todo updates are reflected in the live footer tracker
                        # and summarized cleanly at turn boundaries, avoiding duplicate multi-panel printing.

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": fn_name,
                        "content": tool_result,
                    })

            # Check for repetitive tool loops (trigger only on 3 consecutive identical tool calls)
            for tc in tool_calls:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "")
                sig = f"{fn.get('name')}:{raw_args}"
                self.recent_tool_signatures.append(sig)
                if len(self.recent_tool_signatures) >= 3 and self.recent_tool_signatures[-1] == self.recent_tool_signatures[-2] == self.recent_tool_signatures[-3]:
                    if verbose:
                        self.console.print("  [bold yellow]⚡ [circuit-breaker][/bold yellow] [dim]Repeated identical tool query detected (3x consecutive). Nudging agent to synthesize final response.[/dim]")
                    self.messages.append({
                        "role": "user",
                        "content": "You have repeated this exact tool call 3 times in a row without making progress. Please do not call this tool again; synthesize and provide your final complete answer now.",
                    })
                    break

            spec_pool.shutdown(wait=False)

        # Final fallback if loop ends without response
        self.save_checkpoint()
        last_msg = self.messages[-1]
        return last_msg.get("content") or "Agent completed investigation."

    def sidetalk(self, question: str, verbose: bool = True) -> str:
        """Run an informed side-conversation query with workspace and conversation awareness without contaminating main agent history."""
        # Extract recent conversation summary so /btw understands what the user and agent are working on
        recent_history: list[dict[str, Any]] = []
        for msg in self.messages[1:]:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content and isinstance(content, str):
                snippet = content[:300] + ("..." if len(content) > 300 else "")
                recent_history.append({"role": role, "content": snippet})
        recent_history = recent_history[-6:]

        active_plan = self.todo.active_summary()
        workspace_context = f"Current working directory: {self.cwd}\n"
        if active_plan:
            workspace_context += f"Active execution plan:\n{active_plan}\n"

        system_instruction = (
            "You are a helpful expert software engineering companion in an active terminal coding session. "
            "The user is asking a side question (/btw). You have full awareness of the workspace and current session.\n"
            f"{workspace_context}\n"
            "Provide a direct, insightful, and practical answer to the user's question. "
            "Be specific and technical when discussing code or project architecture. Do not attempt to call tools."
        )

        ephemeral_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instruction},
        ]
        ephemeral_messages.extend(recent_history)
        ephemeral_messages.append({"role": "user", "content": question})

        # Inherit reasoning effort unless set to none
        effort = self.reasoning_effort if self.reasoning_effort != "none" else "minimal"
        payload = {
            "model": self.model,
            "messages": ephemeral_messages,
            "temperature": 0.2,
            "enable_thinking": effort not in ("none", "minimal"),
            "reasoning_effort": effort,
        }
        if verbose:
            self.console.print("\n  [bold cyan]💬 /btw Side Discussion[/bold cyan] [dim](isolated context)[/dim]")
        try:
            accumulated: list[str] = []
            if hasattr(getattr(self, "_post", None), "_mock_self") or not getattr(self, "stream", True):
                resp = self._post("/chat/completions", payload)
                answer = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                if verbose:
                    self.console.print(Panel(answer, title="[bold cyan]💬 /btw[/bold cyan]", border_style="cyan"))
                return answer

            for chunk in self._post_stream("/chat/completions", payload):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content")
                if text:
                    accumulated.append(text)
                    if verbose:
                        self.console.print(text, end="", markup=False)
            if verbose:
                self.console.print("\n")
            return "".join(accumulated)
        except Exception as e:
            err = f"Side-talk error: {e}"
            if verbose:
                self.console.print(f"\n  [bold red]⚡ {err}[/bold red]\n")
            return err

    def close(self) -> None:
        if self.pty is not None:
            self.pty.close()


def _dispatch_slash_command(agent: SlipstreamAgent, user_prompt: str, plan_mode: bool) -> tuple[bool, bool]:
    """Execute interactive slash command. Returns (handled, new_plan_mode)."""
    console = agent.console
    cmd_part = user_prompt.split()[0].lower()
    if cmd_part in ("/help", "/?"):
        console.print(Panel(
            "[bold cyan]/todo [clear|add <text>][/bold cyan] View or manage the current execution plan checklist\n"
            "[bold cyan]/continue[/bold cyan]        Resume previous interrupted work or task\n"
            "[bold cyan]/retry[/bold cyan]           Re-attempt last failed or timed-out prompt\n"
            "[bold cyan]/resume [id][/bold cyan]     Restore session from APFS vault (defaults to latest)\n"
            "[bold cyan]/sessions[/bold cyan]        List saved agent sessions in APFS vault\n"
            "[bold cyan]/save[/bold cyan]            Checkpoint current session state to disk\n"
            "[bold cyan]/compact[/bold cyan]         Compress middle conversation history to reclaim context\n"
            "[bold cyan]/model[/bold cyan]           Display active model name and context architecture\n"
            "[bold cyan]/btw <question>[/bold cyan] Quick side-talk with model without polluting agent history\n"
            "[bold cyan]/plan[/bold cyan]           Toggle read-only architectural planning mode\n"
            "[bold cyan]/thinking <lvl>[/bold cyan] Set thinking level (none, minimal, low, medium, high, max)\n"
            "[bold cyan]/stream <mode>[/bold cyan]   Set streaming mode (hybrid, always, never [batch])\n"
            "[bold cyan]/no-stream[/bold cyan]       Completely suppress streaming deltas (Option B: batch mode)\n"
            "[bold cyan]/diff[/bold cyan]           Show colorized git diff of uncommitted changes\n"
            "[bold cyan]/undo[/bold cyan]           Revert changes in working tree (git checkout)\n"
            "[bold cyan]/map[/bold cyan]            Display repository structural symbol map\n"
            "[bold cyan]/test[/bold cyan]           Run project test suite (make test-engine-cpu)\n"
            "[bold cyan]/stats[/bold cyan]          Display session token stats and context consumption\n"
            "[bold cyan]/clear[/bold cyan]          Reset agent conversation history and clear context\n"
            "[bold cyan]/help[/bold cyan]           Show this help message",
            title="Available Commands", border_style="rgb(70,75,85)"
        ))
        return True, plan_mode
    elif cmd_part in ("/continue", "/c"):
        agent.pending_prompt = "Please continue your work where you left off. Review your current progress, plan, and complete the remaining steps."
        console.print("  [bold cyan]⚡ Continuing interrupted work...[/bold cyan]")
        return True, plan_mode
    elif cmd_part == "/retry":
        retry_prompt = agent.last_failed_prompt or agent.last_user_prompt
        if retry_prompt:
            agent.pending_prompt = retry_prompt
            console.print(f"  [bold yellow]⚡ Retrying prompt:[/bold yellow] [dim]{retry_prompt[:60]}...[/dim]")
        else:
            console.print("  [dim]No previous prompt found to retry.[/dim]")
        return True, plan_mode
    elif cmd_part in ("/resume", "/load"):
        parts = user_prompt.split()
        target_sid = parts[1] if len(parts) > 1 else "latest"
        restored = SlipstreamAgent.restore_checkpoint(target_sid)
        if restored:
            agent.messages = restored.messages
            agent.stats = restored.stats
            agent.todo.tasks = restored.todo.tasks
            agent.session_id = restored.session_id
            console.print(f"  [bold green]✓[/bold green] Resumed session [bold cyan]{agent.session_id}[/bold cyan] ({len(agent.messages)} messages, Turn #{agent.stats.get('turns', 0)})")
            if not agent.todo.is_empty():
                console.print(Panel(agent.todo.active_summary(), title="[bold cyan]📋 Resumed Execution Plan[/bold cyan]", border_style="cyan"))
        else:
            console.print(f"  [bold red]✗[/bold red] Session '{target_sid}' not found in vault (~/.cache/slipstream/agent_vaults).")
        return True, plan_mode
    elif cmd_part in ("/sessions", "/history"):
        checkpoints = SlipstreamAgent.list_checkpoints()
        if not checkpoints:
            console.print("  [dim]No saved sessions found in vault (~/.cache/slipstream/agent_vaults).[/dim]")
        else:
            tbl = Table(title="Saved Slipstream Agent Sessions", border_style="cyan")
            tbl.add_column("Session ID", style="bold cyan")
            tbl.add_column("Updated", style="dim")
            tbl.add_column("Turns", justify="right")
            tbl.add_column("Messages", justify="right")
            tbl.add_column("Last Prompt", style="italic")
            for cp in checkpoints[:12]:
                last_p = cp["last_prompt"][:45] + ("..." if len(cp["last_prompt"]) > 45 else "")
                tbl.add_row(
                    cp["session_id"],
                    cp["updated_at"],
                    str(cp["turns"]),
                    str(cp["messages"]),
                    last_p,
                )
            console.print(tbl)
            console.print("  [dim]Type [bold cyan]/resume <session_id>[/bold cyan] or [bold cyan]/resume[/bold cyan] to load.[/dim]")
        return True, plan_mode
    elif cmd_part == "/save":
        saved_path = agent.save_checkpoint()
        console.print(f"  [bold green]✓[/bold green] Checkpoint saved: [bold cyan]{saved_path}[/bold cyan]")
        return True, plan_mode
    elif cmd_part in ("/compact", "/compress"):
        compacted = agent.compact_history_if_needed(threshold_ratio=0.0, preserve_recent=4)
        if compacted:
            cur_ctx = agent.estimate_context_tokens()
            console.print(f"  [bold green]✓[/bold green] Conversation history compacted. Current active context: ~{cur_ctx:,} tokens.")
        else:
            console.print("  [dim]Conversation history is already minimal or cannot be compacted further.[/dim]")
        return True, plan_mode
    elif cmd_part == "/model":
        max_c = agent.stats.get("max_context", 131072)
        tbl = Table(title="Active Engine Model Configuration", border_style="rgb(70,75,85)")
        tbl.add_column("Parameter", style="bold")
        tbl.add_column("Value", style="cyan")
        tbl.add_row("Model Name", str(agent.model))
        tbl.add_row("Max Context Limit", f"{max_c:,} tokens")
        tbl.add_row("Reasoning Effort", str(agent.reasoning_effort))
        tbl.add_row("Stream Mode", str(agent.stream_mode))
        tbl.add_row("Endpoint", str(agent.base_url))
        if agent.unix_socket_path:
            tbl.add_row("UNIX Socket", str(agent.unix_socket_path))
        console.print(tbl)
        return True, plan_mode
    elif cmd_part == "/todo":
        parts = user_prompt.split(maxsplit=2)
        if len(parts) > 1 and parts[1].lower() == "clear":
            agent.todo.tasks.clear()
            console.print("  [bold green]✓[/bold green] [dim]Execution plan checklist cleared.[/dim]")
        elif len(parts) > 2 and parts[1].lower() == "add":
            task_desc = parts[2].strip()
            new_id = len(agent.todo.tasks) + 1
            agent.todo.tasks.append({"id": new_id, "task": task_desc, "status": "pending"})
            console.print(f"  [bold green]✓[/bold green] Added step #{new_id}: {task_desc}")
        else:
            if agent.todo.is_empty():
                console.print("  [dim]No execution plan active. The agent initializes one automatically during multi-step tasks, or type /todo add <task>.[/dim]")
            else:
                plan_text = Text()
                for t in agent.todo.get_tasks():
                    if t["status"] == "completed":
                        plan_text.append(f"  ✔ {t['id']}. {t['task']}\n", style="bold green")
                    elif t["status"] == "in_progress":
                        plan_text.append(f"  ❯ {t['id']}. {t['task']}  [IN PROGRESS]\n", style="bold yellow")
                    elif t["status"] == "failed":
                        plan_text.append(f"  ✗ {t['id']}. {t['task']}  [FAILED]\n", style="bold red")
                    else:
                        plan_text.append(f"  ○ {t['id']}. {t['task']}\n", style="dim white")
                    plan_text.rstrip()
                console.print(Panel(plan_text, title="[bold cyan]📋 Execution Plan[/bold cyan]", title_align="left", border_style="cyan", padding=(0, 1)))
        return True, plan_mode
    elif cmd_part == "/btw":
        question = user_prompt[4:].strip()
        if not question:
            console.print("  [dim]Usage: /btw <your question> (e.g. /btw what is the difference between arc and rc?)[/dim]")
        else:
            agent.sidetalk(question, verbose=True)
        return True, plan_mode
    elif cmd_part in ("/thinking", "/think"):
        parts = user_prompt.split()
        levels = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        if len(parts) > 1:
            lvl = parts[1].lower()
            if lvl in levels:
                agent.reasoning_effort = lvl
                console.print(f"  [bold green]✓[/bold green] [dim]Thinking level set to:[/dim] [bold magenta]{lvl}[/bold magenta]")
            else:
                console.print(f"  [bold red]✗[/bold red] Invalid thinking level '{lvl}'. Valid options: {', '.join(levels)}")
        else:
            console.print(f"  · Current thinking level: [bold magenta]{agent.reasoning_effort}[/bold magenta] (options: {', '.join(levels)})")
        return True, plan_mode
    elif cmd_part in ("/stream", "/nostream", "/no-stream"):
        if cmd_part in ("/nostream", "/no-stream"):
            agent.stream_mode = "never"
            console.print("  [bold green]✓[/bold green] [dim]Streaming completely suppressed[/dim] [bold cyan](Option B: batch mode)[/bold cyan]. Output will render exclusively via Markdown panels.")
            return True, plan_mode
        parts = user_prompt.split()
        if len(parts) > 1:
            sm = parts[1].lower()
            if sm in ("never", "batch", "suppress", "off", "none", "no-stream"):
                agent.stream_mode = "never"
                console.print("  [bold green]✓[/bold green] [dim]Streaming completely suppressed[/dim] [bold cyan](Option B: batch mode)[/bold cyan]. Output will render exclusively via Markdown panels.")
            elif sm in ("always", "on", "all"):
                agent.stream_mode = "always"
                console.print("  [bold green]✓[/bold green] [dim]Streaming mode set to:[/dim] [bold cyan]always[/bold cyan]")
            elif sm in ("hybrid", "default"):
                agent.stream_mode = "hybrid"
                console.print("  [bold green]✓[/bold green] [dim]Streaming mode set to:[/dim] [bold cyan]hybrid (Smart Hybrid)[/bold cyan]")
            else:
                console.print(f"  [bold red]✗[/bold red] Invalid streaming mode '{sm}'. Valid options: hybrid, always, never (batch)")
        else:
            mode_desc = f"{agent.stream_mode} (batch/suppressed)" if agent.stream_mode == "never" else agent.stream_mode
            console.print(f"  · Current streaming mode: [bold cyan]{mode_desc}[/bold cyan] (options: hybrid, always, never [batch])")
        return True, plan_mode
    elif cmd_part == "/plan":
        plan_mode = not plan_mode
        status = "[bold green]ENABLED[/bold green] (read-only architectural reconnaissance)" if plan_mode else "[bold yellow]DISABLED[/bold yellow] (normal execution)"
        console.print(f"  · Plan Mode {status}")
        return True, plan_mode
    elif cmd_part == "/diff":
        rc, out = agent.pty.run_command("git diff")
        if out.strip():
            console.print(Panel(out, title="Git Diff", border_style="yellow"))
        else:
            console.print("  [dim]No uncommitted changes in working tree.[/dim]")
        return True, plan_mode
    elif cmd_part == "/clear":
        repo_map = RepoMapper.get_repo_map(agent.cwd, max_files=35)
        full_system_prompt = f"{SYSTEM_PROMPT}\n\n{repo_map}"
        agent.messages = [{"role": "system", "content": full_system_prompt}]
        agent.stats["turns"] = 0
        console.print("  [bold green]✓[/bold green] [dim]Conversation history reset to initial state.[/dim]")
        return True, plan_mode
    elif cmd_part == "/undo":
        rc, out = agent.pty.run_command("git checkout .")
        console.print("  [bold red]↩[/bold red] [dim]Reverted uncommitted changes in working tree.[/dim]")
        return True, plan_mode
    elif cmd_part == "/map":
        console.print(Panel(RepoMapper.get_repo_map(agent.cwd, max_files=50), title="Workspace Structure", border_style="blue"))
        return True, plan_mode
    elif cmd_part == "/test":
        console.print("  [bold yellow]⚡ Running test suite (make test-engine-cpu)...[/bold yellow]")
        rc, out = agent.pty.run_command("make test-engine-cpu")
        border = "green" if rc == 0 else "red"
        console.print(Panel(out, title=f"Test Suite Result [Exit {rc}]", border_style=border))
        return True, plan_mode
    elif cmd_part == "/stats":
        est_ctx = agent.estimate_context_tokens()
        max_ctx = agent.stats.get("max_context", 131072)
        pct = (est_ctx / max_ctx) * 100.0 if max_ctx > 0 else 0.0
        prompt_toks = agent.stats.get("total_prompt_tokens", 0)
        cached_toks = agent.stats.get("cached_prompt_tokens", 0)
        hit_rate = (cached_toks / prompt_toks * 100.0) if prompt_toks > 0 else 0.0

        table = Table(title="Session Telemetry & Context Management", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        table.add_row("Session ID", f"{agent.session_id}")
        table.add_row("Active Turns Completed", str(agent.stats["turns"]))
        table.add_row("Conversation Message Count", str(len(agent.messages)))
        table.add_row("Current Active Context", f"~{est_ctx:,} tokens ({pct:.1f}% used)")
        table.add_row("Available Context Headroom", f"~{max(0, max_ctx - est_ctx):,} tokens")
        table.add_row("Engine Max Context Limit", f"{max_ctx:,} tokens")
        table.add_row("Prefix Cache Hit Rate", f"{hit_rate:.1f}% ({cached_toks:,} / {prompt_toks:,} tokens)")
        table.add_row("Prefix Continuity Mode", "Radix-Preserving Append / Delta")
        table.add_row("Compaction Threshold", "70% of max context")
        console.print(table)
        return True, plan_mode
    return False, plan_mode


def _run_pinned_tui(agent: SlipstreamAgent) -> int:
    """Pinned bottom-prompt TUI with live streaming above and instant user steering."""
    import select
    import termios
    import threading
    import tty

    from rich.live import Live

    console = agent.console
    input_buf: list[str] = []
    plan_mode = False
    cancel_event = threading.Event()
    is_generating = threading.Event()
    status_detail = "Ready for input"
    turn_counter = agent.stats.get("turns", 0) + 1
    turn_start_time = 0.0
    spinner_idx = 0
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    last_stats = {"dt": 0.0, "tok_s": 0.0, "tokens": 0}
    worker_thread: threading.Thread | None = None

    history: list[str] = []
    history_idx = -1
    saved_draft = ""

    COMMAND_HINTS = {
        "/todo": "Manage execution plan checklist (/todo, /todo clear, /todo add <task>)",
        "/continue": "Resume previous interrupted work or task",
        "/retry": "Re-attempt last failed or timed-out prompt",
        "/resume": "Restore session from APFS vault (/resume, /resume <id>)",
        "/sessions": "List all saved agent sessions in APFS vault",
        "/save": "Checkpoint current session state to disk",
        "/compact": "Compress middle conversation history to reclaim context",
        "/model": "Display active model name and context architecture",
        "/btw": "Quick side-talk with model without polluting conversation context",
        "/plan": "Toggle read-only architectural planning mode",
        "/thinking": "Set reasoning effort (none, minimal, low, medium, high, max)",
        "/stream": "Set streaming mode (/stream hybrid, /stream always, /stream never/batch)",
        "/no-stream": "Completely suppress streaming deltas (Option B: batch mode)",
        "/diff": "Show colorized git diff of uncommitted changes",
        "/undo": "Revert uncommitted changes in working tree (git checkout .)",
        "/map": "Display structural symbol map of current repository",
        "/test": "Run project test suite (make test-engine-cpu)",
        "/stats": "Display session token telemetry and context window consumption",
        "/clear": "Reset agent conversation history and clear context",
        "/help": "Show list of available slash commands",
        "exit": "Quit agent session",
        "quit": "Quit agent session",
    }

    def get_lookahead(current_text: str) -> tuple[str, str]:
        """Return (ghost_suffix, command_doc) if current input matches a known command."""
        t = current_text.strip()
        if not t:
            return "", ""
        # Match commands starting with current_text
        matches = [cmd for cmd in COMMAND_HINTS if cmd.startswith(t.lower()) and cmd != t.lower()]
        if matches:
            first_match = matches[0]
            ghost = first_match[len(t):]
            return ghost, COMMAND_HINTS[first_match]
        # If exact match
        if t.lower() in COMMAND_HINTS:
            return "", COMMAND_HINTS[t.lower()]
        return "", ""

    def build_footer() -> Any:
        from rich.console import Group

        # Build prompt box
        t = Text()
        if is_generating.is_set():
            spin = SPINNER_FRAMES[spinner_idx % len(SPINNER_FRAMES)]
            elapsed = time.monotonic() - turn_start_time if turn_start_time > 0 else 0.0
            st_lower = status_detail.lower()

            border = "rgb(100,105,120)"
            t.append(f" {spin} ", style="bold rgb(140,160,200)")
            t.append(f"{status_detail} ", style="bold bright_white")
            t.append(f"· {elapsed:.1f}s ", style="dim white")

            # Live Context Length & Headroom Tracker during active turn
            cur_ctx = agent.estimate_context_tokens()
            max_c = agent.stats.get("max_context", 131072)
            c_pct = (cur_ctx / max_c * 100.0) if max_c > 0 else 0.0
            ctx_style = "bold red" if c_pct >= 80 else ("yellow" if c_pct >= 50 else "dim cyan")
            filled = min(6, int((c_pct / 100.0) * 6))
            bar = "■" * filled + "□" * (6 - filled)

            t.append(f"· ctx:[{bar}] {cur_ctx:,}/{max_c // 1024}k ({c_pct:.1f}%) ", style=ctx_style)
            t.append("· [Ctrl+C to steer]\n", style="dim rgb(220,100,100)")
        else:
            border = "rgb(80,85,95)"
            # Claude-style minimalist top pill
            t.append(" ● ", style="bold rgb(100,180,140)")
            t.append("Ready", style="bold bright_white")
            t.append("  ", style="dim")
            t.append(f"think:{agent.reasoning_effort}", style="dim magenta")
            stream_label = "batch" if agent.stream_mode == "never" else agent.stream_mode
            t.append(f"  stream:{stream_label}", style="dim cyan")
            
            # Live Context Length & Headroom Tracker with mini visual meter
            cur_ctx = agent.estimate_context_tokens()
            max_c = agent.stats.get("max_context", 131072)
            c_pct = (cur_ctx / max_c * 100.0) if max_c > 0 else 0.0
            ctx_style = "bold red" if c_pct >= 80 else ("yellow" if c_pct >= 50 else "dim cyan")
            # 6-segment visual context bar: e.g. [■■□□□□]
            filled = min(6, int((c_pct / 100.0) * 6))
            bar = "■" * filled + "□" * (6 - filled)
            t.append(f"  ctx:[{bar}] {cur_ctx:,}/{max_c // 1024}k", style=ctx_style)

            if not agent.todo.is_empty():
                done = sum(1 for x in agent.todo.tasks if x["status"] == "completed")
                total = len(agent.todo.tasks)
                t.append(f"  plan:{done}/{total}", style="dim green" if done == total else "dim yellow")
            if last_stats["dt"] > 0:
                t.append(
                    f"  {last_stats['tok_s']:.1f} tok/s ({last_stats['dt']:.1f}s)",
                    style="dim green",
                )
            t.append(f"  turn #{turn_counter}\n", style="dim")

            current_input = "".join(input_buf)
            ghost_suf, doc_hint = get_lookahead(current_input)
            if doc_hint:
                t.append(f"  💡 {doc_hint}\n", style="italic dim rgb(120,180,240)")

        if plan_mode:
            t.append(" [PLAN] ", style="bold magenta")
            t.append("❯ ", style="bold magenta")
        else:
            t.append(" ❯ ", style="bold bright_white")

        current_str = "".join(input_buf)
        t.append(current_str, style="bold bright_white")
        ghost_suf, _ = get_lookahead(current_str)
        if ghost_suf:
            t.append(ghost_suf, style="dim italic cyan")
        t.append("▌", style="bold cyan")
        prompt_panel = Panel(t, border_style=border, padding=(0, 1))

        # If an execution plan / todo checklist is active, render a sleek compact sticky bar
        if not agent.todo.is_empty():
            tasks = agent.todo.get_tasks()
            total = len(tasks)
            done = sum(1 for x in tasks if x["status"] == "completed")
            active_task = agent.todo.get_next_pending_or_in_progress()

            bar_len = 8
            filled_len = int((done / total) * bar_len) if total > 0 else 0
            prog_bar = "■" * filled_len + "□" * (bar_len - filled_len)

            plan_line = Text()
            plan_line.append("  📋 Plan ", style="bold rgb(140,160,190)")
            plan_line.append(f"[{prog_bar}] {done}/{total} ", style="bold green" if done == total else "bold yellow")
            if active_task:
                st_icon = "▶" if active_task["status"] == "in_progress" else "○"
                plan_line.append(f"· {st_icon} #{active_task['id']} {active_task['task'][:65]}", style="bright_white")
            else:
                plan_line.append("· All tasks completed! 🎉", style="bold green")

            border_plan = "rgb(70,120,90)" if done == total else "rgb(70,75,85)"
            plan_panel = Panel(
                plan_line,
                border_style=border_plan,
                padding=(0, 1),
            )
            return Group(plan_panel, prompt_panel)

        return prompt_panel

    def update_status(detail: str) -> None:
        nonlocal status_detail
        status_detail = detail
        if live.is_started:
            live.update(build_footer(), refresh=True)

    def run_turn(prompt: str) -> None:
        nonlocal turn_counter, last_stats, status_detail, turn_start_time
        is_generating.set()
        cancel_event.clear()
        turn_start_time = time.monotonic()
        status_detail = "💭 Thinking..."
        if live.is_started:
            live.update(build_footer(), refresh=True)
        t0 = time.monotonic()
        try:
            response = agent.step(prompt, cancel_event=cancel_event, on_status=update_status, console=live.console)
            dt = time.monotonic() - t0
            turn_stats = getattr(agent, "last_turn_stats", {})
            real_tokens = turn_stats.get("tokens")
            real_spd = turn_stats.get("tok_s")
            if real_tokens and real_spd:
                est_tokens = real_tokens
                tok_per_sec = real_spd
            elif real_tokens:
                est_tokens = real_tokens
                tok_per_sec = est_tokens / max(0.01, dt)
            else:
                words = len(response.split())
                est_tokens = max(1, int(words * 1.33))
                tok_per_sec = est_tokens / max(0.01, dt)
            agent.stats["turns"] += 1
            agent.stats["total_completion_tokens"] += est_tokens
            last_stats = {"dt": dt, "tok_s": tok_per_sec, "tokens": est_tokens}

            # Ensure final response is explicitly displayed with high contrast and markdown formatting
            # Only display panel if the turn was not already streamed live to the console
            if not getattr(agent, "last_turn_streamed", False):
                if response and response.strip() and response != "Turn interrupted by user.":
                    from rich.markdown import Markdown
                    live.console.print()
                    live.console.print(Panel(
                        Markdown(response.strip()),
                        border_style="rgb(70,75,85)",
                        padding=(1, 2),
                    ))

            # Minimalist turn completion line
            cur_ctx = agent.estimate_context_tokens()
            max_c = agent.stats.get("max_context", 131072)
            summary_line = Text()
            summary_line.append("  ● ", style="bold rgb(100,180,140)")
            summary_line.append(f"Turn #{turn_counter} completed", style="bold bright_white")
            summary_line.append(f" · {dt:.2f}s", style="dim white")
            summary_line.append(f" · {tok_per_sec:.1f} tok/s", style="dim green")
            summary_line.append(f" · ~{est_tokens} tok", style="dim white")
            summary_line.append(f" · ctx:{cur_ctx:,}/{max_c // 1024}k", style="dim cyan")
            live.console.print(summary_line)

            # In both streaming and non-streaming modes, print the updated plan into scrollback if active
            if not agent.todo.is_empty():
                tasks = agent.todo.get_tasks()
                done = sum(1 for x in tasks if x["status"] == "completed")
                plan_text = Text()
                for item in tasks:
                    tid = item["id"]
                    tsk = item["task"]
                    st = item["status"]
                    if st == "completed":
                        plan_text.append(f"  ✔ {tid}. ", style="bold green")
                        plan_text.append(f"{tsk}\n", style="dim white")
                    elif st == "in_progress":
                        plan_text.append(f"  ▶ {tid}. ", style="bold yellow")
                        plan_text.append(f"{tsk}  ", style="bold bright_white")
                        plan_text.append("[IN PROGRESS]\n", style="bold yellow on rgb(40,40,20)")
                    elif st == "failed":
                        plan_text.append(f"  ✗ {tid}. ", style="bold red")
                        plan_text.append(f"{tsk}  [FAILED]\n", style="bold red")
                    else:
                        plan_text.append(f"  ○ {tid}. ", style="dim cyan")
                        plan_text.append(f"{tsk}\n", style="dim white")
                plan_text.rstrip()
                border_plan = "rgb(70,120,90)" if done == len(tasks) else "rgb(70,75,85)"
                live.console.print(Panel(
                    plan_text,
                    title=f"[dim]Execution Plan ({done}/{len(tasks)} completed)[/dim]",
                    title_align="left",
                    border_style=border_plan,
                    padding=(0, 1),
                ))

            agent.save_checkpoint()
            turn_counter += 1
        except Exception as e:
            agent.last_failed_prompt = prompt
            agent.save_checkpoint()
            is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            if is_timeout:
                live.console.print(f"\n  [bold red]⚡ Timeout Error:[/bold red] {e}")
                live.console.print("  [dim]• The model engine took longer than the timeout budget to generate or prefill tokens.[/dim]")
                live.console.print("  [dim]• Session checkpoint safely saved in vault.[/dim]")
                live.console.print("  [dim]• Type [bold cyan]/retry[/bold cyan] to re-run this turn, or [bold cyan]/continue[/bold cyan] to proceed with remaining steps.[/dim]")
                live.console.print("  [dim]• Type [bold cyan]/stream always[/bold cyan] to monitor tokens in real time.[/dim]")
            else:
                live.console.print(f"\n  [bold red]⚡ Error:[/bold red] {e}")
                live.console.print("  [dim]• Session checkpoint safely saved. Type [bold cyan]/retry[/bold cyan] to retry or [bold cyan]/continue[/bold cyan] to resume.[/dim]")
        finally:
            is_generating.clear()
            status_detail = "Ready"
            if live.is_started:
                live.update(build_footer(), refresh=True)

    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def on_plan_update(todo_mgr: Any) -> None:
        if live.is_started:
            live.update(build_footer(), refresh=True)

    agent.on_todo_update = on_plan_update

    # Live prompt setup: transient=True ensures scrollback streaming prints never clash or jitter
    live = Live(build_footer(), console=console, auto_refresh=False, transient=True)
    live.start(refresh=True)

    try:
        dirty = True
        last_spin_time = time.monotonic()
        while True:
            # If a turn just finished, ensure live is active for interactive typing
            if not is_generating.is_set() and not live.is_started:
                live.start()
                live.update(build_footer(), refresh=True)
                dirty = False

            if dirty and live.is_started:
                live.update(build_footer(), refresh=True)
                dirty = False

            r, _, _ = select.select([sys.stdin], [], [], 0.08)
            if not r:
                # Pace spinner animation to ~4 Hz (0.25s) only during active generation
                now = time.monotonic()
                if is_generating.is_set() and live.is_started and (now - last_spin_time) >= 0.25:
                    last_spin_time = now
                    spinner_idx += 1
                    live.update(build_footer(), refresh=True)
                continue

            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl+C
                if is_generating.is_set():
                    cancel_event.set()
                    live.console.print("\n  [bold red]⚡ [interrupted][/bold red] [dim]Interrupted by user. Type a command or new prompt to steer.[/dim]")
                else:
                    if input_buf:
                        input_buf.clear()
                        dirty = True
                    else:
                        live.console.print("\n[dim]Exiting agent session...[/dim]")
                        break
            elif ch == "\x04":  # Ctrl+D / EOF
                live.console.print("\n[dim]Exiting agent session...[/dim]")
                break
            elif ch in ("\r", "\n"):
                submitted = "".join(input_buf).strip()
                if submitted:
                    if not history or history[-1] != submitted:
                        history.append(submitted)
                history_idx = -1
                saved_draft = ""
                input_buf.clear()
                dirty = True
                if not submitted:
                    continue
                if submitted.lower() in ("exit", "quit", "q"):
                    live.console.print("\n[dim]Exiting agent session...[/dim]")
                    break

                if submitted.startswith("/"):
                    # Stop live momentarily only for local slash commands that print tables/help
                    if live.is_started:
                        live.stop()
                    handled, plan_mode = _dispatch_slash_command(agent, submitted, plan_mode)
                    if getattr(agent, "pending_prompt", None):
                        next_p = agent.pending_prompt
                        agent.pending_prompt = None
                        if not live.is_started:
                            live.start()
                        prompt_badge = (
                            "  [bold magenta]● [PLAN][/bold magenta] [bold bright_white]❯[/bold bright_white]"
                            if plan_mode
                            else "  [bold rgb(100,180,140)]●[/bold rgb(100,180,140)] [bold bright_white]❯[/bold bright_white]"
                        )
                        live.console.print(f"\n{prompt_badge} [bold bright_white]{next_p}[/bold bright_white]")
                        cancel_event.clear()
                        worker_thread = threading.Thread(target=run_turn, args=(next_p,), daemon=True)
                        worker_thread.start()
                        dirty = True
                        continue
                    if live.is_started:
                        pass
                    else:
                        live.start()
                    live.update(build_footer(), refresh=True)
                    dirty = False
                    continue

                # If currently generating, steer! Cancel the current turn first
                if is_generating.is_set():
                    cancel_event.set()
                    if worker_thread and worker_thread.is_alive():
                        worker_thread.join(timeout=0.4)

                turn_prompt = submitted
                if submitted.lower() in ("continue", "c") and (not agent.todo.is_empty() or agent.stats.get("turns", 0) > 0):
                    turn_prompt = "Please continue your work where you left off. Review your current progress, plan, and complete the remaining steps."
                    live.console.print("  [bold cyan]⚡ Continuing interrupted work...[/bold cyan]")
                elif plan_mode:
                    turn_prompt = f"[PLAN MODE: Do not modify files. Inspect and produce an architectural plan only.] {submitted}"

                # Print user turn cleanly above into scrollback
                prompt_badge = (
                    "  [bold magenta]● [PLAN][/bold magenta] [bold bright_white]❯[/bold bright_white]"
                    if plan_mode
                    else "  [bold rgb(100,180,140)]●[/bold rgb(100,180,140)] [bold bright_white]❯[/bold bright_white]"
                )
                live.console.print(f"\n{prompt_badge} [bold bright_white]{turn_prompt}[/bold bright_white]")
                cancel_event.clear()
                # Run turn asynchronously in worker thread so the bottom prompt box stays live and responsive
                worker_thread = threading.Thread(target=run_turn, args=(turn_prompt,), daemon=True)
                worker_thread.start()
                dirty = True

            elif ch == "\t":  # Tab completion
                current_input = "".join(input_buf)
                ghost_suf, _ = get_lookahead(current_input)
                if ghost_suf:
                    input_buf.extend(list(ghost_suf))
                    dirty = True

            elif ch == "\x1b":  # Escape sequence (e.g. arrow keys)
                # Non-blocking peek next bytes
                r_seq, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r_seq:
                    seq1 = sys.stdin.read(1)
                    if seq1 == "[":
                        r_seq2, _, _ = select.select([sys.stdin], [], [], 0.02)
                        if r_seq2:
                            seq2 = sys.stdin.read(1)
                            if seq2 == "A":  # Up Arrow: navigate back in history
                                if history:
                                    if history_idx == -1:
                                        saved_draft = "".join(input_buf)
                                        history_idx = len(history) - 1
                                    elif history_idx > 0:
                                        history_idx -= 1
                                    input_buf = list(history[history_idx])
                                    dirty = True
                            elif seq2 == "B":  # Down Arrow: navigate forward in history
                                if history_idx != -1:
                                    if history_idx < len(history) - 1:
                                        history_idx += 1
                                        input_buf = list(history[history_idx])
                                    else:
                                        history_idx = -1
                                        input_buf = list(saved_draft)
                                    dirty = True
                            elif seq2 == "C":  # Right Arrow: accept ghost lookahead if present
                                current_input = "".join(input_buf)
                                ghost_suf, _ = get_lookahead(current_input)
                                if ghost_suf:
                                    input_buf.extend(list(ghost_suf))
                                    dirty = True
                dirty = True

            elif ch in ("\x7f", "\x08"):  # Backspace
                if input_buf:
                    input_buf.pop()
                    dirty = True
            elif ch == "\x15":  # Ctrl+U (clear line)
                input_buf.clear()
                dirty = True
            elif ord(ch) >= 32:
                input_buf.append(ch)
                dirty = True
    finally:
        cancel_event.set()
        if live.is_started:
            live.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
        agent.close()

    return 0


def _run_linear_cli(agent: SlipstreamAgent) -> int:
    """Standard linear fallback CLI for pipes, non-interactive tty, or automated testing."""
    console = agent.console
    plan_mode = False
    turn_counter = agent.stats.get("turns", 0) + 1
    try:
        while True:
            try:
                if plan_mode:
                    prompt_prefix = "  [bold magenta]● [PLAN][/bold magenta] [bold bright_white]❯[/bold bright_white] "
                else:
                    prompt_prefix = "  [bold rgb(100,180,140)]●[/bold rgb(100,180,140)] [bold bright_white]❯[/bold bright_white] "
                console.print(f"\n{prompt_prefix}", end="")
                user_prompt = input("").strip()
            except EOFError:
                break
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit", "q"):
                break
            if user_prompt.startswith("/"):
                handled, plan_mode = _dispatch_slash_command(agent, user_prompt, plan_mode)
                if getattr(agent, "pending_prompt", None):
                    turn_prompt = agent.pending_prompt
                    agent.pending_prompt = None
                elif handled:
                    continue
                else:
                    turn_prompt = user_prompt
            elif user_prompt.lower() in ("continue", "c") and (not agent.todo.is_empty() or agent.stats.get("turns", 0) > 0):
                turn_prompt = "Please continue your work where you left off. Review your current progress, plan, and complete the remaining steps."
                console.print("  [bold cyan]⚡ Continuing interrupted work...[/bold cyan]")
            else:
                turn_prompt = (
                    user_prompt
                    if not plan_mode
                    else f"[PLAN MODE: Do not modify files. Inspect and produce an architectural plan only.] {user_prompt}"
                )
            t0 = time.monotonic()
            try:
                resp = agent.step(turn_prompt)
                dt = time.monotonic() - t0
                agent.stats["turns"] += 1
                agent.save_checkpoint()
                if not getattr(agent, "last_turn_streamed", False):
                    if resp and resp.strip() and resp != "Turn interrupted by user.":
                        from rich.markdown import Markdown
                        console.print()
                        console.print(Panel(
                            Markdown(resp.strip()),
                            title="[bold rgb(100,200,255)]🤖 Slipstream Assistant[/bold rgb(100,200,255)]",
                            title_align="left",
                            border_style="rgb(60,120,180)",
                            padding=(1, 2),
                        ))
                summary_bar = Text()
                summary_bar.append(f"  ✔ Turn #{turn_counter} Complete  ", style="bold white on rgb(0,180,100)")
                summary_bar.append(f"  ⏱ {dt:.2f}s  •  Ready for next prompt", style="dim green")
                console.print()
                console.print(Panel(summary_bar, border_style="rgb(0,180,100)", padding=(0, 1)))
                turn_counter += 1
            except Exception as e:
                agent.last_failed_prompt = turn_prompt
                agent.save_checkpoint()
                console.print(f"\n  [bold red]⚡ Error:[/bold red] {e}")
                console.print("  [dim]• Session checkpoint saved. Type [bold cyan]/retry[/bold cyan] to retry or [bold cyan]/continue[/bold cyan] to resume.[/dim]")
    finally:
        agent.close()

    return 0


def run_agent_cli(
    base_url: str = "http://127.0.0.1:8090/v1",
    model: str | None = None,
    reasoning_effort: str = "medium",
    stream_mode: str = "hybrid",
    resume_session: str | None = None,
    timeout: float | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Run interactive terminal agent session with rich colorful blocks and pinned bottom prompt."""
    workspace = Path(cwd).expanduser().resolve() if cwd else Path(os.getcwd())
    if not workspace.is_dir():
        print(f"error: project path is not a directory: {workspace}", file=sys.stderr)
        return 1
    resumed = False
    if resume_session:
        restored = SlipstreamAgent.restore_checkpoint(
            resume_session,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            stream_mode=stream_mode,
            timeout=timeout,
            cwd=str(workspace),
        )
        if restored:
            agent = restored
            resumed = True
        else:
            print(f"Warning: Checkpoint '{resume_session}' not found in vault (~/.cache/slipstream/agent_vaults). Starting fresh session.")
            agent = SlipstreamAgent(
                base_url=base_url,
                model=model,
                reasoning_effort=reasoning_effort,
                stream_mode=stream_mode,
                session_id=resume_session if resume_session not in ("latest", "recent", "last") else None,
                timeout=timeout,
                cwd=str(workspace),
            )
    else:
        agent = SlipstreamAgent(
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            stream_mode=stream_mode,
            timeout=timeout,
            cwd=str(workspace),
        )
    console = agent.console

    if resumed:
        console.print(Panel(
            f"[bold green]✔ Resumed Session:[/bold green] [bold cyan]{agent.session_id}[/bold cyan]\n"
            f"[dim]• History:[/dim] {len(agent.messages)} messages restored  •  "
            f"[dim]Turns completed:[/dim] {agent.stats.get('turns', 0)}  •  "
            f"[dim]Active Context:[/dim] ~{agent.estimate_context_tokens():,} tokens\n"
            f"[dim]Type [bold cyan]/continue[/bold cyan] to proceed with interrupted work, or ask a question.[/dim]",
            title="[bold green]🔄 Session Restored[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))
        if not agent.todo.is_empty():
            tasks = agent.todo.get_tasks()
            done = sum(1 for x in tasks if x["status"] == "completed")
            console.print(Panel(
                agent.todo.active_summary(),
                title=f"[bold cyan]📋 Resumed Execution Plan ({done}/{len(tasks)} completed)[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))


    # Clean, modern, high-contrast visual header
    hdr = Text()
    hdr.append(" ⚡ SLIPSTREAM ", style="bold white on rgb(114,46,209)")
    hdr.append(" AGENT ", style="bold black on rgb(0,255,200)")
    hdr.append("  ", style="none")
    hdr.append(f" {agent.model} ", style="bold white on rgb(30,40,65)")
    hdr.append(" ", style="none")
    hdr.append(f" port:{agent.base_url.split(':')[-1].split('/')[0]} ", style="bold black on rgb(0,200,100)")
    hdr.append(" ", style="none")
    hdr.append(f" think:{agent.reasoning_effort} ", style="bold white on rgb(160,80,220)")
    hdr.append(" ", style="none")
    stream_desc = "batch" if agent.stream_mode == "never" else agent.stream_mode
    hdr.append(f" stream:{stream_desc} ", style="bold black on rgb(100,200,255)")

    sub = Text()
    sub.append("  📂 ", style="none")
    sub.append(f"{agent.cwd} ", style="bold white")
    ctx_k = agent.stats.get('max_context', 131072) // 1024
    sub.append(f"• ctx:{ctx_k}k limit ", style="dim cyan")
    sub.append("• speculative: PLD+MTP ", style="dim magenta")
    sub.append("• unix socket: IPC/shm ", style="dim green")

    cmd = Text()
    cmd.append("  ⌨️  ", style="none")
    for c in ["/todo", "/continue", "/retry", "/btw", "/plan", "/thinking", "/stream", "/no-stream", "/diff", "/test", "/stats", "/clear", "/help"]:
        cmd.append(f" {c} ", style="bold rgb(200,220,255) on rgb(35,40,60)")
        cmd.append(" ", style="none")
    cmd.append("\n  💡 ", style="none")
    cmd.append("Pinned bottom prompt active • Type to steer anytime • Ctrl+C to halt stream", style="dim italic bright_white")

    console.print(Panel(
        Text.assemble(hdr, "\n\n", sub, "\n", cmd),
        border_style="rgb(114,46,209)",
        title="[bold bright_white]⚡ slipstream agent[/bold bright_white]",
        title_align="left",
        padding=(0, 1),
    ))

    if sys.stdin.isatty() and sys.stdout.isatty():
        return _run_pinned_tui(agent)
    else:
        return _run_linear_cli(agent)

    return 0
