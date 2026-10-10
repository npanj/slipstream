"""Persistent warm PTY shell worker for sub-1.5ms agent tool execution."""

from __future__ import annotations

import os
import pty
import select
import signal
import struct
import termios
import time
from pathlib import Path


class WarmPtyShell:
    """Manages a persistent background shell inside a pseudo-terminal (PTY).

    Maintains environment variables, current working directory, and virtualenvs
    across successive command dispatches without process spawn overhead.
    """

    def __init__(self, cwd: str | Path | None = None, shell: str = "/bin/zsh"):
        self.shell = shell
        self.cwd = str(Path(cwd).resolve()) if cwd else os.getcwd()
        self.master_fd: int | None = None
        self.slave_fd: int | None = None
        self.pid: int | None = None
        self._sentinel_counter = 0
        self._spawn()

    def _spawn(self) -> None:
        self.master_fd, self.slave_fd = pty.openpty()

        # Set raw mode on master
        mode = termios.tcgetattr(self.master_fd)
        mode[3] = mode[3] & ~termios.ECHO  # Disable echo
        termios.tcsetattr(self.master_fd, termios.TCSANOW, mode)

        self.pid = os.fork()
        if self.pid == 0:
            # Child process
            os.close(self.master_fd)
            os.setsid()
            os.dup2(self.slave_fd, 0)
            os.dup2(self.slave_fd, 1)
            os.dup2(self.slave_fd, 2)
            if self.slave_fd > 2:
                os.close(self.slave_fd)
            os.chdir(self.cwd)
            # Execute interactive shell without greeting
            env = dict(os.environ)
            env["TERM"] = "dumb"
            env["PAGER"] = "cat"
            env["PROMPT"] = ""
            os.execve(self.shell, [self.shell, "-f", "+Z"], env)

        # Parent process
        os.close(self.slave_fd)
        self.slave_fd = None
        # Drain initial shell prompt
        time.sleep(0.05)
        self._drain()

    def _drain(self) -> str:
        output = []
        while True:
            r, _, _ = select.select([self.master_fd], [], [], 0.01)
            if not r:
                break
            try:
                chunk = os.read(self.master_fd, 4096)
                if not chunk:
                    break
                output.append(chunk.decode("utf-8", "replace"))
            except OSError:
                break
        return "".join(output)

    def run_command(self, cmd: str, timeout: float = 30.0) -> tuple[int, str]:
        """Execute command in warm shell, returning (exit_code, output)."""
        if self.master_fd is None:
            raise RuntimeError("PTY shell is closed")

        self._sentinel_counter += 1
        sentinel = f"__SLIPSTREAM_PTY_EXIT_{self._sentinel_counter}__"
        # Chain exit code on same command line to prevent prompt execution resetting $?
        wrapped = f" {cmd} ; __rc=$? ; echo \"{sentinel}:$__rc\"\n"
        os.write(self.master_fd, wrapped.encode("utf-8"))

        deadline = time.monotonic() + timeout
        buf = []
        exit_code = 0

        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            r, _, _ = select.select([self.master_fd], [], [], min(remaining, 0.1))
            if not r:
                continue
            try:
                chunk = os.read(self.master_fd, 4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", "replace")
                buf.append(text)
                combined = "".join(buf)
                if sentinel in combined:
                    # Found sentinel
                    parts = combined.split(sentinel)
                    output = parts[0].rstrip("\r\n")
                    trailer = parts[1]
                    # Parse exit code
                    if ":" in trailer:
                        code_str = trailer.split(":", 1)[1].split("\n", 1)[0].strip()
                        try:
                            exit_code = int(code_str)
                        except ValueError:
                            exit_code = 0
                    return exit_code, output
            except OSError:
                break

        # Timed out
        return -1, "".join(buf) + f"\n[Command timed out after {timeout}s]"

    def close(self) -> None:
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except OSError:
                pass
            self.pid = None
