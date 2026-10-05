#!/usr/bin/env python3
"""Run one engine command without risking the machine.

Two engines at once (or one with too big an expert cache) pin more memory
than a 64 GB Mac has, and macOS freezes until its watchdog restarts it -
this happened twice on 2026-09-22. This wrapper:

  * refuses to start if another Splash / llama.cpp engine is running;
  * polls free memory every 0.25 s and kills the command (and its children)
    if it falls under --floor-gib (default 4);
  * kills the command after --max-seconds (default 900), so a hung run
    cannot sit on its memory while the next one starts.

Usage: dev/benchmarks/guarded.py [--floor-gib N] [--max-seconds S] -- <command> [args...]
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

ENGINE_PROGRAMS = {
    "generate-sample",
    "splash",
    "slipstream",
    "slipstream-v2",
    "llama-server",
    "splash-q8",
}


def other_engines():
    """Running engines, judged by program name (not by text in a command
    line, which would match any shell that merely mentions one)."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,comm="], capture_output=True, text=True
    ).stdout
    me = os.getpid()
    found = []
    for line in out.splitlines():
        pid, _, program = line.strip().partition(" ")
        if int(pid) == me:
            continue
        name = os.path.basename(program.strip())
        if name in ENGINE_PROGRAMS:
            found.append(line.strip())
        elif name.lower().startswith("python"):
            args = subprocess.run(
                ["ps", "-o", "args=", "-p", pid], capture_output=True, text=True
            ).stdout
            if "server.server" in args or "llama_cpp.server" in args:
                found.append(f"{pid} {args.strip()[:100]}")
    return found


def free_gib():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {
        k: int(v)
        for k, v in re.findall(
            r"^(Pages [a-z ]+|File-backed pages):\s+(\d+)\.", out, re.M
        )
    }
    usable = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
        + pages.get("Pages purgeable", 0)
    )
    return usable * page / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-gib", type=float, default=4.0)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given")
    running = other_engines()
    if running:
        print(
            "guarded: another engine is running, not starting:\n  "
            + "\n  ".join(running),
            file=sys.stderr,
        )
        return 3
    proc = subprocess.Popen(cmd, start_new_session=True)
    lowest = free_gib()
    deadline = time.monotonic() + a.max_seconds
    while proc.poll() is None:
        now = free_gib()
        lowest = min(lowest, now)
        if now < a.floor_gib:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(
                f"guarded: killed - free memory fell to {now:.1f} GiB (floor {a.floor_gib})",
                file=sys.stderr,
            )
            return 4
        if time.monotonic() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(
                f"guarded: killed - still running after {a.max_seconds:.0f} s",
                file=sys.stderr,
            )
            return 5
        time.sleep(0.25)
    print(f"guarded: lowest free memory {lowest:.1f} GiB", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
