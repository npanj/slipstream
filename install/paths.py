"""Immutable program files and writable per-user data, for source or release."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
DATA = (
    (Path.home() / "Library/Application Support/Slipstream")
    if (Path.home() / "Library/Application Support/Slipstream").exists()
    else (Path.home() / "Library/Application Support/Slipstream-v2")
) if PACKAGED else ROOT
# One model store for every installation, source or release, and for the
# menubar app, so a model is downloaded once. SLIPSTREAM_MODELS moves it, e.g.
# to an external disk.
MODELS = Path(
    os.environ.get("SLIPSTREAM_MODELS") or Path.home() / ".slipstream/models"
).expanduser()
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
_BIN = ROOT / ("engine/slipstream" if PACKAGED else "build/slipstream")
_V2_BIN = ROOT / ("engine/slipstream-v2" if PACKAGED else "build/slipstream-v2")
_SPLASH_BIN = ROOT / ("engine/splash" if PACKAGED else "build/splash")
BINARY = (
    _BIN
    if _BIN.exists()
    else _V2_BIN
    if _V2_BIN.exists()
    else _SPLASH_BIN
    if _SPLASH_BIN.exists()
    else _BIN
)
