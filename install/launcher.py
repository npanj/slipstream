#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server."""

import argparse
import fcntl
import http.client
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from . import catalog, clients, paths
    from . import models as model_artifacts
except ImportError:  # Executed directly by the source or packaged entry point.
    import catalog
    import clients
    import paths

    import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
PORT = 8090
BASE_URL = f"http://127.0.0.1:{PORT}"


class LauncherError(RuntimeError):
    pass


def _request_json(path, timeout=2):
    request = urllib.request.Request(BASE_URL + path)
    if key := (
        os.environ.get("SLIPSTREAM_V2_API_KEY")
        or os.environ.get("SLIPSTREAM_API_KEY")
        or os.environ.get("SPLASH_API_KEY")
    ):
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise LauncherError(
                "Slipstream v2 authentication failed; set SLIPSTREAM_V2_API_KEY to the server's key"
            ) from None
        return None
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return None


def _running_status():
    status = _request_json("/status", timeout=10)
    if not isinstance(status, dict):
        return None
    return status


def _ensure_installed(model_id):
    if not paths.PACKAGED:
        for command in (
            ["make", "platform-check", "install-environment"],
            ["make", "-j4", "all"],
        ):
            if subprocess.run(command, cwd=ROOT).returncode:
                raise LauncherError("source build failed; see the output above")
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(paths.MODELS),
        "--model",
        model_id,
        "prepare",
    ]
    if subprocess.run(command, cwd=ROOT).returncode:
        raise LauncherError("model download or verification failed")


def pull(args):
    """Download a model into the model store without serving it."""
    if not paths.PACKAGED:
        command = ["make", "platform-check", "install-environment"]
        if subprocess.run(command, cwd=ROOT).returncode:
            raise LauncherError("source environment setup failed; see the output above")
    command = [
        str(paths.PYTHON),
        "-u",
        str(ROOT / "install/models.py"),
        "--models",
        str(paths.MODELS),
        "--model",
        args.model,
    ]
    if getattr(args, "check", False):
        command += ["check", "--json"] if getattr(args, "json", False) else ["check"]
    else:
        command.append("prepare")
    # Replaced rather than waited for, so an interrupt (Ctrl+C, or a frontend
    # stopping the download) reaches the download itself, which keeps its
    # partial files for the next pull to resume.
    os.execv(command[0], command)


def _detect_model_format_and_arch(model_path: Path):
    """Detect format ('gguf', 'mlx', 'package') and architecture ('qwen4exp', 'qwen38')."""
    if model_path.is_dir():
        if (model_path / "manifest.json").is_file():
            return "package", None
        if (model_path / "prepared/manifest.json").is_file():
            return "package", None
        gguf_files = sorted(p for p in model_path.glob("*.gguf") if p.is_file())
        if gguf_files:
            try:
                with open(gguf_files[0], "rb") as f:
                    header = f.read(model_artifacts.GGUF_HEADER_BYTES)
                arch = model_artifacts.gguf_architecture(header)
            except Exception:
                arch = None
            if arch == "qwen4exp":
                return "gguf", "qwen4exp"
            return "gguf", "qwen38"
        safetensors = sorted(p for p in model_path.glob("*.safetensors") if p.is_file())
        if safetensors:
            config_file = model_path / "config.json"
            if config_file.is_file():
                try:
                    config = json.loads(config_file.read_text())
                    if config.get("model_type") == "qwen4exp" or config.get("num_hidden_layers") == 48:
                        return "mlx", "qwen4exp"
                except Exception:
                    pass
            return "mlx", "qwen38"
    elif model_path.is_file() and model_path.name.endswith(".gguf"):
        try:
            with open(model_path, "rb") as f:
                header = f.read(model_artifacts.GGUF_HEADER_BYTES)
            arch = model_artifacts.gguf_architecture(header)
        except Exception:
            arch = None
        if arch == "qwen4exp":
            return "gguf", "qwen4exp"
        return "gguf", "qwen38"
    return "unknown", None


def _prepare_model(model_path: Path, consume_source: bool = False):
    """Convert GGUF shards or MLX checkpoints into model_path/prepared, once."""
    if model_path.is_dir():
        prepared_dir = model_path / "prepared"
    else:
        prepared_dir = model_path.parent / (model_path.stem + "-prepared")

    manifest = prepared_dir / "manifest.json"
    layer0 = prepared_dir / "target/layer-0.bin"
    if manifest.is_file() and layer0.is_file():
        if consume_source and model_path.is_dir():
            model_artifacts.remove_gguf_source(model_path)
        return prepared_dir

    kind, arch = _detect_model_format_and_arch(model_path)
    if kind == "package":
        return model_path if (model_path / "manifest.json").is_file() else (model_path / "prepared")

    if kind == "gguf":
        print(f"[Slipstream] Preparing GGUF model ({arch or 'qwen'}) from {model_path}...", flush=True)
        if arch == "qwen4exp":
            converter = ROOT / "models/qwen4exp/tools/convert_qwen4exp_gguf.py"
        else:
            converter = ROOT / "models/qwen38/tools/convert_qwen38_gguf.py"

        cmd = [
            str(paths.PYTHON),
            "-u",
            str(converter),
            "--model-dir",
            str(model_path),
            "--output",
            str(prepared_dir),
        ]
        if consume_source:
            cmd.append("--consume-source")
        prepared = subprocess.run(cmd)
        if prepared.returncode != 0:
            raise LauncherError(f"preparing {model_path} failed; see the output above")
        if consume_source and model_path.is_dir():
            model_artifacts.remove_gguf_source(model_path)
        return prepared_dir

    if kind == "mlx":
        print(f"[Slipstream] Preparing MLX checkpoint from {model_path}...", flush=True)
        if arch == "qwen4exp":
            converter = ROOT / "models/qwen4exp/tools/convert_qwen4exp.py"
            cmd = [
                str(paths.PYTHON),
                "-u",
                str(converter),
                str(model_path),
                "--output",
                str(prepared_dir),
            ]
        else:
            converter = ROOT / "models/qwen38/tools/convert_qwen38_mlx.py"
            cmd = [
                str(paths.PYTHON),
                "-u",
                str(converter),
                "--source",
                str(model_path),
                "--output",
                str(prepared_dir),
            ]
        prepared = subprocess.run(cmd)
        if prepared.returncode != 0:
            raise LauncherError(f"preparing MLX {model_path} failed; see the output above")
        return prepared_dir

    raise LauncherError(
        f"Directory or file {model_path} does not contain supported model files (GGUF, MLX, or manifest.json)"
    )


def _prepare_gguf(model_path, consume_source=False):
    return _prepare_model(model_path, consume_source=consume_source)


def _serve_lock_owner(lock):
    try:
        lock.seek(0)
        owner = json.load(lock)
    except (OSError, UnicodeError, ValueError):
        return ""
    if not isinstance(owner, dict):
        return ""
    pid, model, port = owner.get("pid"), owner.get("model"), owner.get("port")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(model, str)
        or not model
        or not model.isprintable()
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return ""
    return f" (PID {pid}, model {model}, port {port})"


def serve(args):
    # Keep this descriptor across exec: the foreground server owns the lock
    # until it exits.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNTIME_DIR / "serve.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                f"Slipstream v2 is already serving{_serve_lock_owner(lock)}; "
                "stop it with Ctrl+C first"
            ) from None
        port = getattr(args, "port", None) or PORT
        host = getattr(args, "host", None) or "127.0.0.1"
        lock.seek(0)
        lock.truncate()
        json.dump(
            {"pid": os.getpid(), "model": args.model, "port": port, "host": host}, lock
        )
        lock.flush()
        # Fail before downloads/builds if another service owns the default port.
        # The HTTP server also binds before loading weights, closing the race.
        with socket.socket() as probe:
            # Match the HTTP listener: closed connections in TIME_WAIT must
            # not block a restart; a live listener still owns the address.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError as error:
                raise LauncherError(
                    f"cannot listen on {host}:{port} ({error.strerror or error}); "
                    "stop the service using it or choose another address"
                ) from None
        model_str = args.model
        model_path = Path(model_str).expanduser()
        if model_path.exists() or model_str.startswith(("/", "./", "../", "~")):
            # A folder of the user's own is never used up; only the model store's
            # downloads, which can be fetched again, are.
            kind, arch = _detect_model_format_and_arch(model_path)
            if kind == "package":
                root = model_path if (model_path / "manifest.json").exists() else (model_path / "prepared")
                model_id = f"local/{model_path.name}"
            elif kind in ("gguf", "mlx"):
                root = _prepare_model(model_path, consume_source=False)
                model_id = f"local/{model_path.name}"
            else:
                raise LauncherError(
                    f"Directory or file {model_path} does not contain supported model files (GGUF, MLX, or manifest.json)"
                )
        else:
            _ensure_installed(args.model)
            root = model_artifacts.installed_root(paths.MODELS, args.model)
            if model_artifacts.is_gguf_installation(root):
                root = _prepare_model(root, consume_source=not args.keep_gguf)
            model_id = args.model
        command = [
            str(paths.PYTHON),
            "-u",
            str(ROOT / "server/server.py"),
            str(root / "target"),
            str(root / "draft"),
            "--tokenizer",
            str(root / "tokenizer"),
            "--model",
            model_id,
            "--binary",
            str(paths.BINARY),
            "--max-memory",
            "auto" if args.max_memory is None else str(args.max_memory),
            "--max-context",
            "auto" if args.max_context is None else str(args.max_context),
        ]
        command.extend(["--host", host, "--port", str(port)])
        if args.max_image_pixels is not None:
            command.extend(["--max-image-pixels", str(args.max_image_pixels)])
        if args.no_webui:
            command.append("--no-webui")
        for host in args.allowed_host:
            command.extend(["--allowed-host", host])
        environment = dict(
            os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error"
        )
        if args.api_key is not None:
            environment["SLIPSTREAM_V2_API_KEY"] = args.api_key
            environment["SLIPSTREAM_API_KEY"] = args.api_key
            environment["SPLASH_API_KEY"] = args.api_key
        # Detached, because execve replaces this process a line later and a
        # thread would not survive it. Failure is silent by design.
        catalog.spawn_refresh()
        os.set_inheritable(lock.fileno(), True)
        os.execve(command[0], command, environment)


def coding_client(args):
    path = clients.find_executable(args.command)
    snapshot = _running_status()
    if snapshot is None:
        raise LauncherError(
            "No ready Slipstream v2 server. Run 'slipstream-v2 serve --model <HF_REPO_ID>' "
            "in another terminal first."
        )
    catalog = _request_json("/v1/models")
    models = catalog.get("data", []) if isinstance(catalog, dict) else []
    if (
        not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], dict)
        or models[0].get("owned_by") not in ("slipstream-v2", "slipstream", "splash")
    ):
        raise LauncherError("Could not identify the local Slipstream v2 server")
    model, context = models[0].get("id"), snapshot.get("maximum_context_tokens")
    if type(context) is not int or context <= 0:
        raise LauncherError(
            "Slipstream v2 is running but its context limit is not available yet; wait and retry"
        )
    command, environment = clients.command(
        args.command,
        path,
        BASE_URL,
        model,
        context,
        RUNTIME_DIR,
        client_args=args.client_args,
    )
    print(f"Starting {args.command}: {model} · {context:,} context tokens", flush=True)
    if args.command == "claude":
        print(
            "Claude hosted WebSearch is unavailable. "
            "WebFetch, local tools and MCP are unchanged.",
            flush=True,
        )
    elif args.command == "codex":
        print(
            "Codex hosted WebSearch is disabled: Slipstream v2 does not provide "
            "OpenAI's search service. Local tools and MCP are unchanged.",
            flush=True,
        )
    os.execvpe(path, command, environment)


def _parse_max_memory(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    suffixes = {
        unit + suffix: 1024**power
        for power, unit in enumerate(("K", "M", "G"), 1)
        for suffix in ("", "B", "IB")
    }
    multiplier = 1
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            normalized, multiplier = normalized[: -len(suffix)], suffixes[suffix]
            break
    try:
        result = int(normalized) * multiplier
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 32G") from None
    if not 1 <= result <= 2**63 - 1:
        raise argparse.ArgumentTypeError("use a positive value such as 32G")
    return result


def _parse_max_context(value):
    normalized = value.strip().upper()
    if normalized == "AUTO":
        return None
    try:
        result = (
            int(normalized[:-1]) * 1024 if normalized.endswith("K") else int(normalized)
        )
    except ValueError:
        raise argparse.ArgumentTypeError("use a value such as 100K") from None
    if not 1 <= result <= 262144:
        raise argparse.ArgumentTypeError("must be between 1 and 256K tokens")
    return result


def _version():
    if not paths.PACKAGED:
        return "Slipstream v2 (source checkout)"
    return "Slipstream v2 " + str(
        json.loads((paths.ROOT / "release.json").read_text())["version"]
    )


def _parse_model_spec(value):
    path = Path(value).expanduser()
    if path.exists() or value.startswith(("/", "./", "../", "~")):
        return str(path.resolve())
    return model_artifacts.parse_repo_id(value)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args = []
    if argv and argv[0] in clients.INSTALL_URLS:
        argv, client_args = argv[:1], argv[1:]
        if client_args[:1] == ["--"]:
            client_args = client_args[1:]
    elif "--" in argv:
        boundary = argv.index("--")
        argv, client_args = argv[:boundary], argv[boundary + 1 :]
    parser = argparse.ArgumentParser(prog="slipstream-v2", description=__doc__)
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="run the local server; Ctrl+C stops it")
    server.add_argument(
        "--model",
        type=_parse_model_spec,
        required=True,
        metavar="OWNER/REPO_OR_PATH",
        help="Hugging Face repository or local directory containing a model package or GGUF shards",
    )
    server.add_argument(
        "--max-memory",
        type=_parse_max_memory,
        help="Metal budget ceiling, e.g. 28G (default: auto)",
    )
    server.add_argument(
        "--max-context",
        type=_parse_max_context,
        help="context limit, e.g. 100K (default: auto)",
    )
    server.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="additional HTTP Host name to accept (repeatable)",
    )
    server.add_argument(
        "--max-image-pixels", type=int, help="maximum resized pixels per image"
    )
    server.add_argument(
        "--api-key",
        default=(
            os.environ.get("SLIPSTREAM_V2_API_KEY")
            or os.environ.get("SLIPSTREAM_API_KEY")
            or os.environ.get("SPLASH_API_KEY")
        ),
        help="API key (default: SLIPSTREAM_V2_API_KEY environment variable)",
    )
    server.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: 127.0.0.1, this Mac only); 0.0.0.0 "
        "accepts connections from the network, so set --api-key with it",
    )
    server.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"HTTP serving port (default: {PORT})",
    )
    server.add_argument("--no-webui", action="store_true", help="disable the chat page")
    server.add_argument(
        "--keep-gguf",
        action="store_true",
        help="keep a downloaded GGUF model's files after preparing it (needs about twice "
        "the model's size on disk); by default they are used up while preparing",
    )
    puller = commands.add_parser(
        "pull", help="download a model from Hugging Face without serving it"
    )
    puller.add_argument(
        "model",
        type=model_artifacts.parse_repo_id,
        metavar="OWNER/REPO",
        help="Hugging Face repository: a model package, or GGUF files",
    )
    puller.add_argument(
        "--check",
        action="store_true",
        help="only say whether the model can be pulled and served, and its size; download nothing",
    )
    puller.add_argument(
        "--json", action="store_true", help="with --check: print one JSON object"
    )
    for name in clients.INSTALL_URLS:
        commands.add_parser(name, help=f"connect {name} to the running server")
    args = parser.parse_args(argv)
    if args.command == "serve" and args.api_key is not None:
        if not args.api_key or any(ord(c) <= 32 or ord(c) >= 127 for c in args.api_key):
            parser.error("API key must contain only visible ASCII characters")
    if args.command == "pull" and args.json and not args.check:
        parser.error("--json is only supported with --check")
    if client_args and args.command in ("serve", "pull"):
        parser.error("arguments after -- are only supported for coding clients")
    args.client_args = client_args
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "serve":
            return serve(args)
        if args.command == "pull":
            return pull(args)
        return coding_client(args)
    except (LauncherError, clients.ClientError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
