#!/usr/bin/env python3
"""Enforce production dependency boundaries."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".metal"}
INCLUDE = re.compile(r'^\s*#(?:include|import)\s+["<]([^">]+)[">]', re.MULTILINE)
OPERATOR_WORKSPACE_POLICY = re.compile(
    r"\b(?:PrefillAttentionWave|prefillAttentionTiles|"
    r"kQ8VerifySplits|kQ8PrefillAttentionTileRows|"
    r"moeMaximumTiles|kMoePrefillTileRows|kMoeDecodeTileRows|"
    r"Q4DecodeKind|Q4DecodeShape|Q4PrefillShape|kQ4PrefillTileRows|"
    r"narrowAffineKind|narrowResidualKind|headKind|gdnInputGroups|"
    r"attentionGroups|addPrefill128|LinearTile|LinearConfig|LinearSimdgroups|"
    r"PrefillSplitMultiplier|PrefillAttentionConfig|VerifySplitCount|"
    r"VerifyAttentionConfig|AttentionScalePlacement|MoeExpertTile|MoeConfig|"
    r"DraftAttentionConfiguration|selectorShards)\b"
)
# The standard library's clocks that also count sleep, and the timed waits
# that measure on them; production measures time on AwakeClock.
SLEEP_COUNTING_CLOCK = re.compile(
    r"\b(?:steady_clock|high_resolution_clock|wait_for|try_lock_for"
    r"|try_acquire_for)\b"
)


def production_sources() -> list[Path]:
    return sorted(
        path
        for root in ("runtime", "models")
        for path in (ROOT / root).rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_server_dependencies() -> list[str]:
    # Request bridges and model-output handling must not import HTTP or prompt
    # preparation. The wire transport stays independent of those adapters.
    forbidden = {
        "backend": {"frontend", "api_shapes"},
        "constraints": {"frontend", "backend", "output", "api_shapes"},
        "output": {"frontend", "backend", "constraints", "api_shapes"},
        "runtime": {"frontend", "backend", "constraints", "output", "api_shapes"},
    }
    errors = []
    for path in sorted((ROOT / "server").glob("*.py")):
        if path.stem == "server":
            continue
        blocked = {"server", *forbidden.get(path.stem, ())}
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.module is None or (node.module == "server" and not node.level):
                    dependencies.update(alias.name for alias in node.names)
                else:
                    dependencies.add(node.module.removeprefix("server.").split(".")[0])
            elif isinstance(node, ast.Import):
                dependencies.update(
                    alias.name.removeprefix("server.").split(".")[0]
                    for alias in node.names
                )
        for dependency in sorted(dependencies & blocked):
            errors.append(f"{relative(path)}: imports upper serving layer {dependency}")
    return errors


def check() -> list[str]:
    errors = check_server_dependencies()
    obsolete_roots = (
        ROOT / "runtime" / "kernels",
        ROOT / "runtime" / "src" / "metal",
    )
    for path in obsolete_roots:
        if path.exists():
            errors.append(f"obsolete production directory exists: {relative(path)}")

    forbidden_metal_dependencies = ("engine/", "model/", "models/")
    forbidden_model_dependencies = ("engine/",)
    forbidden_operator_dependencies = ("engine/", "model/", "models/")
    # Shared model code reaches a concrete model (models/<name>/) only here.
    # A new model adds itself to these files and nowhere else in runtime/.
    model_registry_sources = {
        "runtime/model/ModelDescriptor.hpp",
        "runtime/model/ModelFactory.hpp",
        "runtime/model/QwenTarget.cpp",
        "runtime/model/Runtime.mm",
    }
    engine_assembly_sources = {
        "runtime/engine/Bootstrap.hpp",
        "runtime/engine/Bootstrap.mm",
        "runtime/engine/RuntimeResources.hpp",
        "runtime/engine/RuntimeResources.mm",
    }
    concrete_model_headers = (
        "model/ModelFactory.hpp",
        "model/QwenState.hpp",
        "model/QwenTarget.hpp",
        "model/Runtime.hpp",
        "model/WeightStore.hpp",
    )
    concrete_model_symbols = re.compile(r"\bmodel::(?:Qwen\w*|DFlash\w*|Runtime)\b")
    client_names = re.compile(r"\b(?:Claude Code|OpenCode|Codex|Hermes)\b", re.I)

    for path in production_sources():
        name = relative(path)
        text = path.read_text(errors="replace")
        includes = INCLUDE.findall(text)
        for include in includes:
            if include.startswith("tuning/"):
                errors.append(f"{name}: production depends on offline tuning {include}")
        if name.startswith("runtime/metal/"):
            for include in includes:
                if any(part in include for part in forbidden_metal_dependencies):
                    errors.append(
                        f"{name}: Metal depends on production layer {include}"
                    )
        if name.startswith("runtime/model/"):
            for include in includes:
                if any(part in include for part in forbidden_model_dependencies):
                    errors.append(f"{name}: model depends on engine layer {include}")
                if include.startswith("models/") and name not in model_registry_sources:
                    errors.append(
                        f"{name}: shared model code depends on concrete model {include}"
                    )
            if re.search(r"\b(?:graph|commandGraph)\.add\s*\(", text):
                errors.append(f"{name}: model dispatches a Metal pipeline directly")
        if name.startswith("models/"):
            # A model folder may launch its own kernels, but never reaches the
            # engine or another model's folder.
            own = "/".join(name.split("/")[:2]) + "/"
            for include in includes:
                if any(part in include for part in forbidden_model_dependencies):
                    errors.append(f"{name}: model depends on engine layer {include}")
                if include.startswith("models/") and not include.startswith(own):
                    errors.append(f"{name}: model depends on another model {include}")
        if name.startswith(("runtime/model/", "runtime/engine/", "models/")):
            if OPERATOR_WORKSPACE_POLICY.search(text):
                layer = name.split("/")[1]
                errors.append(f"{name}: {layer} owns an operator workspace policy")
        if name.startswith("runtime/ops/"):
            for include in includes:
                if any(part in include for part in forbidden_operator_dependencies):
                    errors.append(
                        f"{name}: operator depends on upper production layer {include}"
                    )
        if name.startswith("runtime/engine/") and name not in engine_assembly_sources:
            for include in includes:
                if include in concrete_model_headers or include.startswith("models/"):
                    errors.append(
                        f"{name}: engine policy depends on concrete model {include}"
                    )
        if name == "runtime/main.mm":
            for include in includes:
                if include in concrete_model_headers or include.startswith("models/"):
                    errors.append(
                        f"{name}: startup depends on concrete model {include}"
                    )
            if concrete_model_symbols.search(text):
                errors.append(f"{name}: startup names a concrete model type")
        if (
            name.startswith("runtime/engine/")
            and name != "runtime/engine/Checked.hpp"
            and re.search(r"^namespace splash\s*\{", text, re.MULTILINE)
        ):
            errors.append(f"{name}: engine declarations leak into root namespace")
        if client_names.search(text):
            errors.append(
                f"{name}: production backend contains client-specific behavior"
            )
        # Every timeout and duration counts time the Mac is awake, as the
        # server's time.monotonic() does.
        if SLEEP_COUNTING_CLOCK.search(text):
            errors.append(f"{name}: measures time on a clock that counts sleep")
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"architecture error: {error}", file=sys.stderr)
        return 1
    print("backend architecture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
