#!/usr/bin/env python3
"""E11 real-execution sandbox executor.

This module turns a model's tool selection into a REAL execution outcome
instead of an oracle ``selected in acceptable_set`` lookup. It is the crux of
the execution-level (non-oracle) benchmark requested by reviewer R1-1.

Contract::

    execute(func_name, args, ws) -> ExecOutcome
        ExecOutcome.status in {"success", "error"}
        ONLY status == "error" drives PTE-FD failure-driven escalation.

Design invariants:
  * Faithful tool-name parsing: family/action are split with the PRODUCTION
    ``src.core.tool_exposure._extract_tool_name`` so the sandbox sees exactly
    the same tool identity the runtime engine does (no re-implementation).
  * Deterministic STATUS: the success/error classification is fully
    reproducible for a given (func_name, args). A couple of result payloads
    carry real host facts (datetime.now, os.cpu_count) but their STATUS is
    constant, so the science (which tool call succeeds / fails) is repeatable.
  * Isolated + local: every run materializes fixtures into a private temp
    workspace; no network, no writes outside the workspace fs/state roots.
  * Capability error (not business failure): any tool family outside the 15
    implemented ones returns error_kind="capability_unavailable"; this is
    disclosed in the paper rather than counted as a tool defect.

error_kind taxonomy:
    capability_unavailable - family not implemented in the sandbox (~60/75)
    no_selection           - empty / sentinel selection ("", "__ERROR__", ...)
    invalid_args           - right family, unusable arguments
    not_found              - right family, argument outside the fixture domain
    execution_error        - family ran but the operation itself failed
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ paths
# sandbox_env -> benchmarks -> weclaw_adaptive_runtime -> "8 ..." -> docs -> weclaw
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Faithful, production tool-name parsing (never re-implemented here).
from src.core.tool_exposure import _extract_tool_name  # noqa: E402

# Families registry (handlers return plain tuples; this module wraps them).
try:  # package import (harness adds benchmarks/ to sys.path)
    from .families import FAMILIES, SANDBOX_FAMILIES
except ImportError:  # direct-module import fallback
    from families import FAMILIES, SANDBOX_FAMILIES  # type: ignore

_DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sandbox_fixtures.json"

# Selection sentinels that mean "the model produced no usable tool call".
_SENTINEL_PREFIXES = ("__",)


# ------------------------------------------------------------------ outcome

@dataclass
class ExecOutcome:
    """Result of a single real tool execution inside the sandbox."""

    status: str                      # "success" | "error"
    result: Any = None               # handler payload on success
    error_kind: str | None = None    # taxonomy above; None on success
    error_msg: str = ""              # human-readable diagnostic
    tool: str = ""                   # parsed family (production parser)
    action: str = ""                 # parsed action suffix
    func_name: str = ""              # raw function name as selected
    latency_ms: float = 0.0          # wall time of the dispatch+handler

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ workspace

class SandboxWorkspace:
    """A private, fixture-materialized filesystem/state root for one run.

    ``fs_root`` backs file-like families (file/data_processor/pdf/format...).
    ``state_root`` backs stateful families (todo/finance/diary). Both are
    re-created from the committed fixtures on :meth:`reset`, so each query can
    start from an identical, deterministic state.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        fixtures_path: str | Path | None = None,
        persist: bool = False,
    ) -> None:
        self.fixtures_path = Path(fixtures_path) if fixtures_path else _DEFAULT_FIXTURES
        self.fixtures: dict = json.loads(self.fixtures_path.read_text(encoding="utf-8"))

        self.owns_root = root is None and not persist
        if root is None:
            self.root = Path(tempfile.mkdtemp(prefix="e11_sandbox_"))
        else:
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        self.fs_root = (self.root / "fs").resolve()
        self.state_root = (self.root / "state").resolve()
        self.materialize()

    # -- lifecycle -------------------------------------------------
    def materialize(self) -> None:
        """(Re)create fs_root and state_root from the committed fixtures."""
        for d in (self.fs_root, self.state_root):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        for rel, content in self.fixtures.get("seed_fs", {}).items():
            p = self.resolve_fs(rel)
            if p is None:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(content), encoding="utf-8")

        state = self.fixtures.get("seed_state", {})
        for key, val in state.items():
            if key == "diary":                  # diary is a directory of .md entries
                (self.state_root / "diary").mkdir(parents=True, exist_ok=True)
            else:                               # every other store seeds as <key>.json
                (self.state_root / f"{key}.json").write_text(
                    json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> None:
        """Restore the pristine fixture state (call between queries)."""
        self.materialize()

    def close(self) -> None:
        if self.owns_root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "SandboxWorkspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- path safety -----------------------------------------------
    def resolve_fs(self, rel: Any) -> Path | None:
        """Resolve ``rel`` under fs_root; return None if it would escape.

        Blocks absolute paths and ``..`` traversal so no handler can touch the
        real filesystem outside the sandbox.
        """
        if rel is None or str(rel) == "":
            return None
        candidate = (self.fs_root / str(rel)).resolve()
        try:
            candidate.relative_to(self.fs_root)
        except ValueError:
            return None
        return candidate


# ------------------------------------------------------------------ dispatch

def _resolve_family(func_name: str) -> str:
    """Resolve the tool family for a selected ``func_name``.

    Production ``_extract_tool_name`` stays authoritative for every family it
    parses to something the sandbox implements. Only when production parsing
    yields a family that is NOT implemented (a capability error) do we try a
    longest-prefix match against the sandbox's own family names. This recovers
    universe-only families whose multi-underscore names production KNOWN_PREFIXES
    does not list (e.g. fitness_nutrition / concept_diagrams / meme_generation),
    resolving them to the same family string the benchmark's acceptable_tools
    universe uses -- without re-implementing or overriding production parsing for
    any real tool.
    """
    fn = str(func_name)
    prod = _extract_tool_name(fn)
    if prod in FAMILIES:
        return prod
    best = ""
    for fam in SANDBOX_FAMILIES:
        if (fn == fam or fn.startswith(fam + "_")) and len(fam) > len(best):
            best = fam
    return best or prod


def execute(func_name: str, args: dict | None, ws: SandboxWorkspace) -> ExecOutcome:
    """Execute a selected tool for real and return an :class:`ExecOutcome`.

    ``func_name`` is the raw schema function name (``family_action``); it is
    parsed with the production ``_extract_tool_name`` for faithfulness.
    """
    t0 = time.perf_counter()
    args = args or {}

    if not func_name or str(func_name).startswith(_SENTINEL_PREFIXES):
        out = ExecOutcome(
            status="error", error_kind="no_selection",
            error_msg=f"no usable tool selection: {func_name!r}",
            tool="", action="", func_name=str(func_name or ""))
        out.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return out

    family = _resolve_family(str(func_name))
    action = str(func_name)[len(family) + 1:] if str(func_name).startswith(family + "_") else ""

    handler = FAMILIES.get(family)
    if handler is None:
        out = ExecOutcome(
            status="error", error_kind="capability_unavailable",
            error_msg=(f"tool family '{family}' is not implemented in the E11 sandbox; "
                       f"counted as a capability error, not a business failure"),
            tool=family, action=action, func_name=str(func_name))
        out.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return out

    try:
        status, result, error_kind, error_msg = handler(action, args, ws)
        out = ExecOutcome(
            status=status, result=result, error_kind=error_kind, error_msg=error_msg or "",
            tool=family, action=action, func_name=str(func_name))
    except Exception as exc:  # a handler bug must never crash the harness
        out = ExecOutcome(
            status="error", error_kind="execution_error",
            error_msg=f"sandbox handler raised {type(exc).__name__}: {exc}",
            tool=family, action=action, func_name=str(func_name))

    out.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return out


def coverage_report(universe) -> dict:
    """Report sandbox coverage over a tool universe (paper: 15/N disclosure)."""
    universe = set(universe)
    covered = sorted(universe & SANDBOX_FAMILIES)
    return {
        "sandbox_families": sorted(SANDBOX_FAMILIES),
        "n_sandbox": len(SANDBOX_FAMILIES),
        "universe_size": len(universe),
        "covered": covered,
        "n_covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / len(universe), 1) if universe else 0.0,
    }


__all__ = [
    "ExecOutcome", "SandboxWorkspace", "execute", "coverage_report",
    "SANDBOX_FAMILIES", "FAMILIES",
]
