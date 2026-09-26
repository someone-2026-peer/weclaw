#!/usr/bin/env python3
"""E11 real-execution sandbox package.

Public API (imported by the E11 harness)::

    from sandbox_env import execute, ExecOutcome, SandboxWorkspace, coverage_report, SANDBOX_FAMILIES

The sandbox replaces the oracle ``selected in acceptable_set`` judgement with a
real, deterministic, temp-isolated tool execution. Only ``status == "error"``
drives PTE-FD failure-driven escalation. See ``executor.py`` for the contract
and ``families.py`` for the 15 implemented tool families.
"""
from __future__ import annotations

try:  # package import
    from .executor import (
        ExecOutcome,
        SandboxWorkspace,
        coverage_report,
        execute,
        FAMILIES,
        SANDBOX_FAMILIES,
    )
except ImportError:  # direct-module import fallback
    from executor import (  # type: ignore
        ExecOutcome,
        SandboxWorkspace,
        coverage_report,
        execute,
        FAMILIES,
        SANDBOX_FAMILIES,
    )

__all__ = [
    "execute",
    "ExecOutcome",
    "SandboxWorkspace",
    "coverage_report",
    "FAMILIES",
    "SANDBOX_FAMILIES",
]
