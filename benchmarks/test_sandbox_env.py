#!/usr/bin/env python3
"""Unit tests for the E11 real-execution sandbox (``benchmarks/sandbox_env``).

Run::

    pytest "docs/8 计划发布的论文papers/weclaw_adaptive_runtime/benchmarks/test_sandbox_env.py" -v

What these tests guarantee for the paper's execution-level (non-oracle) claim:
  1. All 49 implemented families (15 base here + 22 Tier-1 in
     ``test_sandbox_tier1.py`` + 12 Tier-2 in ``test_sandbox_tier2.py``) really
     EXECUTE and succeed on in-domain args.
  2. Every out-of-sandbox family returns a deterministic ``capability_unavailable``.
  3. The error taxonomy (not_found / invalid_args / execution_error / no_selection)
     behaves as documented.
  4. NON-ORACLE PROOF: a tool that IS acceptable for its intent still returns
     status=error when its arguments are out of the fixture domain. An oracle
     acceptable-set lookup would call these "correct"; real execution does not.
  5. Determinism: identical (func_name, args) -> identical status (and identical
     payload for payload-stable families), across fresh workspaces.
  6. Isolation/safety: temp-rooted workspace, path-escape blocked, reset restores
     pristine fixture state, close() removes only self-owned temp roots.
  7. Faithfulness: family/action are split by the PRODUCTION ``_extract_tool_name``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from sandbox_env import (  # noqa: E402
    ExecOutcome,
    SandboxWorkspace,
    coverage_report,
    execute,
    SANDBOX_FAMILIES,
)


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def ws():
    """A fresh, pristine temp workspace per test (auto-cleaned)."""
    w = SandboxWorkspace()
    yield w
    w.close()


# In-domain args for each of the 15 families: the RIGHT tool with reasonable
# arguments must really execute and succeed.
IN_DOMAIN = [
    ("file_read", {"file_path": "content.txt"}),
    ("calculator_compute", {"expression": "(12+8)*3"}),
    ("datetime_tool_now", {}),
    ("data_processor_analyze", {"file_path": "data/sales.csv"}),
    ("knowledge_rag_search", {"query": "搜索知识库中的论文"}),
    ("weather_query", {"city": "北京"}),
    ("statistics_compute", {"data": [10, 20, 30, 40]}),
    ("python_runner_execute", {"code": "print(6*7)"}),
    ("poetry_search", {"author": "李白"}),
    ("pdf_tool_merge", {"files": ["a.pdf", "b.pdf"]}),
    ("format_converter_convert", {"file_path": "report.docx", "target_format": "pdf"}),
    ("todo_add", {"text": "write tests"}),
    ("finance_record", {"amount": 100, "kind": "expense"}),
    ("diary_write", {"text": "sunny day"}),
    ("system_monitor_status", {}),
]

# Genuinely-unbuilt Tier-3 families (hardware / dangerous / irreversible).
# NOTE: doc_generator_create / notify_send / ppt_generator_build were REMOVED --
# Tier-1 implements them; browser_use_open / stock_query_price / quant_trading_run /
# fred_query_gdp were REMOVED -- Tier-2 (record-and-freeze real data) implements them.
OUT_OF_SANDBOX = [
    "shell_execute", "voice_input_start", "ocr_recognize", "image_generator_generate",
    "app_control_click", "screen_capture", "speech_to_text_transcribe", "email_send",
]


# ------------------------------------------------------------------ registry

def test_exactly_49_families_registered():
    # 15 base + 22 Tier-1 (families_tier1.py) + 12 Tier-2 (families_tier2.py)
    assert len(SANDBOX_FAMILIES) == 49
    assert "file" in SANDBOX_FAMILIES and "system_monitor" in SANDBOX_FAMILIES
    # Tier-1 families are now in-sandbox:
    assert {"cron", "notify", "ppt_generator", "doc_generator",
            "fitness_nutrition", "music_player"} <= SANDBOX_FAMILIES
    # Tier-2 record-and-freeze real-data families are now in-sandbox:
    assert {"browser", "browser_use", "stock_query", "fred_query", "quant_trading",
            "search", "duckduckgo_search", "crawlee_tool", "literature_search",
            "batch_paper_analyzer", "stock_photo", "literature_review"} <= SANDBOX_FAMILIES
    # genuinely out-of-sandbox (Tier-3 hardware/dangerous) families stay unimplemented:
    assert "shell" not in SANDBOX_FAMILIES and "email" not in SANDBOX_FAMILIES
    assert "screen" not in SANDBOX_FAMILIES and "ocr" not in SANDBOX_FAMILIES


# ------------------------------------------------------------------ success

@pytest.mark.parametrize("func,args", IN_DOMAIN)
def test_in_domain_execution_succeeds(ws, func, args):
    out = execute(func, args, ws)
    assert isinstance(out, ExecOutcome)
    assert out.status == "success", f"{func} -> {out.error_kind}: {out.error_msg}"
    assert out.error_kind is None
    assert out.ok is True
    assert out.latency_ms >= 0.0


@pytest.mark.parametrize("func,exp_tool,exp_action", [
    ("knowledge_rag_search", "knowledge_rag", "search"),
    ("data_processor_analyze", "data_processor", "analyze"),
    ("datetime_tool_now", "datetime_tool", "now"),
    ("system_monitor_status", "system_monitor", "status"),
    ("python_runner_execute", "python_runner", "execute"),
    ("format_converter_convert", "format_converter", "convert"),
    ("pdf_tool_merge", "pdf_tool", "merge"),
    ("browser_use_open", "browser_use", "open"),
])
def test_production_tool_name_parsing(ws, func, exp_tool, exp_action):
    """Family/action come from the production _extract_tool_name (faithful)."""
    out = execute(func, {}, ws)
    assert out.tool == exp_tool
    assert out.action == exp_action


# ------------------------------------------------------------------ capability

@pytest.mark.parametrize("func", OUT_OF_SANDBOX)
def test_out_of_sandbox_is_capability_error(ws, func):
    out = execute(func, {"anything": "value"}, ws)
    assert out.status == "error"
    assert out.error_kind == "capability_unavailable"
    assert out.tool not in SANDBOX_FAMILIES


# ------------------------------------------------------------------ error kinds

@pytest.mark.parametrize("func,args,kind", [
    ("weather_query", {"city": "atlantis"}, "not_found"),
    ("file_read", {"file_path": "missing.xyz"}, "not_found"),
    ("knowledge_rag_search", {"query": "番茄炒蛋的做法"}, "not_found"),
    ("finance_record", {"amount": "abc"}, "invalid_args"),
    ("calculator_compute", {"expression": "hello world"}, "execution_error"),
    ("python_runner_execute", {"code": "import os"}, "execution_error"),
    ("", {}, "no_selection"),
    ("__ERROR__", {}, "no_selection"),
    ("__need_more_tools__", {}, "no_selection"),
])
def test_error_kinds(ws, func, args, kind):
    out = execute(func, args, ws)
    assert out.status == "error"
    assert out.error_kind == kind, f"{func} expected {kind}, got {out.error_kind}: {out.error_msg}"


# ------------------------------------------------------------------ NON-ORACLE

@pytest.mark.parametrize("func,args", [
    ("weather_query", {"city": "atlantis"}),          # acceptable for daily_assistant
    ("knowledge_rag_search", {"query": "番茄炒蛋的做法"}),   # acceptable for knowledge
    ("poetry_search", {"query": "量子力学公式"}),          # acceptable for knowledge
    ("file_read", {"file_path": "does_not_exist.bin"}),  # acceptable for file_operation
])
def test_acceptable_tool_out_of_domain_really_errors(ws, func, args):
    """The crux: this is REAL EXECUTION, not an oracle acceptable-set lookup.

    Each tool below IS an acceptable answer for its intent, so an oracle scorer
    would mark the selection correct regardless of args. The sandbox instead
    returns status=error because the argument is outside the fixture domain --
    and that real error is what legitimately triggers PTE-FD escalation.
    """
    out = execute(func, args, ws)
    assert out.status == "error"
    assert out.error_kind in {"not_found", "invalid_args", "execution_error"}


# ------------------------------------------------------------------ determinism

@pytest.mark.parametrize("func,args", IN_DOMAIN + [("weather_query", {"city": "atlantis"})])
def test_status_deterministic_across_workspaces(func, args):
    statuses = set()
    for _ in range(3):
        w = SandboxWorkspace()
        try:
            statuses.add(execute(func, args, w).status)
        finally:
            w.close()
    assert len(statuses) == 1, f"{func} produced non-deterministic statuses: {statuses}"


@pytest.mark.parametrize("func,args", [
    ("calculator_compute", {"expression": "(12+8)*3"}),
    ("weather_query", {"city": "北京"}),
    ("knowledge_rag_search", {"query": "搜索知识库中的论文"}),
    ("poetry_search", {"author": "李白"}),
    ("statistics_compute", {"data": [10, 20, 30, 40]}),
])
def test_payload_deterministic_for_stable_families(func, args):
    """Payload-stable families return byte-identical results across fresh runs.

    (datetime_tool / system_monitor intentionally embed live host facts, so only
    their STATUS is asserted deterministic above -- not their payload.)
    """
    results = []
    for _ in range(2):
        w = SandboxWorkspace()
        try:
            results.append(execute(func, args, w).result)
        finally:
            w.close()
    assert results[0] == results[1]


# ------------------------------------------------------------------ safety

def test_resolve_fs_blocks_escape(ws):
    assert ws.resolve_fs("../../etc/passwd") is None
    assert ws.resolve_fs("/etc/passwd") is None
    assert ws.resolve_fs("C:\\Windows\\system32\\x.txt") is None
    good = ws.resolve_fs("content.txt")
    assert good is not None and good.is_relative_to(ws.fs_root)


def test_file_write_escape_rejected(ws):
    out = execute("file_write", {"file_path": "../../evil.txt", "content": "x"}, ws)
    assert out.status == "error"
    assert out.error_kind == "invalid_args"


def test_file_write_read_roundtrip(ws):
    w = execute("file_write", {"file_path": "new_note.txt", "content": "hello sandbox"}, ws)
    assert w.status == "success"
    r = execute("file_read", {"file_path": "new_note.txt"}, ws)
    assert r.status == "success" and r.result["content"] == "hello sandbox"


def test_reset_restores_fs_and_state(ws):
    execute("file_write", {"file_path": "ephemeral.txt", "content": "x"}, ws)
    execute("todo_add", {"text": "ephemeral task"}, ws)
    before = execute("todo_list", {}, ws).result["count"]
    ws.reset()
    # written file is gone
    assert execute("file_read", {"file_path": "ephemeral.txt"}, ws).error_kind == "not_found"
    # todo state back to the 3 seeded items
    after = execute("todo_list", {}, ws).result
    assert after["count"] == before - 1
    assert not any(i["text"] == "ephemeral task" for i in after["items"])


def test_temp_isolation_and_close():
    w = SandboxWorkspace()
    root = w.root
    assert root.exists() and root.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert (w.fs_root / "content.txt").exists()
    w.close()
    assert not root.exists()


def test_custom_root_is_not_owned(tmp_path):
    target = tmp_path / "sbx"
    w = SandboxWorkspace(root=target)
    assert w.root.exists() and w.owns_root is False
    w.close()
    assert w.root.exists()  # caller-owned root must survive close()


# ------------------------------------------------------------------ coverage

def test_coverage_report():
    # 5 genuinely-unbuilt Tier-3 families appended to the 49 sandbox families
    extra_out = ["shell", "ocr", "email", "screen", "image_generator"]
    assert not (set(extra_out) & SANDBOX_FAMILIES), "extra families must be out-of-sandbox"
    universe = list(SANDBOX_FAMILIES) + extra_out
    cov = coverage_report(universe)
    assert cov["n_sandbox"] == 49
    assert cov["universe_size"] == 54                       # 49 + 5, no overlap
    assert cov["n_covered"] == 49                           # all sandbox families in universe
    assert cov["coverage_pct"] == round(100 * 49 / 54, 1)   # 90.7
    assert set(cov["sandbox_families"]) == set(SANDBOX_FAMILIES)


def test_outcome_to_dict_roundtrip(ws):
    out = execute("calculator_compute", {"expression": "1+1"}, ws)
    d = out.to_dict()
    assert d["status"] == "success" and d["tool"] == "calculator"
    assert set(d) >= {"status", "result", "error_kind", "error_msg",
                      "tool", "action", "func_name", "latency_ms"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
