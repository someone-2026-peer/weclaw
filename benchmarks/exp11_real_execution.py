#!/usr/bin/env python3
"""E11 real-execution harness -- fork of ``exp_pte_ablation_canonical.py``.

Upgrades the paper's main result from ORACLE selection-level (``selected in
acceptable_set``) to REAL EXECUTION (``sandbox.execute(sel, args).status ==
'success'``), answering reviewer R1-1. The runtime mechanism is NOT rewritten:
this fork reuses the production ``ToolExposureEngine``, ``detect_intent_with_confidence``
and exp1's ``MockToolRegistry / build_schemas / itr_select_tools`` verbatim.

What changed vs canonical (the only edits, all in service of real execution):
  * ``call_llm`` -> backend returns the RAW function name AND its parsed
    ``arguments`` (real execution needs args), not just the tool family.
  * Judgement ``ok = sel in gt`` -> REAL EXECUTION ``sandbox.execute(sel, args)``:
    a selection counts as success only if it (a) really ran without error AND
    (b) is a task-valid tool for the query (in the frozen ``acceptable_tools``).
    Non-sandbox families can never run (``capability_unavailable``), so (a)+(b)
    == "picked a task-valid sandbox tool that really executed". This is strictly
    HARDER than the oracle (an acceptable tool that fails at runtime -- bad args,
    missing file -- is still a failure), so it cannot inflate results, and it
    rejects vacuous-but-executable calls (e.g. ``file_write`` for a web-browsing
    request) that would otherwise manufacture FALSE GAINS on the out-of-sandbox
    negative control (V2 L293). ``exec_status`` is recorded separately so the
    pure-execution outcome stays fully visible/auditable.
  * PTE-FD escalation is triggered by a REAL ``status == "error"`` (not an oracle
    mismatch): ``report_failure() x2`` -> broader tier -> re-select -> RE-EXECUTE.
  * 6 configs (added fair retry arms) with DOUBLE reporting (single attempt +
    after the k=2 recovery budget):
        static         all schemas, one shot
        static_retry   all schemas, + error-text re-prompt (same schemas)
        pte            tiered schemas, one shot
        pte_fd         tiered + failure-driven TIER ESCALATION (no error text)
        keyword        ITR top-10 schemas, one shot
        keyword_retry  ITR top-10, + error-text re-prompt (same schemas)
    Bias control (V2 L290): at temp=0 a re-sample is identity, so the retry arms
    change the PROMPT (inject the real error text) while PTE-FD changes the
    EXPOSED TOOL TIER -- two distinct recovery mechanisms at an equal k=2 budget.

Offline mock backend: ``--backend mock`` replaces the LLM with a deterministic,
seeded stand-in so the whole chain (select -> execute -> escalate/retry ->
record -> checkpoint -> resume -> aggregate) can be validated WITHOUT spending
API calls. The mock only simulates SELECTION; success/error is still decided by
REAL sandbox execution. Mock numbers are for pipeline validation, NOT science.

Usage::
    python exp11_real_execution.py --backend mock --arm pilot            # offline smoke
    python exp11_real_execution.py --backend mock --arm full --resume    # offline dry-run
    python exp11_real_execution.py --backend deepseek --arm pilot        # live gate (Phase 4a)
    python exp11_real_execution.py --backend deepseek --arm full --seeds 42,43,44
    python exp11_real_execution.py --backend qwen --arm subset --seeds 42,43,44
    python exp11_real_execution.py --aggregate --arm full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BENCH_DIR = Path(__file__).parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI  # noqa: E402

from src.core.prompts import detect_intent_with_confidence  # noqa: E402
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name  # noqa: E402
import exp1_pte_ablation as base  # noqa: E402  (read-only reuse)
from sandbox_env import (  # noqa: E402
    ExecOutcome, SandboxWorkspace, coverage_report, execute, SANDBOX_FAMILIES,
)

# ------------------------------------------------------------------ constants

CONFIGS = ("static", "static_retry", "pte", "pte_fd", "keyword", "keyword_retry")
CONFIG_MECHANISM = {
    "static": "none", "pte": "none", "keyword": "none",
    "static_retry": "retry", "keyword_retry": "retry", "pte_fd": "escalate",
}

PROVIDERS = {  # forked verbatim from canonical (paper's four backend identities)
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY",
                 "model": "deepseek-chat", "max_tokens": 256, "paper_name": "deepseek-v4-flash"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_env": "QWEN_API_KEY",
             "model": "qwen-max", "max_tokens": 256, "paper_name": "qwen-max"},
    # kimi: the paper's original moonshot-v1-8k was DEPRECATED (404 "Not found the
    # model") mid-revision; /models now lists only kimi-k2.6/k2.7-code/k3, and ALL of
    # them FORCE temperature=1 (non-deterministic). kimi-k2.7-code is the only current
    # kimi that reliably emits structured tool_calls on the 78-tool schema (5/6 probe;
    # k2.6/k3/highspeed write prose at 0-17%). The substitution and the forced temp=1
    # (vs temp=0 for deepseek/qwen/glm) are disclosed as a limitation; 3-seed averaging
    # mitigates the sampling noise.
    "kimi": {"base_url": "https://api.moonshot.cn/v1", "key_env": "KIMI_API_KEY",
             "model": "kimi-k2.7-code", "max_tokens": 512, "paper_name": "kimi",
             "temperature": 1.0},
    # PAID glm-4-flashx: the paid sibling of the FREE glm-4-flash baseline (author
    # requires a non-free model). Non-reasoning gen-4 flash tier -> emits clean
    # structured tool_calls like deepseek-chat / qwen-max / moonshot (fair tool-
    # selection PK). Validated on the real 78-tool schema: 100% tool-call rate both
    # sequential and under 6-way concurrency (18/18), deterministic at temperature=0,
    # ~1.1s/call, no 429. Reasoning GLMs were REJECTED -- on a 6-query probe they
    # write prose / clarify / refuse instead of calling a tool (tool-call rate:
    # glm-5.1=33%, glm-4.7-flashx=0-67%, glm-4-plus/air=33%).
    "glm": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "key_env": "GLM_API_KEY",
            "model": "glm-4-flashx", "max_tokens": 256, "paper_name": "glm"},
    "mock": {"base_url": "", "key_env": "", "model": "mock-offline",
             "max_tokens": 256, "paper_name": "mock(offline-validation-only)"},
}

SYSTEM_PROMPT = (  # forked verbatim from canonical
    "你是 WeClaw AI 助手的工具选择模块。根据用户请求，选择最合适的工具来完成任务。"
    "你必须从提供的工具列表中选择一个工具。只选择工具，不要执行。"
)

QUERY_SET_PATH = BENCH_DIR / "raw_data" / "e11_query_set.json"
OUT_DIR = BENCH_DIR / "raw_data" / "runs" / "exp11"


# ------------------------------------------------------------------ arguments

def parse_arguments(raw) -> dict:
    """Parse a tool_call's ``arguments`` (JSON string from the API) into a dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"_raw": str(raw)}


def _mock_args(family: str, query: str) -> dict:
    """In-domain arguments per family for the offline mock backend."""
    table = {
        "file": {"file_path": "content.txt"},
        "calculator": {"expression": "12*8+3"},
        "datetime_tool": {},
        "data_processor": {"file_path": "data/sales.csv"},
        "knowledge_rag": {"query": query},
        "weather": {"city": "北京"},
        "statistics": {"data": [10, 20, 30, 40]},
        "python_runner": {"code": "print(6*7)"},
        "poetry": {"author": "李白"},
        "pdf_tool": {"operation": "inspect", "file_path": "report.md"},
        "format_converter": {"file_path": "report.docx", "target_format": "pdf"},
        "todo": {"text": query[:40]},
        "finance": {"amount": 12.5, "kind": "expense"},
        "diary": {"text": query[:40]},
        "system_monitor": {},
    }
    return dict(table.get(family, {"input": query}))


# ------------------------------------------------------------------ backends

class MockBackend:
    """Deterministic offline stand-in for the LLM (pipeline validation only).

    Simulates an imperfect model: for in-sandbox queries it picks a truly
    executable tool with probability that IMPROVES on the recovery attempt
    (mimicking that broader exposure / error feedback helps); otherwise it picks
    a plausible distractor. Out-of-sandbox queries never have an executable
    success tool exposed, so they stay capability errors -- exercising the
    negative control. Selection is simulated; success is decided by REAL sandbox
    execution downstream.
    """

    name = "mock"

    def __init__(self, seed: int):
        self.seed = seed

    def select(self, query, schemas, attempt=0, error_feedback=None, qmeta=None):
        if not schemas:
            return ("", {}, 0)
        h = int(hashlib.sha256(f"{self.seed}|{query}|{attempt}".encode("utf-8")).hexdigest(), 16)
        r = (h % 10000) / 10000.0
        fns = [s["function"]["name"] for s in schemas]
        fam_to_fn: dict[str, str] = {}
        for fn in fns:
            fam_to_fn.setdefault(_extract_tool_name(fn), fn)
        qmeta = qmeta or {}
        exposed_success = [fam_to_fn[t] for t in qmeta.get("sandbox_success_tools", [])
                           if t in fam_to_fn]
        p_correct = 0.78 if attempt > 0 else 0.56
        if qmeta.get("in_sandbox") and exposed_success and r < p_correct:
            fn = exposed_success[h % len(exposed_success)]
        else:
            fn = fns[h % len(fns)]
        return (fn, _mock_args(_extract_tool_name(fn), query), 60)


class LiveBackend:
    """Real OpenAI-compatible backend; returns raw func name + parsed args."""

    def __init__(self, client: OpenAI, model_id: str, max_tokens: int, extra_body=None,
                 temperature: float = 0.0):
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.temperature = temperature   # kimi-k2.7-code forces 1.0; others stay 0.0
        self.name = model_id

    def select(self, query, schemas, attempt=0, error_feedback=None, qmeta=None):
        if not schemas:
            return ("", {}, 0)
        user_content = query
        if error_feedback:
            user_content = f"{query}\n\n[系统反馈] {error_feedback}"
        try:
            kwargs = dict(
                model=self.model_id,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_content}],
                tools=schemas, tool_choice="auto",
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            resp = self.client.chat.completions.create(**kwargs)
            tokens = resp.usage.prompt_tokens if resp.usage else 0
            msg = resp.choices[0].message
            if msg.tool_calls:
                call = msg.tool_calls[0]
                return (call.function.name, parse_arguments(call.function.arguments), tokens)
            return ("", {}, tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"  LLM call error: {exc}")
            return ("__ERROR__", {}, 0)


def make_backend(backend_key: str, seed: int, model_id=None, max_tokens=None):
    prov = PROVIDERS[backend_key]
    if backend_key == "mock":
        return MockBackend(seed)
    api_key = os.getenv(prov["key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing API key env {prov['key_env']} for backend {backend_key}")
    client = OpenAI(api_key=api_key, base_url=prov["base_url"])
    return LiveBackend(client, model_id or prov["model"], max_tokens or prov["max_tokens"],
                       extra_body=prov.get("extra_body"),
                       temperature=prov.get("temperature", 0.0))


# ------------------------------------------------------------------ queries

def load_queries(arm: str, pilot_n: int = 20) -> list[dict]:
    payload = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    qs = payload["queries"]
    if arm == "subset":
        return [q for q in qs if q["in_subset"]]
    full = [q for q in qs if q["in_full"]]
    if arm == "pilot":
        step = max(1, len(full) // pilot_n)
        return full[::step][:pilot_n]
    return full


# ------------------------------------------------------------------ execution

def _feedback(fn: str, outcome: ExecOutcome) -> str:
    """Tool-agnostic generic retry nudge (V2 L290): re-inject the REAL error text."""
    return (f"上一次工具调用 `{fn}` 未能完成请求，返回 [{outcome.error_kind}] "
            f"{outcome.error_msg}。请重新考虑，选择一个能够真正完成该请求的工具或参数。")


def _attempt(backend, query, schemas, ws, attempt, feedback, qmeta):
    """One selection + one REAL execution. Returns (outcome, func_name, args, tokens)."""
    fn, args, tokens = backend.select(query, schemas, attempt=attempt,
                                      error_feedback=feedback, qmeta=qmeta)
    if fn == "__ERROR__":
        o = ExecOutcome(status="error", error_kind="api_error",
                        error_msg="backend API failure", func_name="__ERROR__")
        return (o, fn, args, tokens)
    return (execute(fn, args, ws), fn, args, tokens)


def _schemas_for_config(cfg, engine, intent_result, query, static_schemas):
    if cfg in ("static", "static_retry"):
        return static_schemas
    if cfg in ("pte", "pte_fd"):
        return engine.get_schemas(intent_result)
    return base.itr_select_tools(query, static_schemas, top_k=10)


def _bytes(schemas) -> int:
    return len(json.dumps(schemas, ensure_ascii=False).encode("utf-8"))


def _judge(outcome: ExecOutcome, qmeta: dict) -> int:
    """Real-execution success, immune to vacuous-but-executable tool calls.

    Success requires BOTH (a) the tool really ran without error in the sandbox
    AND (b) it is a task-valid tool for THIS query (in the frozen
    ``acceptable_tools``). Because non-sandbox families always return
    ``capability_unavailable``, (a)+(b) is equivalent to "selected a task-valid
    sandbox tool that really executed". The conjunction is strictly harder than
    the oracle ``sel in gt`` (an acceptable tool that fails at runtime is still
    a failure), so it cannot inflate results; and it rejects structurally-valid
    but semantically-irrelevant calls (e.g. ``file_write`` for a web-browsing
    request) that would otherwise create false gains on out-of-sandbox queries.
    """
    acceptable = qmeta.get("acceptable_tools") or []
    return int(outcome.ok and outcome.tool in acceptable)


def run_config(cfg, backend, query, qmeta, ws, engine, intent_result, static_schemas) -> dict:
    """Run one config for one query with REAL execution; return a per-step record."""
    ws.reset()          # every config faces an identical pristine sandbox world
    engine.reset()
    schemas0 = _schemas_for_config(cfg, engine, intent_result, query, static_schemas)
    b0 = _bytes(schemas0)
    mechanism = CONFIG_MECHANISM[cfg]

    o1, fn1, a1, tk1 = _attempt(backend, query, schemas0, ws, 0, None, qmeta)
    rec = {
        "selected": fn1, "tool": o1.tool, "action": o1.action, "args": a1,
        "exec_status": o1.status, "error_kind": o1.error_kind, "latency_ms": o1.latency_ms,
        "ok_single": _judge(o1, qmeta), "escalated": False, "retried": False,
        "selected_final": fn1, "tool_final": o1.tool, "args_final": a1,
        "exec_status_final": o1.status, "error_kind_final": o1.error_kind,
        "ok_final": _judge(o1, qmeta), "tokens": tk1, "schema_bytes": b0,
    }
    if o1.ok or mechanism == "none":
        return rec

    # recovery budget (k=2): one more selection + REAL re-execution
    if mechanism == "retry":
        rec["retried"] = True
        o2, fn2, a2, tk2 = _attempt(backend, query, schemas0, ws, 1, _feedback(fn1, o1), qmeta)
    else:  # escalate: broaden the exposed tier, NO error text (isolates the mechanism)
        rec["escalated"] = True
        engine.report_failure()
        engine.report_failure()
        esc = engine.get_schemas(intent_result)
        rec["schema_bytes"] = max(b0, _bytes(esc))
        o2, fn2, a2, tk2 = _attempt(backend, query, esc, ws, 1, None, qmeta)

    rec["tokens"] += tk2
    rec.update({
        "selected_final": fn2, "tool_final": o2.tool, "args_final": a2,
        "exec_status_final": o2.status, "error_kind_final": o2.error_kind,
        "ok_final": _judge(o2, qmeta), "latency_ms_final": o2.latency_ms,
    })
    return rec


# ------------------------------------------------------------------ run + IO

def _meta(backend_key, backend, seed, arm, configs, n_queries, run_id, coverage):
    prov = PROVIDERS[backend_key]
    return {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": "E11_real_execution", "backend": backend_key,
        "model_id": getattr(backend, "name", prov["model"]), "paper_name": prov["paper_name"],
        "model_extra_body": getattr(backend, "extra_body", None),
        "model_temperature": getattr(backend, "temperature", 0.0),
        "seed": seed, "arm": arm, "n_queries": n_queries, "configs": list(configs),
        "judgement": ("real execution: sandbox.execute(sel,args).status=='success' "
                      "AND tool in acceptable_tools (non-oracle; strictly harder than sel-in-gt)"),
        "sandbox_coverage": coverage,
    }


def _write(out_path: Path, meta: dict, records: list, summary: dict | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": meta, "records": records}
    if summary is not None:
        payload["summary"] = summary
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------- concurrency (per-thread isolation)
# Each worker thread owns an isolated (SandboxWorkspace, ToolExposureEngine) pair,
# reused across the queries that thread runs and reset per config -- exactly
# mirroring the original single-instance sequential semantics (ws.reset() re-
# materializes fixtures; engine.reset() clears the tier), so parallel results are
# identical to sequential ones. Safety: MockBackend.select is hash-deterministic in
# (seed, query, attempt); LiveBackend uses temperature=0 with independent per-call
# requests; execute() touches only read-only module globals + the passed ws. The
# PTE-FD escalation state machine stays correct because all configs of ONE query run
# sequentially inside a single _process_query call on one thread.
_TLS = threading.local()
_TLS_LOCK = threading.Lock()
_TLS_WORKSPACES: list = []   # every ws created, for deterministic cleanup in finally


def _thread_state():
    st = getattr(_TLS, "st", None)
    if st is None:
        registry = base.MockToolRegistry()
        engine = ToolExposureEngine(registry, enabled=True, enable_annotation=True,
                                    failures_to_upgrade=2)
        ws = SandboxWorkspace()
        st = (ws, engine)
        _TLS.st = st
        with _TLS_LOCK:
            _TLS_WORKSPACES.append(ws)
    return st


def _process_query(q, backend, configs, static_schemas, sleep, backend_key):
    """Run ALL configs for ONE query (sequential within the query) on this thread's
    isolated ws/engine; returns the per-query record."""
    ws, engine = _thread_state()
    intent_result = detect_intent_with_confidence(q["query"])
    rec = {
        "qid": q["qid"], "query": q["query"], "intent": q["intent"],
        "in_sandbox": q["in_sandbox"], "split": q["split"], "source": q["source"],
        "acceptable_tools": q["acceptable_tools"],
        "sandbox_success_tools": q["sandbox_success_tools"],
        "configs": {},
    }
    for cfg in configs:
        rec["configs"][cfg] = run_config(cfg, backend, q["query"], q, ws,
                                         engine, intent_result, static_schemas)
    if sleep and backend_key != "mock":
        time.sleep(sleep)
    return rec


def run_single(backend_key, seed, arm, configs, *, checkpoint_every=20, sleep=0.05,
               resume=False, model_id=None, max_tokens=None, pilot_n=20, workers=1) -> dict:
    queries = load_queries(arm, pilot_n)
    backend = make_backend(backend_key, seed, model_id, max_tokens)
    static_schemas = base.build_schemas(base.get_all_tool_names())
    static_bytes = _bytes(static_schemas)
    universe = base.get_all_tool_names()
    coverage = coverage_report(universe)

    out_path = OUT_DIR / f"exp11_{backend_key}_{arm}_seed{seed}.json"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    records: list[dict] = []          # already-completed records (resume), in query order
    completed: set[str] = set()
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        records = prev.get("records", [])
        completed = {r["qid"] for r in records}
        run_id = prev.get("_meta", {}).get("run_id", run_id)
        print(f"[resume] {len(completed)} queries already done in {out_path.name}")

    pending = [q for q in queries if q["qid"] not in completed]
    meta = _meta(backend_key, backend, seed, arm, configs, len(queries), run_id, coverage)
    print(f"[E11 {backend_key}/{meta['model_id']} arm={arm} seed={seed}] "
          f"n={len(queries)} configs={len(configs)} coverage={coverage['coverage_pct']}% "
          f"static_schemas={len(static_schemas)} ({static_bytes}B) workers={workers}")

    t_start = time.time()
    results_by_qid: dict = {}

    def _ordered() -> list:
        # always re-assemble in original query order -> parallel == sequential output
        return records + [results_by_qid[q["qid"]] for q in pending if q["qid"] in results_by_qid]

    def _flush(n_done: int) -> None:
        if n_done % checkpoint_every == 0 and n_done > 0:
            snap = _ordered()
            _write(out_path, meta, snap, None)
            print(f"  checkpoint: {len(snap)}/{len(queries)} "
                  f"({time.time() - t_start:.0f}s)")

    try:
        if workers <= 1:
            for i, q in enumerate(pending, 1):
                results_by_qid[q["qid"]] = _process_query(
                    q, backend, configs, static_schemas, sleep, backend_key)
                _flush(i)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_q = {ex.submit(_process_query, q, backend, configs,
                                      static_schemas, sleep, backend_key): q
                            for q in pending}
                for i, fut in enumerate(as_completed(fut_to_q), 1):
                    q = fut_to_q[fut]
                    results_by_qid[q["qid"]] = fut.result()
                    _flush(i)
        final_records = _ordered()
    finally:
        with _TLS_LOCK:
            for w in _TLS_WORKSPACES:
                try:
                    w.close()
                except Exception:
                    pass
            _TLS_WORKSPACES.clear()

    records = final_records
    summary = summarize(records, configs, static_bytes, coverage)
    summary["wall_seconds"] = round(time.time() - t_start, 1)
    meta["timestamp"] = datetime.now().isoformat(timespec="seconds")
    _write(out_path, meta, records, summary)
    print(f"\nSaved {out_path}  ({len(records)} queries, {summary['wall_seconds']}s)")
    _print_summary(summary, configs)
    return summary


# ------------------------------------------------------------------ stats

def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple:  # forked from canonical
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (round((center - half) * 100, 2), round((center + half) * 100, 2))


def mcnemar(a: list, b: list) -> dict:  # forked from canonical
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    nd = n01 + n10
    if nd == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / nd
    return {"n01": n01, "n10": n10, "chi2": round(chi2, 3), "p": math.erfc(math.sqrt(chi2 / 2.0))}


def summarize(records, configs, static_bytes, coverage) -> dict:
    """Double reporting (V2 L291): single-attempt + after-recovery, per config."""
    per_cfg = {}
    for cfg in configs:
        single, final, tokens, sbytes = [], [], 0, 0
        escalated = retried = init_fail = recovered = 0
        oracle_gap = vacuous_rejected = 0
        kind_counts: dict[str, int] = {}
        for r in records:
            c = r["configs"].get(cfg)
            if not c:
                continue
            single.append(c["ok_single"])
            final.append(c["ok_final"])
            tokens += c["tokens"]
            sbytes += c["schema_bytes"]
            escalated += int(c["escalated"])
            retried += int(c["retried"])
            # judgement audit on the FINAL attempt (the reported outcome):
            acceptable = r.get("acceptable_tools") or []
            tf, esf = c.get("tool_final"), c.get("exec_status_final")
            # oracle (sel in gt) would credit, but REAL execution refused it:
            oracle_gap += int(tf in acceptable and esf != "success")
            # ran fine but semantically wrong -> conjunction rejects (validity):
            vacuous_rejected += int(esf == "success" and tf not in acceptable)
            if c["error_kind"]:
                kind_counts[c["error_kind"]] = kind_counts.get(c["error_kind"], 0) + 1
            if cfg == "pte_fd":
                if c["ok_single"] == 0:
                    init_fail += 1
                    recovered += c["ok_final"]
        n = len(single) or 1
        entry = {
            "n": len(single),
            "single_rate": round(100 * sum(single) / n, 2),
            "final_rate": round(100 * sum(final) / n, 2),
            "single_correct": sum(single), "final_correct": sum(final),
            "single_ci": wilson_ci(sum(single), len(single)),
            "final_ci": wilson_ci(sum(final), len(final)),
            "avg_tokens": round(tokens / n, 1),
            "token_reduction": round((1 - sbytes / (static_bytes * len(single) or 1)) * 100, 1),
            "escalated": escalated, "retried": retried,
            "oracle_gap": oracle_gap, "vacuous_rejected": vacuous_rejected,
            "error_kind_counts": kind_counts,
            "per_query_single": single, "per_query_final": final,
        }
        if cfg == "pte_fd":
            entry["initial_failures"] = init_fail
            entry["recovered"] = recovered
            entry["recovery_rate"] = round(100 * recovered / max(init_fail, 1), 1)
        per_cfg[cfg] = entry

    splits = {"in_sandbox": 0, "out_of_sandbox": 0}
    for r in records:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    return {
        "n_records": len(records), "splits": splits,
        "coverage_pct": coverage["coverage_pct"], "coverage": f"{coverage['n_covered']}/{coverage['universe_size']}",
        "configs": per_cfg,
    }


def _print_summary(summary, configs) -> None:
    print("  config          single%  final%   tokens  esc/ret  recovery")
    for cfg in configs:
        e = summary["configs"].get(cfg)
        if not e:
            continue
        extra = ""
        if cfg == "pte_fd":
            extra = f"  {e['recovery_rate']}%({e['recovered']}/{e['initial_failures']})"
        trig = e["escalated"] if cfg == "pte_fd" else e["retried"]
        print(f"  {cfg:14s} {e['single_rate']:6.1f} {e['final_rate']:7.1f} "
              f"{e['avg_tokens']:8.1f}  {trig:6d}{extra}")


# ------------------------------------------------------------------ aggregate

def aggregate_mode(arm: str) -> None:
    files = sorted(OUT_DIR.glob(f"exp11_*_{arm}_seed*.json"))
    files = [f for f in files if "mock" not in f.name] or files
    if not files:
        raise SystemExit(f"No E11 run files in {OUT_DIR} for arm={arm}")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    by_backend: dict[str, list] = {}
    for r in runs:
        by_backend.setdefault(r["_meta"]["backend"], []).append(r)

    report = {"arm": arm, "per_backend": {}}
    print("=" * 76)
    print(f"E11 REAL-EXECUTION AGGREGATE (arm={arm})")
    print("=" * 76)
    for bk, bruns in by_backend.items():
        bruns.sort(key=lambda r: r["_meta"]["seed"])
        cfgs = bruns[0]["_meta"]["configs"]
        print(f"\n### {bk} ({bruns[0]['_meta']['paper_name']}) seeds={[r['_meta']['seed'] for r in bruns]}")
        cfg_rep = {}
        for cfg in cfgs:
            singles = [r["summary"]["configs"][cfg]["single_rate"] for r in bruns]
            finals = [r["summary"]["configs"][cfg]["final_rate"] for r in bruns]
            cfg_rep[cfg] = {
                "single_mean": round(sum(singles) / len(singles), 2),
                "final_mean": round(sum(finals) / len(finals), 2),
                "single_per_seed": singles, "final_per_seed": finals,
            }
            print(f"  {cfg:14s} single={cfg_rep[cfg]['single_mean']:6.2f}  final={cfg_rep[cfg]['final_mean']:6.2f}")

        # key contrasts on pooled per-query arrays (final = after recovery budget)
        def pooled(cfg, key):
            return [v for r in bruns for v in r["summary"]["configs"][cfg][key]]

        contrasts = {}
        if "pte_fd" in cfgs and "static_retry" in cfgs:
            contrasts["pte_fd_vs_static_retry_final"] = mcnemar(pooled("static_retry", "per_query_final"),
                                                                pooled("pte_fd", "per_query_final"))
        if "pte_fd" in cfgs and "static" in cfgs:
            contrasts["pte_fd_final_vs_static_single"] = mcnemar(pooled("static", "per_query_single"),
                                                                 pooled("pte_fd", "per_query_final"))
        if "pte_fd" in cfgs and "pte" in cfgs:
            contrasts["pte_fd_vs_pte_single"] = mcnemar(pooled("pte", "per_query_single"),
                                                        pooled("pte_fd", "per_query_single"))
        for name, mc in contrasts.items():
            print(f"  McNemar {name}: b={mc['n10']} c={mc['n01']} chi2={mc['chi2']} p={mc['p']:.3e}")
        cfg_rep["_contrasts"] = contrasts
        report["per_backend"][bk] = cfg_rep

    out = OUT_DIR / f"exp11_aggregate_{arm}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="E11 real-execution benchmark harness")
    ap.add_argument("--backend", choices=list(PROVIDERS.keys()))
    ap.add_argument("--arm", choices=["pilot", "full", "subset"], default="pilot")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--pilot-n", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent query workers (per-thread isolated ws/engine); 1=sequential")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_mode(args.arm)
        return
    if not args.backend:
        ap.error("--backend is required unless --aggregate")

    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    for c in configs:
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; valid: {CONFIGS}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        run_single(args.backend, seed, args.arm, configs,
                   checkpoint_every=args.checkpoint_every, sleep=args.sleep,
                   resume=args.resume, model_id=args.model_id,
                   max_tokens=args.max_tokens, pilot_n=args.pilot_n,
                   workers=args.workers)


if __name__ == "__main__":
    main()
