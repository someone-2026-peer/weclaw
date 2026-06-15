#!/usr/bin/env python3
"""Experiment 10b: n=500 Tool-Selection Benchmark with LLM-based Evaluation.

ESWA extension of exp10. Runs 4 LLM-based methods on the full 500-query
bilingual dataset:
  - Static: LLM sees ALL tool schemas (token-expensive baseline)
  - PTE: LLM sees intent-filtered tools only (no escalation)
  - PTE-FD: PTE with failure-driven escalation (WeClaw's method)
  - ITR-BGE-m3: dense+BM25+cross-encoder retrieval (fair multilingual baseline)

Isolation: This script imports TOOL_ACTIONS / llm_select_tool from exp4b
(copy of exp4 with BGE-m3). exp10 and exp4 are untouched.

Usage:
    $env:WECLAW_BENCHMARK_RUN_ID="20260610_n500"
    python exp10b_tool_selection_500_llm.py --method static
    python exp10b_tool_selection_500_llm.py --method pte
    python exp10b_tool_selection_500_llm.py --method pte-fd
    python exp10b_tool_selection_500_llm.py --method itr-bge-m3
    python exp10b_tool_selection_500_llm.py --method all  (runs all 4)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from openai import OpenAI

from src.core.prompts import (
    INTENT_PRIORITY_MAP,
    INTENT_TOOL_MAPPING,
    INTENT_CATEGORIES,
    detect_intent_with_confidence,
    IntentResult,
)
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name

from benchmark_utils import get_output_dir
from dataset_utils import (
    PLANNED_ARTIFACTS_DIR,
    count_by_key,
    load_dataset_items,
    write_json,
)

# Import full tool registry and LLM selector from exp4b (BGE-m3 variant)
from exp4b_itr_bge_m3 import (
    TOOL_ACTIONS as FULL_TOOL_ACTIONS,
    build_schemas as build_full_schemas,
    get_all_tool_names as get_all_tool_names_full,
    llm_select_tool,
    ITRRetriever,
    ITR_RERANK_N,
    NO_SUITABLE_TOOL,
    is_escalation_signal,
    get_schema_tool_names as get_schema_tool_names_full,
)

# ---------- Constants ----------

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
OUTPUT_DIR = get_output_dir(__file__)
RANDOM_SEED = 42

DEFAULT_DATASET = PLANNED_ARTIFACTS_DIR / "weclaw_tool_selection_500.json"

# Map dataset tool names → exp4b tool names
# The dataset uses short names like "stock_query"; exp4b uses the same.
FULL_TOOL_NAMES = sorted(get_all_tool_names_full())
FULL_SCHEMAS = build_full_schemas(FULL_TOOL_NAMES)


# ---------- Mock Registry for PTE ----------

class MockRegistry:
    def get_all_schemas(self):
        return FULL_SCHEMAS

    def get_schemas_by_names(self, tool_names):
        return build_full_schemas(tool_names)

    def list_all_tool_names(self):
        return get_all_tool_names_full()

    def get_tool_config(self, name):
        return {}


# ---------- Ground-truth matching ----------

def _normalize_tool_name(raw: str) -> str:
    """Normalize tool name from LLM response."""
    return _extract_tool_name(raw.split("(")[0].split(".")[0].strip())


def _check_correctness(selected: str, acceptable: list[str]) -> bool:
    if not selected or selected == NO_SUITABLE_TOOL:
        return False
    return selected in acceptable


# ---------- LLM-based evaluation per method ----------

def eval_static_llm(
    client: OpenAI, query: str, acceptable: list[str],
) -> dict[str, Any]:
    """Static: LLM sees ALL tools."""
    selected, tokens, raw = llm_select_tool(client, query, FULL_SCHEMAS)
    correct = _check_correctness(selected, acceptable)
    return {
        "method": "static",
        "selected": selected,
        "correct": correct,
        "tokens": tokens,
        "tools_exposed": len(FULL_TOOL_NAMES),
    }


def eval_pte_llm(
    client: OpenAI, query: str, acceptable: list[str],
    engine: ToolExposureEngine,
) -> dict[str, Any]:
    """PTE: LLM sees intent-filtered tools only (no escalation)."""
    engine.reset()
    intent_result = detect_intent_with_confidence(query)
    pte_schemas = engine.get_schemas(intent_result)
    exposed_tools = get_schema_tool_names_full(pte_schemas)
    selected, tokens, raw = llm_select_tool(client, query, pte_schemas)
    correct = _check_correctness(selected, acceptable)
    return {
        "method": "pte",
        "selected": selected,
        "correct": correct,
        "tokens": tokens,
        "tools_exposed": len(exposed_tools),
        "exposed_tools": sorted(exposed_tools),
        "detected_intent": intent_result.primary_intent,
        "intent_confidence": round(intent_result.confidence, 4),
    }


def eval_pte_fd_llm(
    client: OpenAI, query: str, acceptable: list[str],
    engine: ToolExposureEngine,
) -> dict[str, Any]:
    """PTE-FD: PTE with failure-driven escalation."""
    engine.reset()
    intent_result = detect_intent_with_confidence(query)
    pte_schemas = engine.get_schemas(intent_result)
    pte_exposed = get_schema_tool_names_full(pte_schemas)

    selected, tokens_first, raw = llm_select_tool(client, query, pte_schemas)
    tokens_total = tokens_first
    escalated = False
    recovered = False

    if is_escalation_signal(selected, pte_exposed):
        # Retry once at same tier
        selected_retry, tokens_retry, _ = llm_select_tool(client, query, pte_schemas)
        tokens_total += tokens_retry
        if is_escalation_signal(selected_retry, pte_exposed):
            # Escalate: report failures to broaden tool set
            engine.report_failure()
            upgraded = engine.report_failure()
            if upgraded:
                fd_schemas = engine.get_schemas(intent_result)
                selected_fd, tokens_fd, _ = llm_select_tool(client, query, fd_schemas)
                tokens_total += tokens_fd
                escalated = True
                selected = selected_fd
                recovered = _check_correctness(selected, acceptable)
        else:
            selected = selected_retry

    correct = _check_correctness(selected, acceptable)
    return {
        "method": "pte-fd",
        "selected": selected,
        "correct": correct,
        "tokens": tokens_total,
        "tools_exposed": len(pte_exposed),
        "escalated": escalated,
        "recovered": recovered,
        "detected_intent": intent_result.primary_intent,
        "intent_confidence": round(intent_result.confidence, 4),
    }


def eval_itr_bge_m3_llm(
    client: OpenAI, query: str, acceptable: list[str],
    itr: ITRRetriever,
) -> dict[str, Any]:
    """ITR-BGE-m3: dense+BM25+cross-encoder → LLM selects from retrieved tools."""
    itr_tools, confidence = itr.retrieve(query, top_k=ITR_RERANK_N)
    itr_schemas = build_full_schemas(itr_tools)
    selected, tokens, raw = llm_select_tool(client, query, itr_schemas)
    correct = _check_correctness(selected, acceptable)
    return {
        "method": "itr-bge-m3",
        "selected": selected,
        "correct": correct,
        "tokens": tokens,
        "tools_exposed": len(itr_tools),
        "confidence": round(confidence, 4),
    }


# ---------- Experiment Runner ----------

def run_experiment(methods: list[str], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Run LLM-based evaluation on the 500-query dataset."""
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    engine = ToolExposureEngine(MockRegistry(), enabled=True, failures_to_upgrade=2)

    # Initialize ITR only if needed
    itr = None
    if "itr-bge-m3" in methods:
        skip_itr = os.getenv("WECLAW_EXP10B_SKIP_ITR", "").strip() == "1"
        if not skip_itr:
            itr = ITRRetriever()
        else:
            print("Skipping ITR retriever (WECLAW_EXP10B_SKIP_ITR=1)")

    # Stats per method
    method_stats: dict[str, dict] = {m: {"correct": 0, "total": 0, "tokens": []} for m in methods}
    all_details: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        query = item["query"]
        acceptable = item["acceptable_tools"]
        primary_tool = item["primary_tool"]

        print(f"\n[{i+1}/{len(items)}] {query[:60]}...")
        detail: dict[str, Any] = {
            "query_id": item["query_id"],
            "query": query,
            "language": item.get("language", "unknown"),
            "primary_intent": item.get("primary_intent", "unknown"),
            "primary_tool": primary_tool,
            "acceptable_tools": acceptable,
            "results": {},
        }

        for method in methods:
            try:
                if method == "static":
                    result = eval_static_llm(client, query, acceptable)
                elif method == "pte":
                    result = eval_pte_llm(client, query, acceptable, engine)
                elif method == "pte-fd":
                    result = eval_pte_fd_llm(client, query, acceptable, engine)
                elif method == "itr-bge-m3":
                    if itr is None:
                        result = {"method": method, "selected": "", "correct": False,
                                  "tokens": 0, "tools_exposed": 0, "skipped": True}
                    else:
                        result = eval_itr_bge_m3_llm(client, query, acceptable, itr)
                else:
                    raise ValueError(f"Unknown method: {method}")

                detail["results"][method] = result
                method_stats[method]["correct"] += int(result["correct"])
                method_stats[method]["total"] += 1
                method_stats[method]["tokens"].append(result["tokens"])

                status = "OK" if result["correct"] else "MISS"
                print(f"  {method:12s}: {status} ({result.get('selected', '?')}) tokens={result['tokens']}")

            except Exception as e:
                print(f"  {method:12s}: ERROR {e}")
                detail["results"][method] = {"method": method, "error": str(e)}
                method_stats[method]["total"] += 1

            time.sleep(0.3)  # rate limiting

        all_details.append(detail)

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 10b: n={len(items)} LLM-based Tool Selection")
    print("=" * 70)

    summary = {}
    for method in methods:
        s = method_stats[method]
        if s["total"] == 0:
            print(f"  {method:12s}: no data")
            continue
        acc = s["correct"] / s["total"] * 100
        avg_tok = float(np.mean(s["tokens"])) if s["tokens"] else 0
        # Wilson 95% CI
        n = s["total"]
        p_hat = s["correct"] / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        ci_low = max(0, center - margin) * 100
        ci_high = min(1, center + margin) * 100
        print(f"  {method:12s}: Acc={acc:.1f}% [{ci_low:.1f}%, {ci_high:.1f}%]  tokens/q={avg_tok:.0f}  n={n}")
        summary[method] = {
            "accuracy": round(acc, 2),
            "wilson_ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "avg_tokens_per_query": round(avg_tok, 1),
            "n_correct": s["correct"],
            "n_total": s["total"],
        }

    # McNemar tests between methods
    if len(methods) >= 2 and len(all_details) > 0:
        print("\n--- McNemar Tests ---")
        for i_m, m1 in enumerate(methods):
            for m2 in methods[i_m + 1:]:
                # Count discordant pairs
                b = c = 0
                for d in all_details:
                    r1 = d["results"].get(m1, {})
                    r2 = d["results"].get(m2, {})
                    if r1.get("error") or r2.get("error"):
                        continue
                    c1, c2 = r1.get("correct", False), r2.get("correct", False)
                    if c1 and not c2:
                        b += 1
                    elif c2 and not c1:
                        c += 1
                if b + c > 0:
                    # McNemar chi-squared (without continuity correction)
                    chi2 = (b - c) ** 2 / (b + c)
                    # p-value from chi-squared(1)
                    from scipy.stats import chi2 as chi2_dist
                    p_val = chi2_dist.sf(chi2, df=1)
                    print(f"  {m1} vs {m2}: b={b}, c={c}, chi2={chi2:.3f}, p={p_val:.4f}")
                    summary[f"mcnemar_{m1}_vs_{m2}"] = {
                        "b": b, "c": c, "chi2": round(chi2, 4),
                        "p_value": round(p_val, 6),
                    }
                else:
                    print(f"  {m1} vs {m2}: no discordant pairs")

    # Build output payload
    run_id = os.getenv("WECLAW_BENCHMARK_RUN_ID", "")
    payload = {
        "experiment": "exp10b_tool_selection_500_llm",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "dataset": str(DEFAULT_DATASET.name),
        "n_items": len(items),
        "methods_evaluated": methods,
        "coverage": {
            "languages": count_by_key(items, "language"),
            "primary_intents": count_by_key(items, "primary_intent"),
        },
        "summary": summary,
        "details": all_details,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="WeClaw exp10b: n=500 LLM tool-selection benchmark")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET) if DEFAULT_DATASET.exists() else None,
        help="Path to weclaw_tool_selection_500 JSON.",
    )
    parser.add_argument(
        "--method",
        choices=["static", "pte", "pte-fd", "itr-bge-m3", "all"],
        default="all",
        help="Which LLM-based method(s) to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    _, _, items = load_dataset_items(args.dataset, [])
    if not items:
        print("ERROR: No items loaded. Check --dataset path.")
        sys.exit(1)
    print(f"Loaded {len(items)} queries from dataset")

    # Determine methods
    if args.method == "all":
        methods = ["static", "pte", "pte-fd", "itr-bge-m3"]
    else:
        methods = [args.method]
    print(f"Evaluating methods: {methods}")

    payload = run_experiment(methods, items)

    # Save
    out_path = OUTPUT_DIR / f"exp10b_{'_'.join(methods)}_results.json"
    write_json(out_path, payload)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
