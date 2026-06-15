#!/usr/bin/env python3
"""Experiment 8: execution-level multistep benchmark.

This version upgrades the earlier scaffold into a runnable baseline benchmark.
It evaluates three deterministic methods:

- Static: select from the full tool set
- Keyword: direct keyword matcher over all tools
- PTE-FD: intent detection + ToolExposureEngine + execution-level escalation

Execution-level escalation is simulated at the benchmark layer:
if the initially selected tool for a step is outside ``acceptable_tools``, that
selection is treated as an execution failure and PTE-FD upgrades the exposure
tier before retrying the same step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.prompts import INTENT_PRIORITY_MAP, detect_intent_with_confidence
from src.core.tool_exposure import ToolExposureEngine

from dataset_utils import count_by_key, get_default_output_path, load_dataset_items, write_json
from exp10_tool_selection_500 import (
    ALL_TOOL_NAMES,
    MockRegistry,
    get_schema_tool_names,
    keyword_match_tool,
    rank_tools,
)

OUTPUT_PATH = get_default_output_path(__file__, "exp8_execution_level_multistep_results.json")


BUILTIN_MULTISTEP_SAMPLE: list[dict[str, Any]] = [
    {
        "task_id": "sample_fin_01",
        "task": "查询苹果股价后生成简要交易建议",
        "domain": "financial_activity",
        "difficulty": "medium",
        "steps": [
            {
                "step_id": "s1",
                "instruction": "查询苹果公司的实时股价",
                "acceptable_tools": ["stock_query", "search"],
                "notes": "首选 stock_query",
            },
            {
                "step_id": "s2",
                "instruction": "分析股价走势并生成交易建议",
                "acceptable_tools": ["quant_trading", "stock_query"],
                "notes": "首选 quant_trading",
            },
        ],
        "success_criteria": ["完成股价查询", "完成交易建议生成"],
    },
    {
        "task_id": "sample_res_01",
        "task": "搜索论文并下载开放 PDF",
        "domain": "research",
        "difficulty": "medium",
        "steps": [
            {
                "step_id": "s1",
                "instruction": "搜索关于 LLM agent tool use 的论文",
                "acceptable_tools": ["literature_search", "search"],
                "notes": "",
            },
            {
                "step_id": "s2",
                "instruction": "检查论文是否有开放 PDF",
                "acceptable_tools": ["oss_pdf_search", "literature_search"],
                "notes": "",
            },
            {
                "step_id": "s3",
                "instruction": "下载开放 PDF",
                "acceptable_tools": ["oss_pdf_download", "oss_pdf_search"],
                "notes": "",
            },
        ],
        "success_criteria": ["找到论文", "确认开放 PDF", "成功下载 PDF"],
    },
    {
        "task_id": "sample_sys_01",
        "task": "查看系统状态并清理临时文件",
        "domain": "system_admin",
        "difficulty": "easy",
        "steps": [
            {
                "step_id": "s1",
                "instruction": "查看 CPU 和内存使用率",
                "acceptable_tools": ["system_monitor", "shell"],
                "notes": "",
            },
            {
                "step_id": "s2",
                "instruction": "清理系统临时文件",
                "acceptable_tools": ["shell", "file"],
                "notes": "",
            },
        ],
        "success_criteria": ["查看系统状态", "清理临时文件"],
    },
]
def build_coverage(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(len(task.get("steps", [])) for task in tasks)
    step_domains = [{"domain": task.get("domain", "unknown")} for task in tasks]
    return {
        "n_tasks": len(tasks),
        "n_steps": total_steps,
        "domains": count_by_key(step_domains, "domain"),
        "difficulties": count_by_key(tasks, "difficulty"),
    }


def evaluate_task_static(task: dict[str, Any]) -> dict[str, Any]:
    step_results = []
    for step in task["steps"]:
        acceptable = set(step["acceptable_tools"])
        selected, top_candidates = rank_tools(step["instruction"], ALL_TOOL_NAMES)
        success = selected in acceptable if selected else False
        step_results.append(
            {
                "step_id": step["step_id"],
                "instruction": step["instruction"],
                "acceptable_tools": step["acceptable_tools"],
                "selected_tool": selected,
                "tool_status": "success" if success else "error",
                "escalated": False,
                "recovered": False,
                "considered_tools": len(ALL_TOOL_NAMES),
                "top_candidates": top_candidates,
            }
        )
    return {"task_complete": all(step["tool_status"] == "success" for step in step_results), "steps": step_results}


def evaluate_task_keyword(task: dict[str, Any]) -> dict[str, Any]:
    step_results = []
    for step in task["steps"]:
        acceptable = set(step["acceptable_tools"])
        selected, top_candidates = keyword_match_tool(step["instruction"])
        if not selected:
            selected, top_candidates = rank_tools(step["instruction"], ALL_TOOL_NAMES)
        success = selected in acceptable if selected else False
        step_results.append(
            {
                "step_id": step["step_id"],
                "instruction": step["instruction"],
                "acceptable_tools": step["acceptable_tools"],
                "selected_tool": selected,
                "tool_status": "success" if success else "error",
                "escalated": False,
                "recovered": False,
                "considered_tools": len(top_candidates) if top_candidates else len(ALL_TOOL_NAMES),
                "top_candidates": top_candidates,
            }
        )
    return {"task_complete": all(step["tool_status"] == "success" for step in step_results), "steps": step_results}


def evaluate_task_pte_fd(task: dict[str, Any]) -> dict[str, Any]:
    registry = MockRegistry()
    engine = ToolExposureEngine(registry, enabled=True, failures_to_upgrade=1)
    engine.reset()
    step_results = []

    for step in task["steps"]:
        acceptable = set(step["acceptable_tools"])
        instruction = step["instruction"]
        intent_result = detect_intent_with_confidence(instruction)
        priority = INTENT_PRIORITY_MAP.get(intent_result.primary_intent, {})
        recommended = set(priority.get("recommended", []))
        alternative = set(priority.get("alternative", []))

        pte_schemas = engine.get_schemas(intent_result)
        exposed_tools = get_schema_tool_names(pte_schemas)
        selected, top_candidates = rank_tools(
            instruction,
            exposed_tools,
            recommended=recommended,
            alternative=alternative,
        )
        success = selected in acceptable if selected else False
        escalated = False
        recovered = False
        retry_selected = None
        retry_candidates: list[dict[str, Any]] = []

        if not success:
            upgraded = engine.report_failure()
            if upgraded:
                escalated = True
                retry_schemas = engine.get_schemas(intent_result)
                retry_exposed = get_schema_tool_names(retry_schemas)
                retry_selected, retry_candidates = rank_tools(
                    instruction,
                    retry_exposed,
                    recommended=recommended,
                    alternative=alternative,
                )
                if retry_selected in acceptable:
                    selected = retry_selected
                    top_candidates = retry_candidates
                    success = True
                    recovered = True
                    exposed_tools = retry_exposed
                    engine.report_success()
            # If still failing, keep the failure counter/tier for later steps.
        else:
            engine.report_success()

        step_results.append(
            {
                "step_id": step["step_id"],
                "instruction": instruction,
                "acceptable_tools": step["acceptable_tools"],
                "selected_tool": selected,
                "tool_status": "success" if success else "error",
                "escalated": escalated,
                "recovered": recovered,
                "considered_tools": len(exposed_tools),
                "detected_intent": intent_result.primary_intent,
                "intent_confidence": round(intent_result.confidence, 4),
                "top_candidates": top_candidates,
                "retry_selected_tool": retry_selected,
                "retry_top_candidates": retry_candidates,
                "current_tier_after_step": engine.current_tier,
            }
        )

    return {"task_complete": all(step["tool_status"] == "success" for step in step_results), "steps": step_results}


def summarize_method(method_details: list[dict[str, Any]]) -> dict[str, Any]:
    n_tasks = len(method_details)
    n_task_complete = sum(1 for item in method_details if item["task_complete"])
    step_rows = [step for item in method_details for step in item["steps"]]
    n_steps = len(step_rows)
    n_step_success = sum(1 for step in step_rows if step["tool_status"] == "success")
    n_escalated = sum(1 for step in step_rows if step["escalated"])
    n_recovered = sum(1 for step in step_rows if step["recovered"])
    avg_considered = sum(step["considered_tools"] for step in step_rows) / n_steps if n_steps else 0.0
    return {
        "task_complete": n_task_complete,
        "task_complete_rate": round(n_task_complete / n_tasks * 100, 2) if n_tasks else 0.0,
        "step_success": n_step_success,
        "step_accuracy": round(n_step_success / n_steps * 100, 2) if n_steps else 0.0,
        "escalated_steps": n_escalated,
        "recovered_steps": n_recovered,
        "recovery_rate": round(n_recovered / n_escalated * 100, 2) if n_escalated else 0.0,
        "avg_considered_tools": round(avg_considered, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WeClaw exp8 execution-level multistep benchmark")
    parser.add_argument("--dataset", help="Optional path to weclaw_multistep_100 style JSON.")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "evaluate"],
        default="dry_run",
        help="dry_run only validates dataset coverage; evaluate runs Static/Keyword/PTE-FD baselines.",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path.")
    args = parser.parse_args()

    dataset_name, version, tasks = load_dataset_items(args.dataset, BUILTIN_MULTISTEP_SAMPLE)
    if not tasks:
        raise RuntimeError("No multistep tasks found.")

    coverage = build_coverage(tasks)

    if args.mode == "dry_run":
        payload = {
            "experiment": "exp8_execution_level_multistep",
            "status": "dataset_validated_only",
            "mode": args.mode,
            "dataset": dataset_name,
            "dataset_version": version,
            "summary": {
                **coverage,
                "implemented_methods": ["static", "keyword", "pte_fd"],
            },
            "details": [],
        }
    else:
        static_details = []
        keyword_details = []
        pte_details = []
        for task in tasks:
            static_details.append({"task_id": task["task_id"], **evaluate_task_static(task)})
            keyword_details.append({"task_id": task["task_id"], **evaluate_task_keyword(task)})
            pte_details.append({"task_id": task["task_id"], **evaluate_task_pte_fd(task)})

        payload = {
            "experiment": "exp8_execution_level_multistep",
            "status": "evaluated_with_real_heuristic_baselines",
            "mode": args.mode,
            "dataset": dataset_name,
            "dataset_version": version,
            "summary": {
                **coverage,
                "implemented_methods": ["static", "keyword", "pte_fd"],
                "methods": {
                    "static": summarize_method(static_details),
                    "keyword": summarize_method(keyword_details),
                    "pte_fd": summarize_method(pte_details),
                },
            },
            "details": {
                "static": static_details,
                "keyword": keyword_details,
                "pte_fd": pte_details,
            },
        }

    write_json(args.output, payload)
    print(f"Saved output file: {Path(args.output).name}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
