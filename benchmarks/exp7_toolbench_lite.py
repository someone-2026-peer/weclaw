#!/usr/bin/env python3
"""Experiment 7: ToolBench-lite adapter smoke test.

This script is a minimal public-benchmark adapter, not a full ToolBench
reproduction. It can evaluate a small JSON subset exported from ToolBench-like
data, and it also includes a tiny built-in public-API-style smoke subset so the
pipeline can run without downloading external datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.prompts import detect_intent_with_confidence
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name
from benchmark_utils import get_output_dir

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
OUTPUT_DIR = get_output_dir(__file__)
RANDOM_SEED = 42
NO_SUITABLE_TOOL = "__need_more_tools__"


TOOL_DESCRIPTIONS: dict[str, str] = {
    "shell": "Execute local shell commands for system administration tasks.",
    "file": "Read, write, move, and inspect local files.",
    "screen": "Capture screenshots or inspect visible screen content.",
    "search": "Search the web for current public information.",
    "browser": "Open URLs and perform simple browser navigation.",
    "browser_use": "Perform multi-step browser automation.",
    "stock_query": "Query market prices, indices, and stock facts.",
    "quant_trading": "Analyze trading signals, backtests, and portfolio risk.",
    "fred_query": "Query macroeconomic time series and public economic data.",
    "weather": "Query weather information for a city or region.",
    "knowledge_rag": "Search local knowledge bases and indexed documents.",
    "literature_search": "Search academic literature metadata.",
    "oss_pdf_search": "Search whether an open-access PDF exists.",
    "oss_pdf_download": "Download open-access scholarly PDFs.",
    "pdf_tool": "Merge, split, inspect, and manipulate PDF files.",
    "format_converter": "Convert between document formats.",
    "doc_generator": "Generate Word or structured text documents.",
    "data_processor": "Read and clean spreadsheet or CSV data.",
    "data_visualization": "Create charts and plots from structured data.",
    "statistics": "Run statistical summaries and analyses.",
    "ocr": "Recognize text from images.",
    "speech_to_text": "Transcribe audio files into text.",
    "voice_input": "Capture or transcribe live voice input.",
    "image_generator": "Generate images from prompts.",
    "stock_photo": "Search free stock images.",
    "tool_info": "List available tools and their capabilities.",
}


BUILTIN_TOOLBENCH_LITE: list[dict[str, Any]] = [
    {
        "id": "tb_weather_01",
        "query": "Find the current weather forecast for Kuala Lumpur.",
        "acceptable_tools": ["weather", "search"],
        "domain": "weather",
    },
    {
        "id": "tb_finance_01",
        "query": "Get the latest price for Apple stock and summarize the movement.",
        "acceptable_tools": ["stock_query", "search"],
        "domain": "finance",
    },
    {
        "id": "tb_macro_01",
        "query": "Retrieve US GDP growth data from a public economic data source.",
        "acceptable_tools": ["fred_query", "search"],
        "domain": "macro",
    },
    {
        "id": "tb_browser_01",
        "query": "Open a public website and extract the headline text.",
        "acceptable_tools": ["browser", "browser_use"],
        "domain": "browser",
    },
    {
        "id": "tb_docs_01",
        "query": "Convert a markdown report into a PDF document.",
        "acceptable_tools": ["format_converter", "pdf_tool", "doc_generator"],
        "domain": "document",
    },
    {
        "id": "tb_data_01",
        "query": "Read a CSV file and compute basic summary statistics.",
        "acceptable_tools": ["data_processor", "statistics", "file"],
        "domain": "data",
    },
    {
        "id": "tb_chart_01",
        "query": "Create a bar chart from monthly revenue data.",
        "acceptable_tools": ["data_visualization", "statistics"],
        "domain": "data",
    },
    {
        "id": "tb_literature_01",
        "query": "Search recent papers about LLM agents and tool use.",
        "acceptable_tools": ["literature_search", "search"],
        "domain": "research",
    },
    {
        "id": "tb_pdf_01",
        "query": "Check whether an open-access PDF is available for a paper DOI.",
        "acceptable_tools": ["oss_pdf_search", "literature_search"],
        "domain": "research",
    },
    {
        "id": "tb_ocr_01",
        "query": "Extract text from a scanned receipt image.",
        "acceptable_tools": ["ocr", "file"],
        "domain": "multimedia",
    },
    {
        "id": "tb_audio_01",
        "query": "Transcribe an audio recording into text.",
        "acceptable_tools": ["speech_to_text", "voice_input"],
        "domain": "multimedia",
    },
    {
        "id": "tb_file_01",
        "query": "Read a local text file and summarize its contents.",
        "acceptable_tools": ["file"],
        "domain": "file",
    },
]


def build_schemas(tool_names: set[str]) -> list[dict[str, Any]]:
    schemas = []
    for tool_name in sorted(tool_names):
        description = TOOL_DESCRIPTIONS.get(tool_name, f"Tool: {tool_name}")
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": f"{tool_name}_execute",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            }
        )
    return schemas


def get_all_tool_names() -> set[str]:
    return set(TOOL_DESCRIPTIONS)


def get_schema_tool_names(tool_schemas: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for schema in tool_schemas:
        function_name = schema.get("function", {}).get("name", "")
        if function_name:
            names.add(_extract_tool_name(function_name))
    return names


def is_escalation_signal(selected_tool: str, exposed_tools: set[str]) -> bool:
    return (
        not selected_tool
        or selected_tool == NO_SUITABLE_TOOL
        or selected_tool not in exposed_tools
    )


def llm_select_tool(client: OpenAI, query: str, tool_schemas: list[dict[str, Any]]) -> tuple[str, int, str]:
    system_prompt = (
        "You are a tool selection evaluator. Given a user query and available tools, "
        "select the single most appropriate tool function name. If none of the listed "
        "tools is suitable, respond exactly NEED_MORE_TOOLS. Return only the function name."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Available tools:\n{json.dumps(tool_schemas, ensure_ascii=False)}\n\nUser query: {query}",
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    if raw.upper() == "NEED_MORE_TOOLS":
        selected = NO_SUITABLE_TOOL
    else:
        selected = _extract_tool_name(raw.split("(")[0].split(".")[0].strip())
    tokens = response.usage.prompt_tokens if response.usage else 0
    return selected, tokens, raw


def load_tasks(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return BUILTIN_TOOLBENCH_LITE

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("tasks", [])
    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        query = row.get("query") or row.get("instruction") or row.get("user_query")
        acceptable = row.get("acceptable_tools") or row.get("ground_truth_tools") or row.get("tools")
        if not query or not acceptable:
            continue
        tasks.append(
            {
                "id": row.get("id", f"external_{idx}"),
                "query": query,
                "acceptable_tools": list(acceptable),
                "domain": row.get("domain", "external"),
            }
        )
    return tasks


class MockRegistry:
    def get_all_schemas(self) -> list[dict[str, Any]]:
        return build_schemas(get_all_tool_names())

    def get_schemas_by_names(self, tool_names: set[str]) -> list[dict[str, Any]]:
        return build_schemas(set(tool_names) & get_all_tool_names())

    def list_all_tool_names(self) -> set[str]:
        return get_all_tool_names()

    def get_tool_config(self, name: str) -> dict[str, Any]:
        return {}


def evaluate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    registry = MockRegistry()
    engine = ToolExposureEngine(registry, enabled=True, failures_to_upgrade=2)

    all_schemas = build_schemas(get_all_tool_names())
    results = {
        "static": {"correct": 0, "tokens": []},
        "pte_fd": {"correct": 0, "tokens": [], "escalated": 0, "recovered": 0},
    }
    details = []

    for index, task in enumerate(tasks, start=1):
        query = task["query"]
        acceptable = set(task["acceptable_tools"])
        print(f"[{index}/{len(tasks)}] {task['id']}: {query[:60]}")

        selected_static, tokens_static, raw_static = llm_select_tool(client, query, all_schemas)
        static_ok = selected_static in acceptable
        results["static"]["correct"] += int(static_ok)
        results["static"]["tokens"].append(tokens_static)

        engine.reset()
        intent_result = detect_intent_with_confidence(query)
        pte_schemas = engine.get_schemas(intent_result)
        pte_exposed = get_schema_tool_names(pte_schemas)
        selected_pte, tokens_pte, raw_pte = llm_select_tool(client, query, pte_schemas)
        pte_ok = selected_pte in acceptable
        escalated = False
        recovered = False

        if is_escalation_signal(selected_pte, pte_exposed):
            selected_retry, tokens_retry, raw_retry = llm_select_tool(client, query, pte_schemas)
            tokens_pte += tokens_retry
            if is_escalation_signal(selected_retry, pte_exposed):
                engine.report_failure()
                upgraded = engine.report_failure()
                if upgraded:
                    pte_fd_schemas = engine.get_schemas(intent_result)
                    selected_fd, tokens_fd, raw_fd = llm_select_tool(client, query, pte_fd_schemas)
                    tokens_pte += tokens_fd
                    selected_pte = selected_fd
                    pte_ok = selected_pte in acceptable
                    escalated = True
                    recovered = pte_ok
            else:
                selected_pte = selected_retry
                pte_ok = selected_pte in acceptable

        results["pte_fd"]["correct"] += int(pte_ok)
        results["pte_fd"]["tokens"].append(tokens_pte)
        results["pte_fd"]["escalated"] += int(escalated)
        results["pte_fd"]["recovered"] += int(recovered)

        details.append(
            {
                "id": task["id"],
                "query": query,
                "acceptable_tools": sorted(acceptable),
                "domain": task.get("domain", ""),
                "static": {"selected": selected_static, "correct": static_ok, "tokens": tokens_static},
                "pte_fd": {
                    "selected": selected_pte,
                    "correct": pte_ok,
                    "tokens": tokens_pte,
                    "escalated": escalated,
                    "recovered": recovered,
                },
            }
        )
        time.sleep(0.5)

    total = len(tasks)
    summary = {
        "static": {
            "accuracy": results["static"]["correct"] / total * 100 if total else 0,
            "avg_tokens": float(np.mean(results["static"]["tokens"])) if total else 0,
            "n_correct": results["static"]["correct"],
            "n_total": total,
        },
        "pte_fd": {
            "accuracy": results["pte_fd"]["correct"] / total * 100 if total else 0,
            "avg_tokens": float(np.mean(results["pte_fd"]["tokens"])) if total else 0,
            "n_correct": results["pte_fd"]["correct"],
            "n_total": total,
            "escalated": results["pte_fd"]["escalated"],
            "recovered": results["pte_fd"]["recovered"],
            "recovery_rate": (
                results["pte_fd"]["recovered"] / results["pte_fd"]["escalated"] * 100
                if results["pte_fd"]["escalated"] else 0
            ),
        },
    }
    return {
        "experiment": "toolbench_lite_adapter",
        "model": MODEL,
        "dataset": "builtin_smoke_subset" if total == len(BUILTIN_TOOLBENCH_LITE) else "external_json",
        "summary": summary,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", help="Optional ToolBench-lite-style JSON file.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "exp7_toolbench_lite_results.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.input_json)
    if not tasks:
        raise RuntimeError("No valid tasks found. Expected query and acceptable_tools fields.")

    output = evaluate(tasks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
