#!/usr/bin/env python3
"""Experiment 10: WeClaw tool-selection benchmark.

This version upgrades the earlier scaffold into a runnable benchmark with three
real baselines that do not require external APIs:

- Static: global lexical ranking over all candidate tools
- Keyword: direct keyword-based tool matcher
- PTE: intent detection + ToolExposureEngine + lexical reranking

These baselines are deterministic and intended as the first reproducible public
benchmark layer before later adding LLM-based Static / PTE-FD / ITR / Hybrid
methods.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.prompts import INTENT_PRIORITY_MAP, INTENT_TOOL_MAPPING, detect_intent_with_confidence
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name

from dataset_utils import (
    PLANNED_ARTIFACTS_DIR,
    count_by_key,
    get_default_output_path,
    load_dataset_items,
    write_json,
)

DEFAULT_DATASET = PLANNED_ARTIFACTS_DIR / "weclaw_tool_selection_500.json"
OUTPUT_PATH = get_default_output_path(__file__, "exp10_tool_selection_500_results.json")


TOOL_PROFILES: dict[str, dict[str, Any]] = {
    "search": {"keywords": ["搜索", "search", "find", "lookup", "查找", "查询网页"], "description": "General web search."},
    "browser": {"keywords": ["打开网页", "网页标题", "访问网站", "open url", "headline", "visit website"], "description": "Open and inspect web pages."},
    "browser_use": {"keywords": ["自动化", "自动登录", "click", "form", "browser automation", "多步网页"], "description": "Multi-step browser automation."},
    "mcp_browserbase": {"keywords": ["cloud browser", "browserbase", "云端浏览器"], "description": "Cloud browser automation."},
    "file": {"keywords": ["读取文件", "read file", "写入文件", "rename file", "文件夹", "目录", "copy file", "move file"], "description": "Local file operations."},
    "shell": {"keywords": ["shell", "命令行", "terminal", "cleanup", "批处理", "临时文件", "process"], "description": "Shell command execution."},
    "system_monitor": {"keywords": ["cpu", "内存", "memory", "disk", "temperature", "resource usage", "系统状态"], "description": "System monitoring."},
    "app_control": {"keywords": ["启动应用", "launch app", "close app", "打开应用"], "description": "App control."},
    "screen": {"keywords": ["截屏", "screenshot", "screen capture", "截图"], "description": "Screen capture."},
    "clipboard": {"keywords": ["剪贴板", "clipboard"], "description": "Clipboard operations."},
    "notify": {"keywords": ["通知", "notification", "提醒弹窗"], "description": "Notifications."},
    "weather": {"keywords": ["天气", "forecast", "weather", "temperature"], "description": "Weather lookup."},
    "datetime_tool": {"keywords": ["日期", "时间", "date", "time"], "description": "Date and time."},
    "calculator": {"keywords": ["计算", "calculate", "math", "算一下"], "description": "Calculator."},
    "cron": {"keywords": ["提醒", "定时", "schedule", "reminder", "tomorrow", "9 am", "早上九点"], "description": "Scheduled tasks and reminders."},
    "knowledge_rag": {"keywords": ["知识库", "rag", "local knowledge", "indexed docs"], "description": "Knowledge base retrieval."},
    "batch_paper_analyzer": {"keywords": ["批量论文", "paper analyzer"], "description": "Batch paper analysis."},
    "literature_search": {"keywords": ["论文", "paper", "literature", "academic", "scholar", "文献", "doi"], "description": "Academic literature search."},
    "oss_pdf_search": {"keywords": ["开放 pdf", "open-access pdf", "check pdf", "pdf exists", "查 pdf"], "description": "Open-access PDF search."},
    "oss_pdf_download": {"keywords": ["下载 pdf", "download pdf", "下载论文", "download paper pdf"], "description": "Open-access PDF download."},
    "poetry": {"keywords": ["诗", "诗词", "李白", "唐诗", "poem"], "description": "Poetry retrieval."},
    "chat_history": {"keywords": ["chat history", "previous chat", "聊天历史", "之前聊过"], "description": "Chat history retrieval."},
    "todo": {"keywords": ["待办", "todo", "task item", "事项"], "description": "Todo management."},
    "diary": {"keywords": ["日记", "diary"], "description": "Diary management."},
    "health": {"keywords": ["健康", "体重", "血压", "health"], "description": "Health data management."},
    "family_member": {"keywords": ["家庭成员", "家人资料", "family member"], "description": "Family member management."},
    "course_schedule": {"keywords": ["课程表", "class schedule"], "description": "Course schedule."},
    "meal_menu": {"keywords": ["食谱", "菜单", "meal menu"], "description": "Meal menu."},
    "stock_query": {"keywords": ["股价", "股票", "行情", "price", "quote", "ticker", "市场价格", "实时股价"], "description": "Stock price lookup."},
    "quant_trading": {"keywords": ["交易信号", "quant", "回测", "仓位", "风险评估", "trading signal"], "description": "Quant trading analysis."},
    "fred_query": {"keywords": ["fred", "gdp", "cpi", "失业率", "economic data", "宏观数据"], "description": "Macroeconomic public data."},
    "voice_input": {"keywords": ["语音输入", "live voice", "microphone", "实时语音"], "description": "Live voice input."},
    "voice_output": {"keywords": ["语音播报", "text to speech"], "description": "Text-to-speech."},
    "ocr": {"keywords": ["ocr", "识别文字", "扫描件", "receipt", "图片文字"], "description": "OCR text recognition."},
    "speech_to_text": {"keywords": ["transcribe", "audio", "recording", "录音", "音频转写", "speech to text"], "description": "Audio transcription."},
    "document_scanner": {"keywords": ["document scanner", "扫描文档"], "description": "Document scanning."},
    "stock_photo": {"keywords": ["图库", "stock image", "free image"], "description": "Stock photo search."},
    "image_generator": {"keywords": ["生成图片", "image generation", "ai image"], "description": "Image generation."},
    "pdf_tool": {"keywords": ["merge pdf", "split pdf", "合并 pdf", "pdf 操作"], "description": "PDF operations."},
    "format_converter": {"keywords": ["convert", "格式转换", "markdown to pdf", "docx to pdf"], "description": "Format conversion."},
    "doc_generator": {"keywords": ["word 文档", "generate document", "docx"], "description": "Document generation."},
    "ppt_generator": {"keywords": ["ppt", "slides", "演示文稿"], "description": "PPT generation."},
    "data_processor": {"keywords": ["csv", "xlsx", "spreadsheet", "读取表格", "clean data", "sales.csv"], "description": "Structured data processing."},
    "data_visualization": {"keywords": ["chart", "plot", "bar chart", "line chart", "图表", "柱状图"], "description": "Data visualization."},
    "statistics": {"keywords": ["统计", "summary statistics", "均值", "variance"], "description": "Statistical analysis."},
    "log_viewer": {"keywords": ["日志", "log", "error log", "工具调用错误"], "description": "Log viewer."},
    "tool_audit": {"keywords": ["审计", "audit", "工具调用报告"], "description": "Tool audit report."},
    "codebase_search": {"keywords": ["代码库", "codebase", "implementation", "哪里实现"], "description": "Codebase search."},
    "self_control": {"keywords": ["重启 agent", "self control", "reset agent"], "description": "Self control and runtime management."},
    "experience_recall": {"keywords": ["历史经验", "previous fix", "经验回忆"], "description": "Experience recall."},
    "research_landscape": {"keywords": ["landscape", "地形图", "研究全景"], "description": "Research landscape."},
    "research_lineage": {"keywords": ["lineage", "谱系", "思想谱系"], "description": "Research lineage tracing."},
    "journal_intelligence": {"keywords": ["journal", "期刊", "投稿建议", "suitable venue"], "description": "Journal intelligence."},
    "paper_lifecycle": {"keywords": ["paper project", "论文项目", "写作项目", "投稿阶段"], "description": "Paper lifecycle management."},
    "qualitative_analysis": {"keywords": ["qualitative", "定性分析", "coding interview"], "description": "Qualitative analysis."},
}

ALL_TOOL_NAMES = sorted(TOOL_PROFILES.keys())

BUILTIN_TOOL_SELECTION_SAMPLE: list[dict[str, Any]] = [
    {
        "query_id": "sample_001",
        "query": "查询苹果公司的实时股价",
        "language": "zh",
        "primary_intent": "financial_activity",
        "sub_intent": "stock_price_lookup",
        "acceptable_tools": ["stock_query", "search"],
        "primary_tool": "stock_query",
        "negative_tools": ["ocr", "voice_input"],
        "notes": "首选金融专用工具，不应误选通用多媒体工具",
    },
    {
        "query_id": "sample_002",
        "query": "Search recent papers about LLM agents and tool use.",
        "language": "en",
        "primary_intent": "knowledge",
        "sub_intent": "literature_search",
        "acceptable_tools": ["literature_search", "search"],
        "primary_tool": "literature_search",
        "negative_tools": ["stock_query", "ocr"],
        "notes": "学术检索优先于通用搜索",
    },
    {
        "query_id": "sample_003",
        "query": "把扫描件里的文字识别出来",
        "language": "zh",
        "primary_intent": "multimedia",
        "sub_intent": "ocr",
        "acceptable_tools": ["ocr", "file"],
        "primary_tool": "ocr",
        "negative_tools": ["voice_input", "stock_photo"],
        "notes": "OCR 是主工具，file 仅作辅助",
    },
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize_english(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_text(text)) if token}


def build_schema(tool_name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": f"{tool_name}_execute",
            "description": TOOL_PROFILES[tool_name]["description"],
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }


class MockRegistry:
    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [build_schema(tool_name) for tool_name in ALL_TOOL_NAMES]

    def get_schemas_by_names(self, tool_names: set[str]) -> list[dict[str, Any]]:
        return [build_schema(tool_name) for tool_name in ALL_TOOL_NAMES if tool_name in tool_names]

    def list_all_tool_names(self) -> set[str]:
        return set(ALL_TOOL_NAMES)

    def get_tool_config(self, name: str) -> dict[str, Any]:
        return {}


def get_schema_tool_names(tool_schemas: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    sorted_profiles = sorted(ALL_TOOL_NAMES, key=len, reverse=True)
    for schema in tool_schemas:
        function_name = schema["function"]["name"]
        matched = None
        for profile_name in sorted_profiles:
            if function_name.startswith(profile_name + "_") or function_name == profile_name:
                matched = profile_name
                break
        names.append(matched or _extract_tool_name(function_name))
    return names


def score_tool(query: str, tool_name: str) -> float:
    query_norm = normalize_text(query)
    query_tokens = tokenize_english(query_norm)
    profile = TOOL_PROFILES[tool_name]
    score = 0.0

    name_tokens = tokenize_english(tool_name)
    description_tokens = tokenize_english(profile["description"])
    score += len(query_tokens & name_tokens) * 2.0
    score += len(query_tokens & description_tokens) * 1.2

    for keyword in profile["keywords"]:
        keyword_norm = normalize_text(keyword)
        if not keyword_norm:
            continue
        if " " in keyword_norm:
            if keyword_norm in query_norm:
                score += 3.0
        elif any("\u4e00" <= ch <= "\u9fff" for ch in keyword_norm):
            if keyword_norm in query_norm:
                score += 2.5
        elif keyword_norm in query_tokens:
            score += 2.5

    return score


def rank_tools(query: str, candidate_tools: list[str], recommended: set[str] | None = None, alternative: set[str] | None = None) -> tuple[str | None, list[dict[str, Any]]]:
    recommended = recommended or set()
    alternative = alternative or set()
    ranked = []
    for tool_name in candidate_tools:
        base = score_tool(query, tool_name)
        bonus = 1.5 if tool_name in recommended else 0.5 if tool_name in alternative else 0.0
        ranked.append(
            {
                "tool": tool_name,
                "base_score": round(base, 4),
                "bonus": bonus,
                "score": round(base + bonus, 4),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], -item["base_score"], item["tool"]))
    selected = ranked[0]["tool"] if ranked else None
    return selected, ranked[:5]


def keyword_match_tool(query: str) -> tuple[str | None, list[dict[str, Any]]]:
    ranked = []
    for tool_name in ALL_TOOL_NAMES:
        score = 0.0
        query_norm = normalize_text(query)
        for keyword in TOOL_PROFILES[tool_name]["keywords"]:
            keyword_norm = normalize_text(keyword)
            if not keyword_norm:
                continue
            if " " in keyword_norm and keyword_norm in query_norm:
                score += 3.0
            elif any("\u4e00" <= ch <= "\u9fff" for ch in keyword_norm) and keyword_norm in query_norm:
                score += 2.5
            elif keyword_norm in tokenize_english(query_norm):
                score += 2.5
        if score > 0:
            ranked.append({"tool": tool_name, "score": round(score, 4)})

    ranked.sort(key=lambda item: (-item["score"], item["tool"]))
    selected = ranked[0]["tool"] if ranked else None
    return selected, ranked[:5]


def evaluate_methods(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = MockRegistry()
    engine = ToolExposureEngine(registry, enabled=True, failures_to_upgrade=2)
    methods = {
        "static": {"acceptable": 0, "primary": 0, "considered_tools": []},
        "keyword": {"acceptable": 0, "primary": 0, "considered_tools": []},
        "pte": {"acceptable": 0, "primary": 0, "considered_tools": []},
    }
    details = []

    for item in items:
        acceptable = set(item["acceptable_tools"])
        primary_tool = item["primary_tool"]
        query = item["query"]

        static_selected, static_top = rank_tools(query, ALL_TOOL_NAMES)
        static_ok = static_selected in acceptable if static_selected else False
        static_primary = static_selected == primary_tool
        methods["static"]["acceptable"] += int(static_ok)
        methods["static"]["primary"] += int(static_primary)
        methods["static"]["considered_tools"].append(len(ALL_TOOL_NAMES))

        keyword_selected, keyword_top = keyword_match_tool(query)
        if not keyword_selected:
            keyword_selected, keyword_top = rank_tools(query, ALL_TOOL_NAMES)
        keyword_ok = keyword_selected in acceptable if keyword_selected else False
        keyword_primary = keyword_selected == primary_tool
        methods["keyword"]["acceptable"] += int(keyword_ok)
        methods["keyword"]["primary"] += int(keyword_primary)
        methods["keyword"]["considered_tools"].append(len(keyword_top) if keyword_top else len(ALL_TOOL_NAMES))

        intent_result = detect_intent_with_confidence(query)
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        exposed_tools = get_schema_tool_names(pte_schemas)
        priority = INTENT_PRIORITY_MAP.get(intent_result.primary_intent, {})
        recommended = set(priority.get("recommended", []))
        alternative = set(priority.get("alternative", []))
        pte_selected, pte_top = rank_tools(query, exposed_tools, recommended=recommended, alternative=alternative)
        pte_ok = pte_selected in acceptable if pte_selected else False
        pte_primary = pte_selected == primary_tool
        methods["pte"]["acceptable"] += int(pte_ok)
        methods["pte"]["primary"] += int(pte_primary)
        methods["pte"]["considered_tools"].append(len(exposed_tools))

        details.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "language": item["language"],
                "primary_intent": item["primary_intent"],
                "primary_tool": primary_tool,
                "acceptable_tools": item["acceptable_tools"],
                "detected_intent": intent_result.primary_intent,
                "intent_confidence": round(intent_result.confidence, 4),
                "static": {
                    "selected_tool": static_selected,
                    "correct": static_ok,
                    "primary_match": static_primary,
                    "considered_tools": len(ALL_TOOL_NAMES),
                    "top_candidates": static_top,
                },
                "keyword": {
                    "selected_tool": keyword_selected,
                    "correct": keyword_ok,
                    "primary_match": keyword_primary,
                    "considered_tools": len(keyword_top) if keyword_top else len(ALL_TOOL_NAMES),
                    "top_candidates": keyword_top,
                },
                "pte": {
                    "selected_tool": pte_selected,
                    "correct": pte_ok,
                    "primary_match": pte_primary,
                    "considered_tools": len(exposed_tools),
                    "exposed_tools": exposed_tools,
                    "recommended_tools": sorted(recommended),
                    "top_candidates": pte_top,
                },
            }
        )

    total = len(items)
    summary = {}
    for method_name, stats in methods.items():
        avg_considered = sum(stats["considered_tools"]) / total if total else 0.0
        summary[method_name] = {
            "acceptable_accuracy": round(stats["acceptable"] / total * 100, 2) if total else 0.0,
            "primary_accuracy": round(stats["primary"] / total * 100, 2) if total else 0.0,
            "n_correct": stats["acceptable"],
            "n_primary_match": stats["primary"],
            "n_total": total,
            "avg_considered_tools": round(avg_considered, 2),
        }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="WeClaw exp10 tool-selection benchmark")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET) if DEFAULT_DATASET.exists() else None,
        help="Path to weclaw_tool_selection_500 JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "evaluate"],
        default="evaluate",
        help="dry_run only validates dataset coverage; evaluate runs Static/Keyword/PTE baselines.",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path.")
    args = parser.parse_args()

    dataset_name, version, items = load_dataset_items(args.dataset, BUILTIN_TOOL_SELECTION_SAMPLE)
    if not items:
        raise RuntimeError("No tool-selection items found.")

    coverage = {
        "n_items": len(items),
        "languages": count_by_key(items, "language"),
        "primary_intents": count_by_key(items, "primary_intent"),
        "sub_intents": count_by_key(items, "sub_intent"),
    }

    if args.mode == "dry_run":
        summary = {
            **coverage,
            "status": "dataset_validated_only",
            "implemented_methods": ["static", "keyword", "pte"],
        }
        details: list[dict[str, Any]] = []
    else:
        baseline_summary, details = evaluate_methods(items)
        summary = {
            **coverage,
            "status": "evaluated_with_real_heuristic_baselines",
            "implemented_methods": ["static", "keyword", "pte"],
            "methods": baseline_summary,
        }

    payload = {
        "experiment": "exp10_tool_selection_500",
        "dataset": dataset_name,
        "dataset_version": version,
        "mode": args.mode,
        "summary": summary,
        "details": details,
        "next_step": "Add LLM-based Static / PTE-FD / ITR / Hybrid baselines on top of the same dataset.",
    }
    write_json(args.output, payload)
    print(f"Saved output file: {Path(args.output).name}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
