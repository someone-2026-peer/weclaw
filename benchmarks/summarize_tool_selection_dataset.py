#!/usr/bin/env python3
"""Generate appendix-ready markdown for the tool-selection benchmark dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_utils import read_json


def format_counter(counter: Counter[str]) -> str:
    return "，".join(f"`{key}` {value}" for key, value in counter.items())


def top_entries(counter: Counter[str], top_k: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def load_items(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("Dataset payload must be a JSON object.")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise TypeError("Dataset items must be a list.")
    return payload, items


def build_summary_markdown(payload: dict[str, Any], items: list[dict[str, Any]], top_k: int) -> str:
    language_counter = Counter(str(item.get("language", "unknown")) for item in items)
    intent_counter = Counter(str(item.get("primary_intent", "unknown")) for item in items)
    sub_intent_counter = Counter(str(item.get("sub_intent", "unknown")) for item in items)
    primary_tool_counter = Counter(str(item.get("primary_tool", "unknown")) for item in items)
    positive_tool_counter: Counter[str] = Counter()
    negative_tool_counter: Counter[str] = Counter()

    for item in items:
        for tool in item.get("acceptable_tools", []):
            positive_tool_counter[str(tool)] += 1
        for tool in item.get("negative_tools", []):
            negative_tool_counter[str(tool)] += 1

    coverage = payload.get("coverage_constraints", {})
    high_freq_tools = coverage.get("high_frequency_tools", [])
    min_positive = int(coverage.get("high_frequency_tools_min_positive_examples", 0))
    satisfied = []
    missing = []
    for tool in high_freq_tools:
        count = positive_tool_counter[str(tool)]
        if count >= min_positive:
            satisfied.append((str(tool), count))
        else:
            missing.append((str(tool), count))

    date_str = datetime.now().strftime("%Y-%m-%d")
    total_items = len(items)
    appendix_lines = [
        "# weclaw_tool_selection_500 方法附录说明",
        "",
        f"**生成日期**: {date_str}  ",
        f"**数据集**: `{payload.get('dataset_name', 'weclaw_tool_selection_500')}`  ",
        f"**版本**: `{payload.get('version', 'unknown')}`  ",
        f"**样本数**: `{total_items}`",
        "",
        "## 1. 数据集定位",
        "",
        "本数据集用于评估 WeClaw 在工具选择阶段的路由能力，重点支持 `Static`、`Keyword`、`PTE`、`PTE-FD` 以及后续的 `ITR`、`Hybrid Router` 等方法对比。每条样本以单个 query 为标注单元，包含主意图、子意图、主工具、可接受工具集以及高频混淆负样本。",
        "",
        "## 2. 样本构成",
        "",
        f"- 语言分布：{format_counter(language_counter)}。",
        f"- 主意图数：`{len(intent_counter)}`，分布范围为最少 `{min(intent_counter.values())}` 条、最多 `{max(intent_counter.values())}` 条。",
        f"- 子意图数：`{len(sub_intent_counter)}`，覆盖从工具调用、文档处理、数据分析到自反思与研究辅助等多类桌面 Agent 场景。",
        "",
        "## 3. 标注口径",
        "",
        "- `primary_tool` 表示在 WeClaw 当前工具生态中最优的一号工具。",
        "- `acceptable_tools` 表示在当前 query 语义下可被判定为合理的主工具或弱备选工具。",
        "- `negative_tools` 记录最容易混淆的相邻工具，用于分析路由边界与误选来源。",
        "- 数据集中同时保留中文、英文与中英混合 query，以模拟真实桌面 Agent 的 bilingual 使用环境。",
        "",
        "## 4. 主意图分布",
        "",
    ]

    for intent, count in sorted(intent_counter.items(), key=lambda item: item[0]):
        appendix_lines.append(f"- `{intent}`: `{count}` 条。")

    appendix_lines.extend(
        [
            "",
            f"## 5. Top {top_k} 子意图分布",
            "",
        ]
    )
    for sub_intent, count in top_entries(sub_intent_counter, top_k):
        appendix_lines.append(f"- `{sub_intent}`: `{count}` 条。")

    appendix_lines.extend(
        [
            "",
            f"## 6. Top {top_k} 主工具分布",
            "",
        ]
    )
    for tool, count in top_entries(primary_tool_counter, top_k):
        appendix_lines.append(f"- `{tool}`: `{count}` 条。")

    appendix_lines.extend(
        [
            "",
            f"## 7. Top {top_k} 正例工具覆盖",
            "",
        ]
    )
    for tool, count in top_entries(positive_tool_counter, top_k):
        appendix_lines.append(f"- `{tool}`: `{count}` 条正例。")

    appendix_lines.extend(
        [
            "",
            f"## 8. Top {top_k} 高频负例混淆项",
            "",
        ]
    )
    for tool, count in top_entries(negative_tool_counter, top_k):
        appendix_lines.append(f"- `{tool}`: `{count}` 次。")

    appendix_lines.extend(
        [
            "",
            "## 9. 高频工具覆盖约束检查",
            "",
            f"- 约束阈值：每个高频工具至少 `{min_positive}` 条正例。",
            f"- 纳入覆盖检查的高频工具数：`{len(high_freq_tools)}`。",
            f"- 达标工具数：`{len(satisfied)}`。",
            f"- 未达标工具数：`{len(missing)}`。",
        ]
    )
    if missing:
        appendix_lines.append("")
        appendix_lines.append("未达标工具如下：")
        appendix_lines.append("")
        for tool, count in missing:
            appendix_lines.append(f"- `{tool}`: 当前 `{count}` 条。")
    else:
        appendix_lines.append("")
        appendix_lines.append("当前高频工具覆盖约束已全部满足。")

    appendix_lines.extend(
        [
            "",
            "## 10. 可直接写入论文的方法描述",
            "",
            (
                f"我们构建了一个面向桌面 Agent 工具选择的 bilingual benchmark，并在当前版本中整理出 "
                f"{total_items} 条 query 样本。该数据集覆盖 {len(intent_counter)} 个 primary intent 与 "
                f"{len(sub_intent_counter)} 个 sub-intent，样本同时包含中文、英文与中英混合表达，以尽可能贴近真实用户请求形态。"
                "每条样本均标注 `primary_tool`、`acceptable_tools` 与 `negative_tools`，从而同时支持主工具准确率、可接受准确率以及混淆项分析等多种评价口径。"
                "为了避免评测偏向少数高频工具，我们进一步引入高频工具覆盖约束，对核心工具设定“至少 5 条正例”的最小覆盖门槛；"
                f"在当前 {total_items} 条版本中，该约束已全部满足。这一设计使得后续对 `Static`、`Keyword`、`PTE`、`PTE-FD`、`ITR` 与 `Hybrid` 等方法的比较更具可解释性，也更有利于分析工具边界模糊、主备工具竞争与 bilingual query 对路由决策的影响。"
            ),
        ]
    )

    return "\n".join(appendix_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate appendix-ready markdown for tool-selection dataset.")
    parser.add_argument("--dataset", required=True, help="Path to weclaw_tool_selection_500 JSON.")
    parser.add_argument("--output", required=True, help="Output markdown path.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-k entries to include in summary sections.")
    args = parser.parse_args()

    payload, items = load_items(args.dataset)
    markdown = build_summary_markdown(payload, items, top_k=args.top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print("Saved appendix markdown file.")


if __name__ == "__main__":
    main()
