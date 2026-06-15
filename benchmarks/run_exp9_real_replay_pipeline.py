#!/usr/bin/env python3
"""Run the real-snapshot exp9 replay pipeline end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_utils import PLANNED_ARTIFACTS_DIR, get_default_output_path

BENCHMARKS_DIR = Path(__file__).parent
EXPORTER = BENCHMARKS_DIR / "export_experience_store_replay.py"
EXP9 = BENCHMARKS_DIR / "exp9_ebeac_log_replay.py"
PROJECT_ROOT = BENCHMARKS_DIR.parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_tag(custom_tag: str | None) -> str:
    if custom_tag:
        return custom_tag
    return datetime.now().strftime("realdb_%Y%m%d_%H%M%S")


def run_command(args: list[str]) -> None:
    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def build_markdown_summary(
    dataset_path: Path,
    results_path: Path,
    tag: str,
) -> str:
    dataset_payload = read_json(dataset_path)
    results_payload = read_json(results_path)
    export_summary = dataset_payload.get("export_summary", {})
    summary = results_payload.get("summary", {})
    baselines = summary.get("baselines", {})
    source_dbs = export_summary.get("source_dbs", [])
    source_db_lines = "\n".join(f"- `{path}`" for path in source_dbs) or "- `N/A`"

    def baseline_line(name: str) -> str:
        row = baselines.get(name, {})
        return (
            f"- `{name}`: recall@1={row.get('recall_at_1', 0.0)}, "
            f"recall@3={row.get('recall_at_3', 0.0)}, "
            f"precision={row.get('precision', 0.0)}, "
            f"coverage_rate={row.get('coverage_rate', 0.0)}, "
            f"prevention_rate={row.get('prevention_rate', 0.0)}, "
            f"false_positive_rate={row.get('false_positive_rate', 0.0)}, "
            f"miss_rate={row.get('miss_rate', 0.0)}, "
            f"avg_latency_ms={row.get('avg_latency_ms', 0.0)}"
        )

    return "\n".join(
        [
            f"# exp9 真实快照回放摘要（{tag}）",
            "",
            f"**日期**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**数据集**: `{dataset_path}`",
            f"**结果文件**: `{results_path}`",
            "",
            "## 导出概况",
            "",
            f"- 快照数量: {export_summary.get('source_db_count', 0)}",
            f"- 去重前记录数: {export_summary.get('rows_before_dedup', 0)}",
            f"- 去重后记录数: {export_summary.get('rows_after_dedup', 0)}",
            f"- 训练比例: {export_summary.get('train_ratio', 0.0)}",
            "",
            "### 快照来源",
            "",
            source_db_lines,
            "",
            "## 回放概况",
            "",
            f"- n_train: {summary.get('n_train', 0)}",
            f"- n_test: {summary.get('n_test', 0)}",
            f"- n_test_errors: {summary.get('n_test_errors', 0)}",
            "",
            "## Baseline 指标",
            "",
            baseline_line("no_memory"),
            baseline_line("experience_store_memory"),
            baseline_line("keyword_memory"),
            baseline_line("similarity_memory"),
            baseline_line("hybrid_memory"),
            "",
            "## 结论提示",
            "",
            "- 当前摘要适合作为阶段性实验记录和论文补充材料输入。",
            "- 若后续导入更多生产快照，可直接复用同一脚本重跑，并对比 prevention_rate 与 false_positive_rate 的变化。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp9 real replay pipeline from one or more ExperienceStore snapshots.")
    parser.add_argument("--db", action="append", default=[], help="Path to ExperienceStore SQLite database. Repeatable.")
    parser.add_argument("--db-dir", help="Directory containing one or more ExperienceStore *.db snapshots.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train split ratio.")
    parser.add_argument("--tag", help="Optional tag used in generated filenames.")
    args = parser.parse_args()

    if not args.db and not args.db_dir:
        raise ValueError("Please provide at least one --db or a --db-dir.")

    tag = build_tag(args.tag)
    dataset_path = PLANNED_ARTIFACTS_DIR / f"weclaw_ebeac_log_replay_{tag}.json"
    results_path = get_default_output_path(__file__, f"exp9_ebeac_log_replay_{tag}_results.json")
    summary_path = get_default_output_path(__file__, f"exp9_ebeac_log_replay_{tag}_summary.md")

    export_cmd = [sys.executable, str(EXPORTER), "--output", str(dataset_path), "--train-ratio", str(args.train_ratio)]
    for db in args.db:
        export_cmd.extend(["--db", db])
    if args.db_dir:
        export_cmd.extend(["--db-dir", args.db_dir])
    run_command(export_cmd)

    run_command(
        [
            sys.executable,
            str(EXP9),
            "--dataset",
            str(dataset_path),
            "--output",
            str(results_path),
        ]
    )

    write_text(summary_path, build_markdown_summary(dataset_path, results_path, tag))
    print(f"Dataset file: {dataset_path.name}")
    print(f"Results file: {results_path.name}")
    print(f"Summary file: {summary_path.name}")


if __name__ == "__main__":
    main()
