#!/usr/bin/env python3
"""Summarize multiple real-snapshot exp9 replay runs into markdown."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_utils import read_json

RAW_DATA_DIR = Path(__file__).parent / "raw_data"


def collect_result_files(pattern: str) -> list[Path]:
    files = sorted(RAW_DATA_DIR.glob(pattern))
    return [path for path in files if "realdb" in path.name]


def safe_get(mapping: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def infer_tag(path: Path) -> str:
    stem = path.stem
    prefix = "exp9_ebeac_log_replay_"
    suffix = "_results"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def build_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    summary = payload.get("summary", {})
    baselines = summary.get("baselines", {})
    exp_store = baselines.get("experience_store_memory", {})
    keyword = baselines.get("keyword_memory", {})
    return {
        "tag": infer_tag(path),
        "file": path.name,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": payload.get("dataset_version", "unknown"),
        "n_train": summary.get("n_train", 0),
        "n_test": summary.get("n_test", 0),
        "n_test_errors": summary.get("n_test_errors", 0),
        "exp_recall1": exp_store.get("recall_at_1", 0.0),
        "exp_recall3": exp_store.get("recall_at_3", 0.0),
        "exp_precision": exp_store.get("precision", 0.0),
        "exp_coverage": exp_store.get("coverage_rate", 0.0),
        "exp_prevention": exp_store.get("prevention_rate", 0.0),
        "exp_false_positive": exp_store.get("false_positive_rate", 0.0),
        "exp_miss": exp_store.get("miss_rate", 0.0),
        "exp_latency": exp_store.get("avg_latency_ms", 0.0),
        "kw_prevention": keyword.get("prevention_rate", 0.0),
        "kw_false_positive": keyword.get("false_positive_rate", 0.0),
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# exp9 真实快照回放结果汇总",
        "",
        f"**日期**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**结果文件数量**: {len(rows)}",
        "",
        "本汇总用于对比多次真实 `experiences.db` 快照回放的关键指标，重点观察样本规模、`prevention_rate`、`false_positive_rate` 与 `coverage_rate` 的变化。",
        "",
        "## 结果列表",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "- 未找到任何 `exp9_ebeac_log_replay_*realdb*_results.json` 结果文件。",
                "",
            ]
        )
        return "\n".join(lines)

    for row in rows:
        lines.extend(
            [
                f"### {row['tag']}",
                "",
                f"- 结果文件: `{row['file']}`",
                f"- 最近修改时间: {row['mtime']}",
                f"- 数据集版本: `{row['dataset_version']}`",
                f"- 样本规模: n_train={row['n_train']}, n_test={row['n_test']}, n_test_errors={row['n_test_errors']}",
                (
                    f"- ExperienceStore: recall@1={row['exp_recall1']}, recall@3={row['exp_recall3']}, "
                    f"precision={row['exp_precision']}, coverage_rate={row['exp_coverage']}, "
                    f"prevention_rate={row['exp_prevention']}, false_positive_rate={row['exp_false_positive']}, "
                    f"miss_rate={row['exp_miss']}, avg_latency_ms={row['exp_latency']}"
                ),
                (
                    f"- Keyword 对照: prevention_rate={row['kw_prevention']}, "
                    f"false_positive_rate={row['kw_false_positive']}"
                ),
                "",
            ]
        )

    if len(rows) >= 2:
        latest = rows[0]
        earliest = rows[-1]
        lines.extend(
            [
                "## 趋势观察",
                "",
                f"- 最新结果为 `{latest['tag']}`，最早结果为 `{earliest['tag']}`。",
                (
                    f"- ExperienceStore prevention_rate 变化：{earliest['exp_prevention']} -> {latest['exp_prevention']}"
                ),
                (
                    f"- ExperienceStore false_positive_rate 变化：{earliest['exp_false_positive']} -> {latest['exp_false_positive']}"
                ),
                (
                    f"- ExperienceStore coverage_rate 变化：{earliest['exp_coverage']} -> {latest['exp_coverage']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 使用建议",
            "",
            "- 每次新增真实快照并重跑 `run_exp9_real_replay_pipeline.py` 后，再运行本汇总脚本。",
            "- 当样本规模明显扩大时，优先观察 `prevention_rate` 是否提升，以及 `false_positive_rate` 是否保持较低水平。",
            "- 若多次结果开始稳定，可将该汇总作为论文中的阶段性趋势证据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exp9 real replay result files.")
    parser.add_argument(
        "--pattern",
        default="exp9_ebeac_log_replay*_results.json",
        help="Glob pattern inside benchmarks/raw_data.",
    )
    parser.add_argument(
        "--output",
        default=str(RAW_DATA_DIR / "exp9_real_replay_runs_summary.md"),
        help="Output markdown path.",
    )
    args = parser.parse_args()

    files = collect_result_files(args.pattern)
    rows = [build_row(path) for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Saved summary file: {output_path.name}")


if __name__ == "__main__":
    main()
