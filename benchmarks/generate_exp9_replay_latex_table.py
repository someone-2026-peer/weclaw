#!/usr/bin/env python3
"""Generate LaTeX table drafts for exp9 real replay runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_utils import read_json

RAW_DATA_DIR = Path(__file__).parent / "raw_data"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def infer_tag(path: Path) -> str:
    stem = path.stem
    prefix = "exp9_ebeac_log_replay_"
    suffix = "_results"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def collect_realdb_results() -> list[Path]:
    files = sorted(RAW_DATA_DIR.glob("exp9_ebeac_log_replay*_results.json"))
    return [path for path in files if "realdb" in path.name]


def pct(value: Any) -> str:
    return f"{float(value):.1f}\\%"


def num(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def build_snapshot_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        payload = read_json(path)
        summary = payload.get("summary", {})
        exp_store = summary.get("baselines", {}).get("experience_store_memory", {})
        rows.append(
            {
                "tag": infer_tag(path),
                "n_train": summary.get("n_train", 0),
                "n_test": summary.get("n_test", 0),
                "r1": exp_store.get("recall_at_1", 0.0),
                "r3": exp_store.get("recall_at_3", 0.0),
                "precision": exp_store.get("precision", 0.0),
                "coverage": exp_store.get("coverage_rate", 0.0),
                "prevention": exp_store.get("prevention_rate", 0.0),
                "fp": exp_store.get("false_positive_rate", 0.0),
                "latency": exp_store.get("avg_latency_ms", 0.0),
            }
        )
    return rows


def build_snapshot_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Preliminary production-snapshot replay validation across available real \texttt{ExperienceStore} snapshots. Split denotes \texttt{train/test}. \emph{Cov.}: coverage rate. \emph{Prev.}: prevention rate. \emph{FP}: false-positive rate.}",
        r"\label{tab:exp9_real_snapshot_runs}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Snapshot} & \textbf{Split} & \textbf{R@1} & \textbf{R@3} & \textbf{Prec.} & \textbf{Cov.} & \textbf{Prev.} & \textbf{FP} & \textbf{Lat. (ms)} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " ".join(
                [
                    rf"\texttt{{{latex_escape(row['tag'])}}}",
                    f"& {row['n_train']}/{row['n_test']}",
                    f"& {pct(row['r1'])}",
                    f"& {pct(row['r3'])}",
                    f"& {pct(row['precision'])}",
                    f"& {pct(row['coverage'])}",
                    f"& {pct(row['prevention'])}",
                    f"& {pct(row['fp'])}",
                    rf"& {num(row['latency'])} \\",
                ]
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_latest_baseline_table(path: Path) -> str:
    payload = read_json(path)
    baselines = payload.get("summary", {}).get("baselines", {})
    label_tag = latex_escape(infer_tag(path))
    method_order = [
        ("no_memory", "No memory"),
        ("experience_store_memory", r"\textbf{ExperienceStore}"),
        ("keyword_memory", "Keyword"),
        ("similarity_memory", "Similarity"),
        ("hybrid_memory", "Hybrid"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Baseline comparison on the latest real-snapshot replay validation run (\texttt{{{label_tag}}}). \emph{{Cov.}}: coverage rate. \emph{{Prev.}}: prevention rate. \emph{{FP}}: false-positive rate. \emph{{Miss}}: miss rate.}}",
        r"\label{tab:exp9_real_snapshot_baselines}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{R@1} & \textbf{R@3} & \textbf{Prec.} & \textbf{Cov.} & \textbf{Prev.} & \textbf{FP} & \textbf{Miss} & \textbf{Lat. (ms)} \\",
        r"\midrule",
    ]
    for key, display in method_order:
        row = baselines.get(key, {})
        lines.append(
            " ".join(
                [
                    display,
                    f"& {pct(row.get('recall_at_1', 0.0))}",
                    f"& {pct(row.get('recall_at_3', 0.0))}",
                    f"& {pct(row.get('precision', 0.0))}",
                    f"& {pct(row.get('coverage_rate', 0.0))}",
                    f"& {pct(row.get('prevention_rate', 0.0))}",
                    f"& {pct(row.get('false_positive_rate', 0.0))}",
                    f"& {pct(row.get('miss_rate', 0.0))}",
                    rf"& {num(row.get('avg_latency_ms', 0.0))} \\",
                ]
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table drafts for exp9 real replay runs.")
    parser.add_argument(
        "--output",
        default=str(RAW_DATA_DIR / "exp9_real_replay_table_draft.tex"),
        help="Output LaTeX file path.",
    )
    args = parser.parse_args()

    result_files = collect_realdb_results()
    if not result_files:
        raise RuntimeError("No exp9 real replay result files found.")

    snapshot_rows = build_snapshot_rows(result_files)
    latest = snapshot_rows[0]["tag"]
    latest_file = next(path for path in result_files if infer_tag(path) == latest)

    content = "\n\n".join(
        [
            f"% Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "% Draft tables for direct inclusion in the paper body or appendix.",
            build_snapshot_table(snapshot_rows),
            build_latest_baseline_table(latest_file),
        ]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved LaTeX table draft: {output_path.name}")


if __name__ == "__main__":
    main()
