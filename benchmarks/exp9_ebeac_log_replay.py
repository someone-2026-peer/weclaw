#!/usr/bin/env python3
"""Experiment 9: EBEAC log replay benchmark.

This version upgrades the earlier scaffold into a reproducible offline replay
benchmark with deterministic baselines:

- no_memory
- keyword_memory
- similarity_memory
- hybrid_memory

The implementation is still benchmark-side, but it now computes real top-k
retrieval statistics from the replay data instead of placeholder hit flags.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.experience_store import ExperienceStore

from dataset_utils import count_by_key, get_default_output_path, load_dataset_items, write_json

OUTPUT_PATH = get_default_output_path(__file__, "exp9_ebeac_log_replay_results.json")


class SqliteOnlyExperienceStore(ExperienceStore):
    """SQLite-only benchmark adapter to keep exp9 reproducible offline."""

    @property
    def vector_store(self):  # type: ignore[override]
        return None

    def _vector_recall(self, query: str, top_k: int):  # type: ignore[override]
        return None


BUILTIN_LOG_REPLAY_SAMPLE: list[dict[str, Any]] = [
    {
        "event_id": "e001",
        "timestamp": "2026-06-01T10:00:00",
        "session_id": "s01",
        "tool_name": "stock_query",
        "action": "query",
        "status": "error",
        "error_type": "empty_result",
        "fix_pattern": "fallback_to_null_safe_parse",
        "split": "train",
        "metadata": {"query": "查询停牌股票价格"},
    },
    {
        "event_id": "e002",
        "timestamp": "2026-06-01T10:01:00",
        "session_id": "s01",
        "tool_name": "stock_query",
        "action": "query",
        "status": "success",
        "error_type": None,
        "fix_pattern": "fallback_to_null_safe_parse",
        "split": "train",
        "metadata": {"query": "查询停牌股票价格"},
    },
    {
        "event_id": "e003",
        "timestamp": "2026-06-01T11:00:00",
        "session_id": "s02",
        "tool_name": "browser_use",
        "action": "execute",
        "status": "error",
        "error_type": "structured_output_unsupported",
        "fix_pattern": "disable_forced_schema_output",
        "split": "train",
        "metadata": {"query": "执行浏览器自动化"},
    },
    {
        "event_id": "e004",
        "timestamp": "2026-06-02T09:00:00",
        "session_id": "s03",
        "tool_name": "stock_query",
        "action": "query",
        "status": "error",
        "error_type": "empty_result",
        "fix_pattern": "fallback_to_null_safe_parse",
        "split": "test",
        "metadata": {"query": "查询停牌股票实时价格"},
    },
    {
        "event_id": "e005",
        "timestamp": "2026-06-02T09:10:00",
        "session_id": "s04",
        "tool_name": "browser_use",
        "action": "execute",
        "status": "error",
        "error_type": "structured_output_unsupported",
        "fix_pattern": "disable_forced_schema_output",
        "split": "test",
        "metadata": {"query": "让浏览器自动打开并提取页面"},
    },
    {
        "event_id": "e006",
        "timestamp": "2026-06-02T09:20:00",
        "session_id": "s05",
        "tool_name": "ocr",
        "action": "recognize",
        "status": "error",
        "error_type": "file_not_found",
        "fix_pattern": "check_input_path_first",
        "split": "test",
        "metadata": {"query": "识别图片文字"},
    },
]
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("_", " ").replace("-", " ")).strip()


def tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", normalize_text(text)) if token}


def event_text(event: dict[str, Any]) -> str:
    metadata = event.get("metadata", {}) or {}
    return " ".join(
        [
            str(event.get("tool_name", "")),
            str(event.get("action", "")),
            str(event.get("error_type", "")),
            str(event.get("fix_pattern", "")),
            str(metadata.get("query", "")),
        ]
    )


def lexical_score(train_event: dict[str, Any], test_event: dict[str, Any]) -> float:
    score = 0.0
    if train_event.get("tool_name") == test_event.get("tool_name"):
        score += 3.0
    if train_event.get("error_type") == test_event.get("error_type"):
        score += 4.0
    if train_event.get("action") == test_event.get("action"):
        score += 1.0
    overlap = tokenize(event_text(train_event)) & tokenize(event_text(test_event))
    score += len(overlap) * 0.5
    return score


def similarity_score(train_event: dict[str, Any], test_event: dict[str, Any]) -> float:
    train_tokens = tokenize(event_text(train_event))
    test_tokens = tokenize(event_text(test_event))
    if not train_tokens or not test_tokens:
        return 0.0
    intersection = len(train_tokens & test_tokens)
    union = len(train_tokens | test_tokens)
    jaccard = intersection / union if union else 0.0
    tool_bonus = 0.15 if train_event.get("tool_name") == test_event.get("tool_name") else 0.0
    error_bonus = 0.2 if train_event.get("error_type") == test_event.get("error_type") else 0.0
    return jaccard + tool_bonus + error_bonus


def hybrid_score(train_event: dict[str, Any], test_event: dict[str, Any]) -> float:
    return lexical_score(train_event, test_event) + similarity_score(train_event, test_event)


def build_recall_query(event: dict[str, Any]) -> str:
    return " ".join(
        [
            str(event.get("tool_name", "")),
            str(event.get("error_type", "")),
        ]
    ).strip()


def init_stats() -> dict[str, Any]:
    return {"top1": 0, "top3": 0, "predictions": 0, "latency_ms": []}


def summarize_stats(stats: dict[str, Any], n_test_errors: int) -> dict[str, Any]:
    def safe_rate(value: float) -> float:
        return round(value / n_test_errors * 100, 2) if n_test_errors else 0.0

    false_positive = max(0, stats["predictions"] - stats["top3"])
    return {
        "recall_at_1": safe_rate(stats["top1"]),
        "recall_at_3": safe_rate(stats["top3"]),
        "precision": round(stats["top1"] / stats["predictions"] * 100, 2) if stats["predictions"] else 0.0,
        "coverage_rate": safe_rate(stats["predictions"]),
        "prevention_rate": safe_rate(stats["top3"]),
        "false_positive_rate": safe_rate(false_positive),
        "miss_rate": safe_rate(max(0, n_test_errors - stats["top3"])),
        "avg_latency_ms": round(sum(stats["latency_ms"]) / len(stats["latency_ms"]), 4) if stats["latency_ms"] else 0.0,
    }


async def seed_experience_store(store: ExperienceStore, train_events: list[dict[str, Any]]) -> None:
    for event in train_events:
        if event.get("status") != "error" or not event.get("fix_pattern"):
            continue
        tool_name = str(event.get("tool_name", ""))
        action = str(event.get("action", ""))
        error_type = str(event.get("error_type") or "unknown_error")
        query = str((event.get("metadata", {}) or {}).get("query", ""))
        fix_pattern = str(event.get("fix_pattern", ""))
        trigger = f"{tool_name} {action} {error_type} {query}".strip()
        diagnosis = f"{tool_name} {error_type} {query}".strip()
        abstract_pattern = f"{tool_name}:{error_type}:{fix_pattern}"
        await store.record(
            trigger=trigger,
            diagnosis=diagnosis,
            fix_summary=fix_pattern,
            abstract_pattern=abstract_pattern,
            outcome="success",
            source_type="tool_retry",
            session_id=str(event.get("session_id", "")),
            tools=[tool_name] if tool_name else [],
        )


def evaluate_experience_store_memory(
    train_events: list[dict[str, Any]],
    test_events: list[dict[str, Any]],
    min_similarity: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_test_errors = sum(1 for event in test_events if event.get("status") == "error")
    stats = init_stats()
    details = []

    with tempfile.TemporaryDirectory(prefix="weclaw_exp9_") as tmpdir:
        tmp_path = Path(tmpdir)
        store = SqliteOnlyExperienceStore(
            db_path=str(tmp_path / "experiences.db"),
            vector_db_dir=str(tmp_path / "experience_vectors"),
        )
        try:
            asyncio.run(seed_experience_store(store, train_events))

            for event in test_events:
                if event.get("status") != "error":
                    continue
                started = time.perf_counter()
                recalled = store.recall(build_recall_query(event), top_k=3, min_similarity=min_similarity)
                stats["latency_ms"].append((time.perf_counter() - started) * 1000)
                predictions = [exp.fix_summary for exp in recalled if exp.fix_summary][:3]
                ground_truth = event.get("fix_pattern")
                if predictions:
                    stats["predictions"] += 1
                    stats["top1"] += int(predictions[0] == ground_truth)
                    stats["top3"] += int(ground_truth in predictions)
                details.append(
                    {
                        "event_id": event["event_id"],
                        "tool_name": event["tool_name"],
                        "error_type": event["error_type"],
                        "ground_truth_fix_pattern": ground_truth,
                        "experience_store_top3": predictions,
                        "preventable": ground_truth in predictions,
                        "recalled_patterns": [exp.abstract_pattern for exp in recalled],
                    }
                )
        finally:
            store.close()
    return summarize_stats(stats, n_test_errors), details


def evaluate_replay(events: list[dict[str, Any]]) -> dict[str, Any]:
    train_events = [event for event in events if event.get("split") == "train"]
    test_events = [event for event in events if event.get("split") == "test"]
    n_test_errors = sum(1 for event in test_events if event.get("status") == "error")
    baselines = {
        "no_memory": init_stats(),
        "keyword_memory": init_stats(),
        "similarity_memory": init_stats(),
        "hybrid_memory": init_stats(),
    }
    details = []

    for event in test_events:
        if event.get("status") != "error":
            continue
        ground_truth = event.get("fix_pattern")

        keyword_ranked = []
        sim_ranked = []
        hybrid_ranked = []

        started = time.perf_counter()
        for candidate in train_events:
            if candidate.get("status") != "error" or not candidate.get("fix_pattern"):
                continue
            keyword_ranked.append((candidate, lexical_score(candidate, event)))
        baselines["keyword_memory"]["latency_ms"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        for candidate in train_events:
            if candidate.get("status") != "error" or not candidate.get("fix_pattern"):
                continue
            sim_ranked.append((candidate, similarity_score(candidate, event)))
        baselines["similarity_memory"]["latency_ms"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        for candidate in train_events:
            if candidate.get("status") != "error" or not candidate.get("fix_pattern"):
                continue
            hybrid_ranked.append((candidate, hybrid_score(candidate, event)))
        baselines["hybrid_memory"]["latency_ms"].append((time.perf_counter() - started) * 1000)

        keyword_ranked.sort(key=lambda item: (-item[1], item[0]["event_id"]))
        sim_ranked.sort(key=lambda item: (-item[1], item[0]["event_id"]))
        hybrid_ranked.sort(key=lambda item: (-item[1], item[0]["event_id"]))

        keyword_top = [candidate["fix_pattern"] for candidate, score in keyword_ranked if score > 0][:3]
        sim_top = [candidate["fix_pattern"] for candidate, score in sim_ranked if score > 0][:3]
        hybrid_top = [candidate["fix_pattern"] for candidate, score in hybrid_ranked if score > 0][:3]

        for baseline_name, top_predictions in [
            ("keyword_memory", keyword_top),
            ("similarity_memory", sim_top),
            ("hybrid_memory", hybrid_top),
        ]:
            if top_predictions:
                baselines[baseline_name]["predictions"] += 1
                baselines[baseline_name]["top1"] += int(top_predictions[0] == ground_truth)
                baselines[baseline_name]["top3"] += int(ground_truth in top_predictions)

        details.append(
            {
                "event_id": event["event_id"],
                "tool_name": event["tool_name"],
                "error_type": event["error_type"],
                "ground_truth_fix_pattern": ground_truth,
                "keyword_top3": keyword_top,
                "similarity_top3": sim_top,
                "hybrid_top3": hybrid_top,
                "keyword_preventable": ground_truth in keyword_top,
                "similarity_preventable": ground_truth in sim_top,
                "hybrid_preventable": ground_truth in hybrid_top,
            }
        )

    experience_store_summary, experience_store_details = evaluate_experience_store_memory(
        train_events,
        test_events,
        min_similarity=0.6,
    )

    summary_baselines = {
        "no_memory": summarize_stats(baselines["no_memory"], n_test_errors),
        "experience_store_memory": experience_store_summary,
    }
    for baseline_name in ["keyword_memory", "similarity_memory", "hybrid_memory"]:
        summary_baselines[baseline_name] = summarize_stats(baselines[baseline_name], n_test_errors)

    return {
        "summary": {
            "n_train": len(train_events),
            "n_test": len(test_events),
            "n_test_errors": n_test_errors,
            "tools": count_by_key(events, "tool_name"),
            "error_types": count_by_key([event for event in events if event.get("error_type")], "error_type"),
            "baselines": summary_baselines,
            "has_real_experience_store_integration": True,
            "implemented_methods": ["no_memory", "experience_store_memory", "keyword_memory", "similarity_memory", "hybrid_memory"],
        },
        "details": {
            "lexical_replay": details,
            "experience_store_memory": experience_store_details,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WeClaw exp9 EBEAC log replay benchmark")
    parser.add_argument("--dataset", help="Optional path to weclaw_ebeac_log_replay style JSON.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path.")
    args = parser.parse_args()

    dataset_name, version, events = load_dataset_items(args.dataset, BUILTIN_LOG_REPLAY_SAMPLE)
    if not events:
        raise RuntimeError("No log replay events found.")

    evaluation = evaluate_replay(events)
    payload = {
        "experiment": "exp9_ebeac_log_replay",
        "status": "evaluated_with_experience_store_replay",
        "dataset": dataset_name,
        "dataset_version": version,
        **evaluation,
        "next_step": "Export larger time-split replay sets from real experiences.db snapshots, compare multi-snapshot replay, and stabilize prevention-rate evaluation.",
    }
    write_json(args.output, payload)
    print(f"Saved output file: {Path(args.output).name}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
