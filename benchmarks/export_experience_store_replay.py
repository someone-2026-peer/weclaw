#!/usr/bin/env python3
"""Export ExperienceStore SQLite records into a replay benchmark JSON."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from dataset_utils import write_json


def parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [str(item) for item in data]
    except Exception:
        pass
    return []


def derive_action(trigger: str, diagnosis: str) -> str:
    text = f"{trigger} {diagnosis}"
    match = re.search(r"([a-zA-Z_]+)\.([a-zA-Z_]+)", text)
    if match:
        return match.group(2)
    if "retry" in text.lower() or "重试" in text:
        return "retry"
    return "execute"


def derive_error_type(trigger: str, diagnosis: str, abstract_pattern: str) -> str:
    text = f"{trigger} {diagnosis} {abstract_pattern}".lower()
    rules = [
        ("empty_result", ["empty_result", "空结果", "空字符串"]),
        ("timeout_error", ["timeout", "timed out", "超时"]),
        ("permission_error", ["permission", "权限", "denied"]),
        ("file_not_found", ["file_not_found", "找不到", "not found"]),
        ("structured_output_unsupported", ["structured_output_unsupported", "schema", "结构化输出"]),
        ("unsupported_format", ["unsupported format", "格式", "convert"]),
        ("audio_device_unavailable", ["audio device", "麦克风", "device unavailable"]),
        ("login_redirect_loop", ["redirect", "login", "登录"]),
        ("low_confidence_retrieval", ["low confidence", "低置信度", "retrieval"]),
        ("parse_error", ["parse", "解析", "decode", "encoding"]),
        ("network_error", ["network", "connection", "连接"]),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "generic_tool_retry"


def derive_fix_pattern(fix_summary: str, abstract_pattern: str) -> str:
    if abstract_pattern.strip():
        return abstract_pattern.strip()
    return fix_summary.strip() or "unspecified_fix_pattern"


def normalize_source_tag(db_path: Path) -> str:
    raw = f"{db_path.parent.name}_{db_path.stem}"
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower() or "snapshot"


def rows_to_records(rows: list[tuple[Any, ...]], db_path: Path) -> list[dict[str, Any]]:
    source_tag = normalize_source_tag(db_path)
    valid_rows = []
    for row in rows:
        (
            exp_id,
            trigger,
            diagnosis,
            fix_summary,
            abstract_pattern,
            outcome,
            source_type,
            related_files,
            session_id,
            tool_names,
            hit_count,
            created_at,
            updated_at,
        ) = row
        tools = parse_json_list(tool_names)
        if not tools:
            continue
        valid_rows.append(
            {
                "source_db": str(db_path),
                "source_tag": source_tag,
                "id": exp_id,
                "trigger": trigger or "",
                "diagnosis": diagnosis or "",
                "fix_summary": fix_summary or "",
                "abstract_pattern": abstract_pattern or "",
                "outcome": outcome or "pending",
                "source_type": source_type or "manual",
                "related_files": parse_json_list(related_files),
                "session_id": session_id or f"exp_session_{exp_id}",
                "tool_name": tools[0],
                "tool_names": tools,
                "hit_count": hit_count or 0,
                "created_at": created_at or "",
                "updated_at": updated_at or "",
            }
        )
    return valid_rows


def dedup_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["tool_name"],
        row["trigger"],
        row["diagnosis"],
        row["fix_summary"],
        row["abstract_pattern"],
        row["created_at"],
    )


def export_rows(valid_rows: list[dict[str, Any]], train_ratio: float, source_dbs: list[Path], rows_before_dedup: int) -> dict[str, Any]:
    if len(valid_rows) < 2:
        raise ValueError("Not enough experience rows with tool_names to create a replay split.")

    valid_rows.sort(key=lambda row: (row["created_at"], row["source_tag"], row["id"]))
    split_idx = max(1, min(len(valid_rows) - 1, int(len(valid_rows) * train_ratio)))
    items = []
    for idx, row in enumerate(valid_rows):
        split = "train" if idx < split_idx else "test"
        items.append(
            {
                "event_id": f"expdb_{row['source_tag']}_{row['id']:05d}",
                "timestamp": row["created_at"],
                "session_id": row["session_id"],
                "tool_name": row["tool_name"],
                "action": derive_action(row["trigger"], row["diagnosis"]),
                "status": "error",
                "error_type": derive_error_type(row["trigger"], row["diagnosis"], row["abstract_pattern"]),
                "fix_pattern": derive_fix_pattern(row["fix_summary"], row["abstract_pattern"]),
                "split": split,
                "metadata": {
                    "query": row["trigger"],
                    "diagnosis": row["diagnosis"],
                    "source_type": row["source_type"],
                    "tool_names": row["tool_names"],
                    "related_files": row["related_files"],
                    "hit_count": row["hit_count"],
                    "source_db": row["source_db"],
                    "record_origin": "experience_store_export",
                },
            }
        )

    return {
        "dataset_name": "weclaw_ebeac_log_replay",
        "version": "0.3.0-real-db-export",
        "notes": (
            "Replay dataset exported from one or more ExperienceStore SQLite snapshots. "
            "Rows are deduplicated, sorted chronologically, and split into train/test partitions "
            "for exp9 replay evaluation."
        ),
        "export_summary": {
            "source_db_count": len(source_dbs),
            "source_dbs": [str(path) for path in source_dbs],
            "rows_before_dedup": rows_before_dedup,
            "rows_after_dedup": len(valid_rows),
            "train_ratio": train_ratio,
        },
        "items": items,
    }


def ensure_experiences_table(conn: sqlite3.Connection, db_path: Path) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'experiences'"
    ).fetchone()
    if row:
        return
    raise RuntimeError(
        "The provided SQLite file does not contain an 'experiences' table. "
        f"Path: {db_path}. This usually means you copied only the main .db file "
        "while the real data still lived in the SQLite WAL file, or you copied the wrong database. "
        "Please export a consistent snapshot first (for example via SQLite backup/VACUUM INTO), "
        "or copy experiences.db together with experiences.db-wal and experiences.db-shm after checkpoint."
    )


def load_rows_from_db(db_path: Path) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_experiences_table(conn, db_path)
        return conn.execute(
            """
            SELECT id, trigger, diagnosis, fix_summary, abstract_pattern,
                   outcome, source_type, related_files, session_id,
                   tool_names, hit_count, created_at, updated_at
            FROM experiences
            ORDER BY created_at ASC
            """
        ).fetchall()
    finally:
        conn.close()


def resolve_db_paths(explicit_dbs: list[str], db_dir: str | None) -> list[Path]:
    db_paths = [Path(path) for path in explicit_dbs]
    if db_dir:
        db_paths.extend(sorted(Path(db_dir).glob("*.db")))
    unique_paths = []
    seen = set()
    for path in db_paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiences.db into replay benchmark JSON.")
    parser.add_argument("--db", action="append", default=[], help="Path to ExperienceStore SQLite database. Repeatable.")
    parser.add_argument("--db-dir", help="Optional directory containing multiple SQLite snapshots (*.db).")
    parser.add_argument("--output", required=True, help="Path to output replay JSON.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train split ratio.")
    args = parser.parse_args()

    db_paths = resolve_db_paths(args.db, args.db_dir)
    if not db_paths:
        raise ValueError("Please provide at least one --db or a --db-dir containing *.db snapshots.")

    missing = [path for path in db_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Database not found: {missing[0]}")

    rows_before_dedup = 0
    seen_signatures: set[tuple[str, ...]] = set()
    merged_rows: list[dict[str, Any]] = []
    for db_path in db_paths:
        db_rows = rows_to_records(load_rows_from_db(db_path), db_path)
        rows_before_dedup += len(db_rows)
        for row in db_rows:
            signature = dedup_signature(row)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged_rows.append(row)

    payload = export_rows(merged_rows, train_ratio=args.train_ratio, source_dbs=db_paths, rows_before_dedup=rows_before_dedup)
    write_json(args.output, payload)
    print("Exported replay dataset from ExperienceStore.")


if __name__ == "__main__":
    main()
