"""Shared dataset helpers for planned benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_utils import get_output_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANNED_ARTIFACTS_DIR = Path(__file__).parent / "planned_artifacts"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_default_output_path(script_file: str, filename: str) -> Path:
    return get_output_dir(script_file) / filename


def load_dataset_items(path: str | Path | None, builtin_items: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    """Load dataset items from JSON or fall back to builtin samples."""
    if not path:
        return "builtin_sample", "0.0.0-builtin", builtin_items

    payload = read_json(path)
    if isinstance(payload, list):
        return Path(path).stem, "external-list", payload

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported dataset payload type: {type(payload)!r}")

    dataset_name = str(payload.get("dataset_name", Path(path).stem))
    version = str(payload.get("version", "external"))
    items = payload.get("items", payload.get("tasks", []))
    if not isinstance(items, list):
        raise TypeError("Dataset items must be a list")
    return dataset_name, version, items


def count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))
