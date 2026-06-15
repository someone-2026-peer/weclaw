"""Shared helpers for benchmark experiment scripts."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def get_output_dir(script_file: str) -> Path:
    """Return the benchmark output directory.

    Defaults to the historical ``raw_data`` directory for backward
    compatibility. To keep repeated full reruns separate, set either:

    - ``WECLAW_BENCHMARK_OUTPUT_DIR``: explicit output directory
    - ``WECLAW_BENCHMARK_RUN_ID``: creates ``raw_data/runs/<run_id>``
    """
    base_dir = Path(script_file).parent / "raw_data"

    explicit_dir = os.getenv("WECLAW_BENCHMARK_OUTPUT_DIR", "").strip()
    if explicit_dir:
        output_dir = Path(explicit_dir)
        if not output_dir.is_absolute():
            output_dir = Path(script_file).parent / output_dir
    else:
        run_id = os.getenv("WECLAW_BENCHMARK_RUN_ID", "").strip()
        if run_id.lower() == "auto":
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / "runs" / run_id if run_id else base_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
