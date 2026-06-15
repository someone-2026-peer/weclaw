#!/usr/bin/env python3
"""Run all benchmark experiments sequentially."""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent

def main():
    run_id = os.getenv("WECLAW_BENCHMARK_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env["WECLAW_BENCHMARK_RUN_ID"] = run_id
    output_dir = BENCHMARKS_DIR / "raw_data" / "runs" / run_id

    scripts = [
        ("Experiment 1: PTE-FD Ablation", "exp1_pte_ablation.py"),
        ("Experiment 2: EBEAC Cost", "exp2_ebeac_cost.py"),
        ("Experiment 3: RCR Pipeline", "exp3_rcr_pipeline.py"),
        ("Experiment 4: ITR Faithful Reproduction", "exp4_itr_reproduction.py"),
        ("Experiment 5: EBEAC Recall Curve", "exp5_recall_curve.py"),
        ("Experiment 6: Multi-Step Task Evaluation", "exp6_multistep.py"),
        ("Experiment 7: ToolBench-lite Adapter Smoke Test", "exp7_toolbench_lite.py"),
    ]

    for name, script in scripts:
        print(f"\n{'='*60}")
        print(f"Running {name}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(BENCHMARKS_DIR / script)],
            cwd=str(BENCHMARKS_DIR.parent),
            env=env,
        )
        if result.returncode != 0:
            print(f"ERROR: {name} failed with exit code {result.returncode}")
            sys.exit(1)
        print(f"{name} completed successfully.")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Run ID: {run_id}")
    print(f"Results in: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
