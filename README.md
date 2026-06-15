# WeClaw Adaptive Runtime - Reproducibility Artifact

This repository is the anonymized reproducibility artifact for the paper
*"Adaptive Runtime for LLM Agents"* (under review). It contains the benchmark
harness, the raw per-run experimental data, and the core mechanism source
modules referenced in the paper (PTE-FD, EBEAC, RCR).

It is a **curated artifact**, not the full product: only the source files needed
to inspect and reproduce the reported experiments are included.

## Layout

```
src/core/        Core mechanism modules used by the benchmarks
  prompts.py             Intent detection + intent->tool mapping
  tool_exposure.py       PTE-FD: progressive tool exposure + failure-driven escalation
  experience_store.py    EBEAC: experience capture / retrieval store
  context_compressor.py  RCR: robust context pipeline
  ...                    (supporting modules: events, token_utils, redact, ...)
config/tools.json  Tool registry (backs the action/tool/category counts)
benchmarks/        Experiment scripts + raw_data/ (per-run JSON outputs)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in API keys for the providers you use
```

All experiments call OpenAI-compatible chat-completion endpoints with
`temperature=0.0`. See `.env.example` for the four supported back-ends.

## Experiment -> paper mapping

| Script | Paper artifact |
| --- | --- |
| `benchmarks/exp_pte_ablation_canonical.py` | **Table 4** (n=100 single-step ablation) and **Table 8** (cross-model generalization) - canonical source |
| `benchmarks/exp1_pte_ablation.py` | Historical PTE-FD ablation (methodology lineage) |
| `benchmarks/exp3_rcr_pipeline.py` | RCR context pipeline |
| `benchmarks/exp4_itr_reproduction.py`, `exp4b_itr_bge_m3.py` | ITR retrieval baseline reproduction |
| `benchmarks/exp2_ebeac_cost.py` | EBEAC cost analysis |
| `benchmarks/exp5_recall_curve.py` | EBEAC recall curve |
| `benchmarks/exp9_ebeac_log_replay.py` | EBEAC log-replay evaluation |
| `benchmarks/exp6_multistep.py`, `exp8_execution_level_multistep.py` | Multi-step execution |
| `benchmarks/exp7_toolbench_lite.py` | ToolBench-lite adapter |
| `benchmarks/exp10_tool_selection_500.py`, `exp10b_tool_selection_500_llm.py` | n=500 tool-selection study |

## Reproducing Table 4 / Table 8

The authoritative ablation (3 seeds x 4 back-ends, oracle matching) is:

```bash
cd benchmarks
# Run one back-end for three seeds
python exp_pte_ablation_canonical.py --model deepseek --seeds 42,43,44 --n 100
python exp_pte_ablation_canonical.py --model qwen     --seeds 42,43,44 --n 100
python exp_pte_ablation_canonical.py --model kimi     --seeds 42,43,44 --n 100
python exp_pte_ablation_canonical.py --model glm      --seeds 42,43,44 --n 100
# Aggregate all runs (mean / std / Wilson CI / McNemar)
python exp_pte_ablation_canonical.py --aggregate
```

The pre-computed raw outputs from our runs are provided under
`benchmarks/raw_data/runs/recanon/` (12 per-run JSON files +
`recanon_aggregate.json`).

## Notes

- Local filesystem paths in archived datasets have been normalized to
  `<HOME>` / `<PROJECT>` placeholders.
- The `exp9` replay dataset is exported from a development ExperienceStore
  snapshot; it captures tool-call error/retry events only.
