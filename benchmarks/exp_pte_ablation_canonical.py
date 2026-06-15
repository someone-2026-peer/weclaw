#!/usr/bin/env python3
"""Canonical PTE-FD ablation (recanon): n=100, 4 configs, multi-model, multi-seed.

Rebuilds the authoritative source for the n=100 single-step ablation
(paper Table 4) and the cross-model generalization (paper Table 8) after the
original producing scripts were found missing/unreproducible in the repo.

Configurations (oracle matching against intent ground-truth tool sets):
    static  - all tool schemas in the benchmark registry
    pte     - tiered exposure (recommended tier), no escalation
    pte_fd  - tiered exposure + failure-driven escalation (k=2)
    keyword - keyword/phrase retrieval baseline (top-k=10)

Methodology is reused verbatim from exp1_pte_ablation (tool universe, query
generator with seeded RNG, keyword retrieval, ToolExposureEngine escalation)
so that results stay comparable to the historical ablation lineage.

Providers (OpenAI-compatible endpoints, function calling, temperature=0.0):
    deepseek / qwen / kimi / glm

Usage:
    python exp_pte_ablation_canonical.py --model deepseek --seeds 42,43,44 --n 100
    python exp_pte_ablation_canonical.py --model qwen --seeds 42 --n 100
    python exp_pte_ablation_canonical.py --aggregate
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

from src.core.prompts import detect_intent_with_confidence
from src.core.tool_exposure import ToolExposureEngine, _extract_tool_name

# Reuse validated building blocks from the historical ablation script.
import exp1_pte_ablation as base

CONFIGS = ("static", "pte", "pte_fd", "keyword")

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "max_tokens": 256,
        "paper_name": "deepseek-v4-flash",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "QWEN_API_KEY",
        "model": "qwen-max",
        "max_tokens": 256,
        "paper_name": "qwen-max",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "KIMI_API_KEY",
        "model": "moonshot-v1-8k",
        "max_tokens": 512,
        "paper_name": "kimi",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "GLM_API_KEY",
        "model": "glm-4-flash",
        "max_tokens": 512,
        "paper_name": "glm",
    },
}

SYSTEM_PROMPT = (
    "你是 WeClaw AI 助手的工具选择模块。根据用户请求，选择最合适的工具来完成任务。"
    "你必须从提供的工具列表中选择一个工具。只选择工具，不要执行。"
)

OUT_DIR = BENCH_DIR / "raw_data" / "runs" / "recanon"


def call_llm(client: OpenAI, model_id: str, query: str, schemas: list, max_tokens: int) -> tuple:
    """Return (selected_tool_name, prompt_tokens). '__ERROR__' marks an API failure."""
    if not schemas:
        return ("", 0)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=schemas,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        msg = response.choices[0].message
        if msg.tool_calls:
            return (_extract_tool_name(msg.tool_calls[0].function.name), prompt_tokens)
        return ("", prompt_tokens)
    except Exception as exc:
        print(f"  LLM call error: {exc}")
        return ("__ERROR__", 0)


def run_single(model_key: str, seed: int, n: int, model_id: str = None,
               max_tokens: int = None, sleep: float = 0.05) -> dict:
    prov = PROVIDERS[model_key]
    model_id = model_id or prov["model"]
    max_tokens = max_tokens or prov["max_tokens"]
    api_key = os.getenv(prov["key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing API key env {prov['key_env']} for model {model_key}")

    client = OpenAI(api_key=api_key, base_url=prov["base_url"])
    registry = base.MockToolRegistry()
    queries = base.generate_test_queries(n=n, seed=seed)

    all_names = base.get_all_tool_names()
    static_schemas = base.build_schemas(all_names)
    static_bytes = len(json.dumps(static_schemas, ensure_ascii=False).encode("utf-8"))

    engine = ToolExposureEngine(registry, enabled=True, enable_annotation=True,
                                failures_to_upgrade=2)

    stats = {c: {"correct": 0, "total": 0, "tokens": 0, "schema_bytes": 0} for c in CONFIGS}
    stats["pte_fd"].update({"recovered": 0, "initial_failures": 0})
    per_query = {c: [] for c in CONFIGS}
    errors = 0

    print(f"[{model_key}/{model_id} seed={seed}] n={len(queries)} "
          f"static_schemas={len(static_schemas)} ({static_bytes} bytes)")

    for i, q in enumerate(queries):
        query = q["query"]
        gt = q["ground_truth_tools"]
        intent_result = detect_intent_with_confidence(query)

        # static
        engine.reset()
        sel, tk = call_llm(client, model_id, query, static_schemas, max_tokens)
        if sel == "__ERROR__":
            errors += 1
            sel = ""
        ok = sel in gt
        stats["static"]["correct"] += int(ok)
        stats["static"]["total"] += 1
        stats["static"]["tokens"] += tk
        stats["static"]["schema_bytes"] += static_bytes
        per_query["static"].append(int(ok))

        # pte (tiered, no escalation)
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        pte_bytes = len(json.dumps(pte_schemas, ensure_ascii=False).encode("utf-8"))
        sel, tk = call_llm(client, model_id, query, pte_schemas, max_tokens)
        if sel == "__ERROR__":
            errors += 1
            sel = ""
        ok = sel in gt
        stats["pte"]["correct"] += int(ok)
        stats["pte"]["total"] += 1
        stats["pte"]["tokens"] += tk
        stats["pte"]["schema_bytes"] += pte_bytes
        per_query["pte"].append(int(ok))

        # pte_fd (tiered + failure-driven escalation)
        engine.reset()
        pte_schemas = engine.get_schemas(intent_result)
        fd_bytes = len(json.dumps(pte_schemas, ensure_ascii=False).encode("utf-8"))
        sel, tk = call_llm(client, model_id, query, pte_schemas, max_tokens)
        if sel == "__ERROR__":
            errors += 1
            sel = ""
        fd_tokens = tk
        ok_initial = sel in gt
        if not ok_initial:
            stats["pte_fd"]["initial_failures"] += 1
            engine.report_failure()
            engine.report_failure()
            esc_schemas = engine.get_schemas(intent_result)
            esc_bytes = len(json.dumps(esc_schemas, ensure_ascii=False).encode("utf-8"))
            sel2, tk2 = call_llm(client, model_id, query, esc_schemas, max_tokens)
            if sel2 == "__ERROR__":
                errors += 1
                sel2 = ""
            fd_tokens += tk2
            ok_final = sel2 in gt
            if ok_final:
                stats["pte_fd"]["recovered"] += 1
            fd_bytes = max(fd_bytes, esc_bytes)
        else:
            engine.report_success()
            ok_final = True
        stats["pte_fd"]["correct"] += int(ok_final)
        stats["pte_fd"]["total"] += 1
        stats["pte_fd"]["tokens"] += fd_tokens
        stats["pte_fd"]["schema_bytes"] += fd_bytes
        per_query["pte_fd"].append(int(ok_final))

        # keyword retrieval baseline
        kw_schemas = base.itr_select_tools(query, static_schemas, top_k=10)
        kw_bytes = len(json.dumps(kw_schemas, ensure_ascii=False).encode("utf-8"))
        sel, tk = call_llm(client, model_id, query, kw_schemas, max_tokens)
        if sel == "__ERROR__":
            errors += 1
            sel = ""
        ok = sel in gt
        stats["keyword"]["correct"] += int(ok)
        stats["keyword"]["total"] += 1
        stats["keyword"]["tokens"] += tk
        stats["keyword"]["schema_bytes"] += kw_bytes
        per_query["keyword"].append(int(ok))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(queries)} done")
        time.sleep(sleep)

    summary = {
        "model_key": model_key,
        "model_id": model_id,
        "paper_name": prov["paper_name"],
        "seed": seed,
        "n": len(queries),
        "errors": errors,
        "configs": {},
        "per_query": per_query,
    }
    static_bytes_total = stats["static"]["schema_bytes"] or 1
    for c in CONFIGS:
        s = stats[c]
        total = s["total"] or 1
        acc = s["correct"] / total * 100
        tok_red = 0.0 if c == "static" else (1 - s["schema_bytes"] / static_bytes_total) * 100
        entry = {
            "accuracy": round(acc, 2),
            "correct": s["correct"],
            "total": s["total"],
            "avg_tokens": round(s["tokens"] / total, 1),
            "token_reduction": round(tok_red, 1),
        }
        if c == "pte_fd":
            init_fail = s["initial_failures"]
            entry["initial_failures"] = init_fail
            entry["recovered"] = s["recovered"]
            entry["recovery_rate"] = round(s["recovered"] / max(init_fail, 1) * 100, 1)
        summary["configs"][c] = entry

    return summary


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (round((center - half) * 100, 2), round((center + half) * 100, 2))


def mcnemar(a: list, b: list) -> dict:
    """McNemar test on paired correctness arrays a vs b. Returns b/c and p (chi2, 1 df)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    nd = n01 + n10
    if nd == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / nd  # continuity-corrected
    # survival of chi-square with 1 df = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"n01": n01, "n10": n10, "chi2": round(chi2, 3), "p": p}


def run_mode(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        summary = run_single(args.model, seed, args.n, model_id=args.model_id,
                             max_tokens=args.max_tokens, sleep=args.sleep)
        out_path = OUT_DIR / f"{args.model}_seed{seed}_n{args.n}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved {out_path}")
        for c in CONFIGS:
            e = summary["configs"][c]
            print(f"  {c:8s} acc={e['accuracy']:5.1f}%  avg_tokens={e['avg_tokens']:7.1f}  "
                  f"({e['correct']}/{e['total']})")
        if summary["errors"]:
            print(f"  WARNING: {summary['errors']} API errors")


def aggregate_mode(args) -> None:
    files = sorted(OUT_DIR.glob("*_n*.json"))
    if not files:
        raise SystemExit(f"No run files in {OUT_DIR}")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    by_model = {}
    for r in runs:
        by_model.setdefault(r["model_key"], []).append(r)

    report = {"per_model": {}}
    print("=" * 72)
    print("RECANON AGGREGATE")
    print("=" * 72)
    for model_key, model_runs in by_model.items():
        model_runs.sort(key=lambda r: r["seed"])
        paper_name = model_runs[0]["paper_name"]
        model_id = model_runs[0]["model_id"]
        print(f"\n### {model_key} ({model_id}, paper={paper_name})  "
              f"seeds={[r['seed'] for r in model_runs]}")
        cfg_report = {}
        for c in CONFIGS:
            accs = [r["configs"][c]["accuracy"] for r in model_runs]
            toks = [r["configs"][c]["avg_tokens"] for r in model_runs]
            mean = sum(accs) / len(accs)
            if len(accs) > 1:
                var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            tot_correct = sum(r["configs"][c]["correct"] for r in model_runs)
            tot_total = sum(r["configs"][c]["total"] for r in model_runs)
            ci = wilson_ci(tot_correct, tot_total)
            cfg_report[c] = {
                "acc_mean": round(mean, 1),
                "acc_std": round(std, 1),
                "acc_per_seed": accs,
                "avg_tokens_mean": round(sum(toks) / len(toks), 1),
                "pooled_correct": tot_correct,
                "pooled_total": tot_total,
                "wilson_ci_pooled": ci,
            }
            line = f"  {c:8s} acc={mean:5.1f}"
            if len(accs) > 1:
                line += f" +/- {std:4.1f}"
            line += f"  tokens={cfg_report[c]['avg_tokens_mean']:7.1f}  CI{ci}"
            print(line)

        pooled_pte = [v for r in model_runs for v in r["per_query"]["pte"]]
        pooled_fd = [v for r in model_runs for v in r["per_query"]["pte_fd"]]
        pooled_static = [v for r in model_runs for v in r["per_query"]["static"]]
        mc_fd_pte = mcnemar(pooled_pte, pooled_fd)
        mc_fd_static = mcnemar(pooled_static, pooled_fd)
        cfg_report["_mcnemar_fd_vs_pte"] = mc_fd_pte
        cfg_report["_mcnemar_fd_vs_static"] = mc_fd_static
        print(f"  McNemar FD vs PTE:    b={mc_fd_pte['n10']} c={mc_fd_pte['n01']} "
              f"chi2={mc_fd_pte['chi2']} p={mc_fd_pte['p']:.2e}")
        print(f"  McNemar FD vs Static: b={mc_fd_static['n10']} c={mc_fd_static['n01']} "
              f"chi2={mc_fd_static['chi2']} p={mc_fd_static['p']:.2e}")
        report["per_model"][model_key] = cfg_report

    out = OUT_DIR / "recanon_aggregate.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical PTE-FD ablation (recanon)")
    ap.add_argument("--model", choices=list(PROVIDERS.keys()))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model-id", default=None, help="override provider model id")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_mode(args)
        return
    if not args.model:
        ap.error("--model is required unless --aggregate")
    run_mode(args)


if __name__ == "__main__":
    main()
