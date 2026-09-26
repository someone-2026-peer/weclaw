#!/usr/bin/env python3
"""E11 real-execution query-set builder.

Produces the committed query artifact for the E11 execution-level benchmark
(reviewer R1-1). Reuses the frozen, paper-reproduction building blocks
READ-ONLY (never modified):

  * ``exp1_pte_ablation.QUERY_TEMPLATES`` / ``INTENT_GROUND_TRUTH`` / ``get_all_tool_names``
  * ``e11_public_queries.PUBLIC_TOOLBENCH_LITE_E11`` (100 ToolBench-lite style)
  * ``sandbox_env.SANDBOX_FAMILIES`` (the 49 real-execution families: 15 base +
    22 Tier-1 + 12 Tier-2 record-and-freeze real-data)

Remapping (V2 L287): a query's real-execution SUCCESS PATH is
``acceptable_tools ∩ SANDBOX_FAMILIES`` -- derived from the frozen ground truth,
so no oracle acceptable-set is hand-edited. A query is ``in_sandbox`` when that
intersection is non-empty.

Sets produced:
  own_catalog  100 = 80 in-sandbox (8 intents, coverage-weighted) + 20 out-of-sandbox
                    (4 E11-specific NEGATIVE-CONTROL intents; the 20% control of
                    V2 L293). Under the 49-family sandbox NO exp1 intent is fully
                    out-of-sandbox any more, so the control is rebuilt here from
                    queries whose ``acceptable_tools`` are drawn ONLY from the 26
                    T3 families (dangerous / hardware / irreversible / never
                    executable in the sandbox). exp1's frozen INTENT_GROUND_TRUTH
                    and QUERY_TEMPLATES are NOT modified (fork-don't-modify).
  public       100 ToolBench-lite style, multi-domain, download-free
  FULL         200 = own(100) + public(100)          -> deepseek + glm
  SUBSET       100 = stratified own(50) + public(50)  -> qwen + moonshot

Everything is seed-fixed, deduplicated, validated (every tool is in the 75-tool
universe) and written to ``raw_data/e11_query_set.json``.

Usage::
    python e11_query_set.py            # build + write artifact
    python e11_query_set.py --check    # build + validate, no write
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import exp1_pte_ablation as base  # noqa: E402  (read-only reuse)
from sandbox_env import SANDBOX_FAMILIES, coverage_report  # noqa: E402
from e11_public_queries import PUBLIC_TOOLBENCH_LITE_E11  # noqa: E402

E11_QUERY_SEED = 20260604  # construction seed (independent of run seeds 42/43/44)
OUT_PATH = BENCH_DIR / "raw_data" / "e11_query_set.json"

# In-sandbox intents (success path lands in the 49 families), coverage-weighted.
# Weight ~ number of sandbox families reachable, so strong intents dominate.
IN_SANDBOX_INTENT_COUNTS = {
    "daily_assistant": 13,     # weather/datetime_tool/calculator/statistics
    "knowledge": 13,           # knowledge_rag/poetry/python_runner/file
    "data_analysis": 12,       # data_processor/statistics/file
    "document_processing": 11, # pdf_tool/format_converter
    "life_management": 11,     # diary/finance/todo
    "file_operation": 10,      # file
    "system_monitoring": 6,    # system_monitor
    "document_assembly": 4,    # file/weather (weak: natural tool doc_generator is out)
}
# E11-specific out-of-sandbox NEGATIVE CONTROL (V2 L293): 4 intents x 5 = 20.
# Every ``acceptable_tools`` entry is a T3-only family (universe - SANDBOX_FAMILIES),
# so NO acceptable tool can ever execute in the sandbox -> zero false gains is
# guaranteed BY CONSTRUCTION, not by the model's behaviour. These queries and
# their acceptable-sets are defined here (E11-specific); exp1's frozen
# INTENT_GROUND_TRUTH / QUERY_TEMPLATES are never modified.
OUT_OF_SANDBOX_CONTROL = {
    "screen_system_control": {
        "acceptable_tools": ["screen", "shell", "app_control"],
        "queries": [
            "帮我截取当前电脑屏幕的画面",
            "在命令行执行一条 shell 命令查看磁盘占用",
            "自动打开桌面上的记事本应用并点击菜单",
            "录制一段屏幕操作演示视频",
            "用系统命令列出当前正在运行的进程",
        ],
    },
    "speech_voice_control": {
        "acceptable_tools": ["speech_to_text", "voice_input", "voice_output"],
        "queries": [
            "把这段录音语音转写成文字稿",
            "用语音输入的方式口述一段消息",
            "把这篇文章用语音朗读出来",
            "实时听写我说的话并转成文本",
            "用语音播报今天的日程安排",
        ],
    },
    "image_scan_control": {
        "acceptable_tools": ["image_generator", "ocr", "document_scanner",
                             "id_photo", "media_capture"],
        "queries": [
            "帮我生成一张科幻风格的概念图片",
            "识别这张照片里的文字并提取出来",
            "扫描一份纸质文档保存为电子版",
            "拍一张标准证件照",
            "用摄像头采集一张图像",
        ],
    },
    "messaging_share_control": {
        "acceptable_tools": ["email", "wechat", "remote_file_share"],
        "queries": [
            "给张经理发一封会议通知邮件",
            "通过微信给家人发一条消息",
            "把这份报告远程分享给同事",
            "群发一封节日祝福邮件给所有客户",
            "把本地文件传输到远程服务器",
        ],
    },
}

_SUFFIXES = ["，请帮我处理", "，谢谢", "，尽快完成", "，详细一点", "，急用", ""]


# ------------------------------------------------------------------ helpers

def _sandbox_success_tools(acceptable_tools) -> list[str]:
    """Real-execution success path = acceptable ∩ 49 sandbox families."""
    return sorted(set(acceptable_tools) & SANDBOX_FAMILIES)


def _gen_intent_queries(intent: str, count: int, rng: random.Random) -> list[str]:
    """Deterministically produce ``count`` unique queries for an intent.

    Uses exp1's QUERY_TEMPLATES verbatim, then seeded suffix variation (the same
    technique as exp1.generate_test_queries) to exceed 10 templates when needed.
    """
    templates = list(base.QUERY_TEMPLATES[intent])
    out: list[str] = []
    seen: set[str] = set()
    for t in templates:                       # base pass
        if len(out) >= count:
            break
        if t not in seen:
            seen.add(t)
            out.append(t)
    i = 0                                      # seeded expansion pass
    guard = count * 30 + 50
    while len(out) < count and i < guard:
        cand = templates[i % len(templates)] + rng.choice(_SUFFIXES)
        i += 1
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    rng.shuffle(out)
    return out[:count]


def _record(qid, query, intent_or_domain, acceptable, source, language, split) -> dict:
    sbx = _sandbox_success_tools(acceptable)
    return {
        "qid": qid,
        "query": query,
        "intent": intent_or_domain,
        "acceptable_tools": sorted(set(acceptable)),   # oracle upper bound (reporting only)
        "sandbox_success_tools": sbx,                  # real-execution success path
        "in_sandbox": bool(sbx),
        "source": source,
        "language": language,
        "split": split,
        "in_full": False,
        "in_subset": False,
    }


# ------------------------------------------------------------------ builders

def build_own_catalog(seed: int = E11_QUERY_SEED) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    n = 0
    for intent, count in IN_SANDBOX_INTENT_COUNTS.items():
        acc = base.INTENT_GROUND_TRUTH[intent]
        for q in _gen_intent_queries(intent, count, rng):
            records.append(_record(f"e11_own_{n:03d}", q, intent, acc,
                                   "own_catalog", "zh", "in_sandbox"))
            n += 1
    for intent, spec in OUT_OF_SANDBOX_CONTROL.items():
        acc = spec["acceptable_tools"]
        for q in spec["queries"]:
            records.append(_record(f"e11_own_{n:03d}", q, intent, acc,
                                   "own_catalog", "zh", "out_of_sandbox"))
            n += 1
    return records


def build_public(seed: int = E11_QUERY_SEED) -> list[dict]:
    records: list[dict] = []
    for k, e in enumerate(PUBLIC_TOOLBENCH_LITE_E11):
        split = "in_sandbox" if _sandbox_success_tools(e["acceptable_tools"]) else "out_of_sandbox"
        records.append(_record(f"e11_pub_{k:03d}", e["query"], e["domain"],
                               e["acceptable_tools"], "public_toolbench_lite",
                               e.get("language", "en"), split))
    return records


def mark_subset(own: list[dict], public: list[dict], seed: int = E11_QUERY_SEED) -> None:
    """Stratified subset arm: 50 own (40 in / 10 out) + 50 public (domain-balanced)."""
    rng = random.Random(seed + 1)

    own_in = [r for r in own if r["split"] == "in_sandbox"]
    own_out = [r for r in own if r["split"] == "out_of_sandbox"]
    rng.shuffle(own_in)
    rng.shuffle(own_out)
    for r in own_in[:40] + own_out[:10]:
        r["in_subset"] = True

    # public: round-robin across domains for stratification
    by_domain: dict[str, list[dict]] = {}
    for r in public:
        by_domain.setdefault(r["intent"], []).append(r)
    for dom in by_domain:
        rng.shuffle(by_domain[dom])
    picked, domains = 0, sorted(by_domain)
    while picked < 50:
        progressed = False
        for dom in domains:
            if by_domain[dom] and picked < 50:
                r = by_domain[dom].pop()
                r["in_subset"] = True
                picked += 1
                progressed = True
        if not progressed:
            break


# ------------------------------------------------------------------ validation

def validate(own: list[dict], public: list[dict]) -> dict:
    universe = base.get_all_tool_names()
    bad = set()
    for r in own + public:
        for t in r["acceptable_tools"]:
            if t not in universe:
                bad.add(t)
    if bad:
        raise ValueError(f"acceptable_tools not in the {len(universe)}-tool universe: {sorted(bad)}")

    for name, recs in (("own", own), ("public", public)):
        qs = [r["query"] for r in recs]
        dupes = [q for q, c in Counter(qs).items() if c > 1]
        if dupes:
            raise ValueError(f"duplicate queries in {name}: {dupes[:5]}")
        ids = [r["qid"] for r in recs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate qids in {name}")

    own_in = sum(1 for r in own if r["split"] == "in_sandbox")
    own_out = sum(1 for r in own if r["split"] == "out_of_sandbox")
    assert len(own) == 100 and own_in == 80 and own_out == 20, (len(own), own_in, own_out)
    assert len(public) == 100, len(public)

    # The negative control must be GENUINE (V2 L293): no own out-of-sandbox query
    # may have a runnable sandbox success path -- all its acceptable tools are
    # T3-only, so it can never execute and can never manufacture a false gain.
    leaked = [r["qid"] for r in own if r["split"] == "out_of_sandbox" and r["sandbox_success_tools"]]
    assert not leaked, f"own negative control leaked into the sandbox: {leaked}"
    # ...and every own in-sandbox query really has an executable success path.
    dead = [r["qid"] for r in own if r["split"] == "in_sandbox" and not r["sandbox_success_tools"]]
    assert not dead, f"own in-sandbox queries with no success path: {dead}"

    cov = coverage_report(universe)
    return {
        "universe_size": len(universe),
        "sandbox_families": sorted(SANDBOX_FAMILIES),
        "coverage_pct": cov["coverage_pct"],
        "own_total": len(own), "own_in_sandbox": own_in, "own_out_of_sandbox": own_out,
        "public_total": len(public),
        "public_in_sandbox": sum(1 for r in public if r["in_sandbox"]),
        "public_out_of_sandbox": sum(1 for r in public if not r["in_sandbox"]),
        "own_intent_dist": dict(Counter(r["intent"] for r in own)),
        "public_domain_dist": dict(Counter(r["intent"] for r in public)),
    }


def build_all(seed: int = E11_QUERY_SEED):
    own = build_own_catalog(seed)
    public = build_public(seed)
    for r in own + public:
        r["in_full"] = True
    mark_subset(own, public, seed)
    stats = validate(own, public)
    all_q = own + public
    stats["full_total"] = sum(1 for r in all_q if r["in_full"])
    stats["subset_total"] = sum(1 for r in all_q if r["in_subset"])
    stats["subset_own"] = sum(1 for r in own if r["in_subset"])
    stats["subset_public"] = sum(1 for r in public if r["in_subset"])
    stats["full_in_sandbox"] = sum(1 for r in all_q if r["in_sandbox"])
    stats["full_out_of_sandbox"] = sum(1 for r in all_q if not r["in_sandbox"])
    return all_q, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the E11 real-execution query set.")
    ap.add_argument("--seed", type=int, default=E11_QUERY_SEED)
    ap.add_argument("--check", action="store_true", help="validate only, do not write")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    all_q, stats = build_all(args.seed)

    print("=== E11 query set ===")
    for k in ("universe_size", "coverage_pct", "own_total", "own_in_sandbox",
              "own_out_of_sandbox", "public_total", "public_in_sandbox",
              "public_out_of_sandbox", "full_total", "full_in_sandbox",
              "full_out_of_sandbox", "subset_total", "subset_own", "subset_public"):
        print(f"  {k}: {stats[k]}")
    print("  own_intent_dist:", json.dumps(stats["own_intent_dist"], ensure_ascii=False))
    print("  public_domain_dist:", json.dumps(stats["public_domain_dist"], ensure_ascii=False))

    if args.check:
        print("CHECK_ONLY: validation passed, nothing written.")
        return

    payload = {
        "_meta": {
            "purpose": "E11 real-execution benchmark query set (upgrades oracle selection-level to execution-level; reviewer R1-1).",
            "generated_by": "e11_query_set.py",
            "seed": args.seed,
            "reuses_read_only": ["exp1_pte_ablation.QUERY_TEMPLATES/INTENT_GROUND_TRUTH/get_all_tool_names",
                                 "e11_public_queries.PUBLIC_TOOLBENCH_LITE_E11", "sandbox_env.SANDBOX_FAMILIES"],
            "remapping": "sandbox_success_tools = acceptable_tools ∩ SANDBOX_FAMILIES (derived from frozen ground truth).",
            "arms": {"FULL_200": "deepseek + glm", "SUBSET_100": "qwen + moonshot (stratified 50 own + 50 public)"},
            "stats": {k: v for k, v in stats.items() if k not in ("own_intent_dist", "public_domain_dist")},
            "own_intent_dist": stats["own_intent_dist"],
            "public_domain_dist": stats["public_domain_dist"],
        },
        "queries": all_q,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out} ({len(all_q)} queries)")


if __name__ == "__main__":
    main()
