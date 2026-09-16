#!/usr/bin/env python3
r"""Experiment 12: EBEAC vs.\ simplified ERL vs.\ ExpeL-style insights on the SAME traces.

回应 R1-6：原 Table 3（`tab:ablation_ebe`）把 ERL/ExpeL 与 EBEAC 混排在同一张表，
且 ERL/ExpeL 的 Recall/Prevention 两列为 `---`，无法比较。本脚本在**同一批 196 条
真实生产轨迹**上重实现三方（含 Raw-Log 基线共四方），用**同一判据**产出
Recall / Prevention，使 Table 3b 成为真正可比的同轨迹对照。

──────────────────────── 同源可比性（本脚本的核心约束）────────────────────────
四方共享 exp9（E13，已发表 §6.4）的**全部**判据，逐条对齐、不另立口径：
  * 数据集   ：`weclaw_ebeac_log_replay_realdb_20260808.json`，196 条，train/test=137/59
  * 分母     ：`n_test_errors` = test split 中 status=="error" 的条数（=59）
  * 标准答案 ：`event["fix_pattern"]`，**精确字符串相等**判定命中
  * 预测形状 ：每方法对每个 test 错误事件产出 **top-3** 有序 fix_pattern 列表
  * 指标公式 ：recall@1 = top1/N、recall@3 = top3/N、precision = top1/predictions、
              coverage = predictions/N、prevention = top3/N、FP = (predictions-top3)/N
  * EBEAC 一方**直接复用** exp9 的 `evaluate_experience_store_memory()` 代码路径
    （同一个 `SqliteOnlyExperienceStore` + `store.recall(min_similarity=0.6)`），
    因此它必然复现 §6.4 已发表的六个数字 —— 这构成本脚本最强的正对照。

──────────────────────── 为什么不能沿用既有基准的错误处理 ────────────────────────
`exp_pte_ablation_canonical.py` 的 `call_llm()` 捕获异常后返回 `"__ERROR__"`，
`run_single()` 再把它当作 `selected=""` ⇒ **判为答错**。限流因此不会让实验失败，
而是被静默计为错误答案，把限流严重方法的指标结构性拉低。E12 若沿用该模式，
三方差异里就会混入基础设施噪声。本脚本的处置：
  1. `max_retries=0` 交给 SDK 之外自己控制：显式重试 + 指数退避，只重试**可重试类**
     （限流/超时/5xx），对 404/400/401 立即失败（重试无意义且浪费配额）
  2. 重试耗尽后**绝不**转成空答案，而是记入 `api_failures` 并**双口径**报告：
       - `strict`     ：失败事件算 miss，分母恒为 N（与 exp9 同分母，跨方法可比）
       - `valid_only` ：分母 = N - failures（仅报告 LLM 真正应答过的事件）
     两口径同时给出，审稿人可自行判断基础设施噪声的影响上界。
  3. `temperature=0.0`（与全部既有基准一致；探测已排除只允许 temperature=1 的
     kimi-k2.5，故该口径无冲突）
  4. checkpoint：每次 LLM 调用结果增量落盘 JSONL，`--resume` 可断点续跑

──────────────────────── 后端选型依据（作者要求：先排除易错模型）────────────────────────
默认 `deepseek-chat`。选型来自全模型稳定性探测（13 候选 × 6 调用、`max_retries=0`）：
排除 glm-4.7-flash（429 code 1305 限流 2/6，成功率 67%）、moonshot-v1-8k（404 已下架）、
kimi-k2.5（400「only temperature=1 is allowed」）、minimax-m2.7:free（免费版已下架）。
deepseek-chat 实测 100% 成功 / JSON 100% 合规 / p50=1.08s / p95=1.66s，且它正是
exp2 当初模拟 ERL/ExpeL 时使用的后端 ⇒ 跨实验后端一致，成本口径可直接对齐。

Usage:
    # 零 API：只跑 EBEAC + Raw-Log + 全部负对照，验证能复现 §6.4（先跑这个）
    python exp12_ebeac_baselines_same_traces.py --dry-run

    # 冒烟：每方法只跑前 3 条 train 提取 + 前 3 条 test 检索
    python exp12_ebeac_baselines_same_traces.py --limit-extract 3 --limit-test 3

    # 全量：196 条轨迹上的完整四方对照
    python exp12_ebeac_baselines_same_traces.py

    # 断点续跑
    python exp12_ebeac_baselines_same_traces.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
BENCH_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BENCH_DIR))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from benchmark_utils import get_output_dir
from dataset_utils import count_by_key, get_default_output_path, load_dataset_items, write_json

# 复用 exp9（E13）的 EBEAC 代码路径 —— 同源可比性的技术保障，不重新实现
from exp9_ebeac_log_replay import (
    BUILTIN_LOG_REPLAY_SAMPLE,
    evaluate_experience_store_memory,
    tokenize,
)

OUTPUT_DIR = get_output_dir(__file__)
OUTPUT_PATH = get_default_output_path(__file__, "exp12_same_traces_results.json")
CHECKPOINT_PATH = OUTPUT_DIR / "exp12_llm_checkpoint.jsonl"

DEFAULT_DATASET = (BENCH_DIR / "planned_artifacts"
                   / "weclaw_ebeac_log_replay_realdb_20260808.json")

# ---- 后端配置（选型依据见模块 docstring）----
BASE_URL = "https://api.deepseek.com"
KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_MODEL = "deepseek-chat"
TEMPERATURE = 0.0          # 与全部既有基准一致，保证可复现
MAX_TOKENS = 400
MAX_RETRIES = 3            # 自管重试（SDK 侧 max_retries=0）
BACKOFF = (1.0, 3.0, 9.0)  # 指数退避
REQUEST_TIMEOUT = 60.0
CALL_GAP = 0.3             # 调用间隔（介于 canonical 0.05 与 exp4b 0.5 之间）

RETRYABLE = {"RATE_LIMIT", "TIMEOUT", "SERVER_ERROR"}

# §6.4（E13）已发表的六个数字 —— EBEAC 一方必须精确复现，这是本脚本的正对照
PUBLISHED_EBEAC = {
    "recall_at_1": 71.19, "recall_at_3": 76.27, "precision": 89.36,
    "coverage_rate": 79.66, "prevention_rate": 76.27, "false_positive_rate": 3.39,
}
PUBLISHED_COUNTS = {"n_train": 137, "n_test": 59, "n_test_errors": 59}


# ══════════════════════════════════════════════════════════════════════════
# 一、错误分类与鲁棒 LLM 调用
# ══════════════════════════════════════════════════════════════════════════

def classify(exc: Exception) -> str:
    """把异常映射到可决策的错误类。顺序敏感：先判限流，再判模型名/鉴权。"""
    s = str(exc).lower()
    code = (getattr(getattr(exc, "response", None), "status_code", None)
            or getattr(exc, "status_code", None))
    if code == 429 or "rate limit" in s or "rate_limit" in s or "too many requests" in s \
            or "并发" in s or "频繁" in s or "访问量过大" in s:
        return "RATE_LIMIT"
    if code is not None and 500 <= int(code) < 600:
        return "SERVER_ERROR"
    if "temperature" in s and ("invalid" in s or "unsupported" in s or "not support" in s):
        return "TEMP_UNSUPPORTED"
    if code in (401, 403) or "invalid api key" in s or "incorrect api key" in s:
        return "AUTH"
    if code == 404 or "model not found" in s or "does not exist" in s or "not found the model" in s:
        return "MODEL_NOT_FOUND"
    if "timeout" in s or "timed out" in s:
        return "TIMEOUT"
    if "context" in s and ("length" in s or "exceed" in s):
        return "CONTEXT_LENGTH"
    return "OTHER"


def parse_ranked(raw: str, n_candidates: int) -> list[int] | None:
    """解析 LLM 的排序输出为候选**索引**列表。

    强制输出索引而非自由文本：若让 LLM 自由生成 fix_pattern，其措辞几乎不可能与
    ground truth 精确相等，命中会被系统性低估 —— 那对本方法有利、对基线不公，
    属于对照实验的设计缺陷。限定在候选集内选择可消除该偏差。
    """
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    seq = obj.get("ranked")
    if not isinstance(seq, list):
        return None
    out: list[int] = []
    for v in seq:
        try:
            k = int(v)
        except Exception:
            continue
        if 0 <= k < n_candidates and k not in out:
            out.append(k)
    return out or None


def make_caller(model: str, dry: bool) -> Callable[[list[dict]], dict]:
    """返回一个鲁棒调用器。dry=True 时不发任何网络请求（供 --dry-run 与负对照）。"""
    if dry:
        def _dry(messages: list[dict]) -> dict:
            return {"ok": False, "err_class": "DRY_RUN", "attempts": 0,
                    "content": "", "err": "dry-run: no API call issued"}
        return _dry

    from openai import OpenAI  # 延迟导入：dry-run 无需网络栈
    api_key = os.getenv(KEY_ENV, "")
    if not api_key:
        raise SystemExit(f"Missing API key env {KEY_ENV}; use --dry-run for a zero-API run")
    # ⚠️ max_retries=0：SDK 默认 2 次自动重试会掩盖限流，重试策略由本脚本显式掌控
    client = OpenAI(api_key=api_key, base_url=BASE_URL,
                    timeout=REQUEST_TIMEOUT, max_retries=0)

    def _call(messages: list[dict]) -> dict:
        last: dict = {}
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                )
                content = (resp.choices[0].message.content or "").strip()
                usage = resp.usage
                return {"ok": True, "content": content, "attempts": attempt + 1,
                        "err_class": "",
                        "in_tok": getattr(usage, "prompt_tokens", 0) or 0,
                        "out_tok": getattr(usage, "completion_tokens", 0) or 0}
            except Exception as exc:
                cls = classify(exc)
                last = {"ok": False, "content": "", "attempts": attempt + 1,
                        "err_class": cls, "err": str(exc)[:220], "in_tok": 0, "out_tok": 0}
                # 只对可重试类退避重试；404/400/401 立即放弃（重试无意义且烧配额）
                if cls not in RETRYABLE or attempt >= MAX_RETRIES:
                    return last
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
        return last

    return _call


# ══════════════════════════════════════════════════════════════════════════
# 二、checkpoint（断点续跑；避免一次限流毁掉整批付费调用）
# ══════════════════════════════════════════════════════════════════════════

class Checkpoint:
    def __init__(self, path: Path, resume: bool, enabled: bool = True):
        self.path, self.enabled = path, enabled
        self.cache: dict[str, dict] = {}
        if resume and enabled and path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    self.cache[rec["key"]] = rec["val"]
                except Exception:
                    continue
        self._fh = (open(path, "a", encoding="utf-8", newline="\n") if enabled else None)

    def get(self, key: str) -> dict | None:
        return self.cache.get(key)

    def put(self, key: str, val: dict) -> None:
        self.cache[key] = val
        if self._fh:
            self._fh.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def dry_disabled(path: Path, resume: bool) -> bool:
    """占位：checkpoint 始终可写；单独成函数以便负对照替换其行为。"""
    return False


# ══════════════════════════════════════════════════════════════════════════
# 三、四方方法实现（同一轨迹、同一判据）
# ══════════════════════════════════════════════════════════════════════════

SYS_EXTRACT = (
    "你是 WeClaw 的经验提炼模块。阅读工具调用失败记录，归纳可复用的修复经验。"
    "只输出 JSON，不要任何解释文字。"
)
SYS_RANK = (
    "你是 WeClaw 的经验检索模块。给定一条新的工具失败事件和若干条候选经验，"
    "按相关性从高到低选出最多 3 条候选的编号。只输出 JSON：{\"ranked\": [编号, ...]}。"
)


def event_brief(ev: dict[str, Any]) -> str:
    md = ev.get("metadata", {}) or {}
    return (f"tool={ev.get('tool_name')} action={ev.get('action')} "
            f"error_type={ev.get('error_type')} fix_pattern={ev.get('fix_pattern')}\n"
            f"query={md.get('query', '')}\ndiagnosis={str(md.get('diagnosis', ''))[:300]}")


def seedable(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """与 exp9 的 `seed_experience_store` 完全同一筛选条件，保证四方共享同一知识源。"""
    return [e for e in events
            if e.get("status") == "error" and e.get("fix_pattern")]


def _fp(s: str) -> str:
    """内容指纹（12 位十六进制）：让 checkpoint 的 key 随**提示词内容**失效。

    两侧踩过同一类坑（key 未能唯一确定它缓存的那次调用的输入）：
      * 检索侧：LLM 返回的是 kb 的**位置索引**，kb 一变索引即失义（见 llm_retrieve）
      * 提取侧：`sess_{session_id[:24]}` 截断碰撞 —— 实测 6 个长 98 的 session_id
        共享前缀 `pwa_68e2619e-a46d-42d5-a`，5 个单元复用了别人的规则
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# ---------- 方法 1：Raw Error Log（零抽象基线，Table 3 现有行的可复算实现）----------

def raw_log_text(ev: dict[str, Any]) -> str:
    """Raw Error Log 只允许用**原始自由文本**（query + diagnosis）。

    ⚠️ 不得用 tool_name / error_type / fix_pattern 等结构化字段，两个理由：
      1. error_type 本身已经是一种抽象，用了就不叫 "raw" error log；
      2. fix_pattern 与 (tool_name, error_type) 近乎一一对应（如「工具试错模式:
         shell工具重试成功」），用它们检索会使基线退化成 **oracle 查表**。
    首版实现用了 exp9 的 `lexical_score`（error_type 相同 +4、tool_name 相同 +3）
    犯了这个错，实测 recall@1 虚高到 83.05%、**反超本文方法 EBEAC（71.19%）**，
    且因未去重导致 R@1 ≡ R@3。两个缺陷均已修正。
    """
    md = ev.get("metadata", {}) or {}
    return f"{md.get('query', '')} {md.get('diagnosis', '')}"


def raw_log_score(train_ev: dict[str, Any], test_ev: dict[str, Any]) -> float:
    """纯 Jaccard token 重叠，无任何结构化字段加成（避免 oracle 泄漏）。"""
    a, b = tokenize(raw_log_text(train_ev)), tokenize(raw_log_text(test_ev))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_raw_log(train: list[dict], test_errors: list[dict]) -> dict:
    """不做任何抽象：直接把 train 的原始错误文本当知识库，纯词面检索 top-3。

    这一行在原 Table 3 里是 `0.0 / 4 / 7.0% / 5.3%`，但**全仓零可计算出处**
    （exp2 输出根本没有 recall/prevention 字段）。本函数给它一个真实、可复算的实现。
    预测列表**去重**，与 EBEAC 的 per-abstract_pattern 去重同口径，否则同一
    fix_pattern 的多条 train 记录会占满 top-3，使 R@3 失去意义。
    """
    kb = seedable(train)
    stats = {"top1": 0, "top3": 0, "predictions": 0, "latency_ms": [], "api_failures": 0}
    for ev in test_errors:
        t0 = time.perf_counter()
        scored = sorted(((raw_log_score(c, ev), c["event_id"], c["fix_pattern"]) for c in kb),
                        key=lambda x: (-x[0], x[1]))
        stats["latency_ms"].append((time.perf_counter() - t0) * 1000)
        preds: list[str] = []
        for s, _eid, fp in scored:
            if s <= 0.0:
                break
            if fp not in preds:
                preds.append(fp)
            if len(preds) >= 3:
                break
        _tally(stats, preds, ev.get("fix_pattern"))
    return stats | {"kb_size": len(kb), "llm_calls": 0}


# ---------- 方法 2/3 共用：LLM 提取 + LLM 检索 ----------

def llm_extract(caller, ckpt: Checkpoint, tag: str, units: list[dict],
                prompt_fn: Callable[[dict], str], counters: dict) -> list[dict]:
    """对每个单元调一次 LLM 产出经验条目。失败单元被显式记录，不静默降级。"""
    out: list[dict] = []
    for u in units:
        key = f"{tag}::{u['key']}"
        hit = ckpt.get(key)
        if hit is not None:
            res = hit
            counters["cache_hits"] += 1
        else:
            res = caller([{"role": "system", "content": SYS_EXTRACT},
                          {"role": "user", "content": prompt_fn(u)}])
            ckpt.put(key, res)
            counters["llm_calls"] += 1
            counters["in_tok"] += res.get("in_tok", 0)
            counters["out_tok"] += res.get("out_tok", 0)
            time.sleep(CALL_GAP)
        if not res.get("ok"):
            counters["extract_failures"] += 1
            counters["err_classes"][res.get("err_class", "?")] += 1
            continue
        parsed = _parse_insight(res.get("content", ""))
        if parsed is None:
            counters["parse_failures"] += 1
            continue
        # 经验条目**必须**携带来源事件的 fix_pattern 作为 payload，
        # 使四方落在同一候选空间（否则 LLM 方法永远无法精确命中，对照不公）
        out.append({**parsed, "fix_patterns": u["fix_patterns"], "key": u["key"]})
    return out


def _parse_insight(raw: str) -> dict | None:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or not str(obj.get("rule", "")).strip():
        return None
    return {"rule": str(obj["rule"]).strip()[:600],
            "scope_tool": str(obj.get("scope_tool", "")).strip()[:60]}


def llm_retrieve(caller, ckpt: Checkpoint, tag: str, kb: list[dict],
                 test_errors: list[dict], counters: dict) -> dict:
    """对每个 test 错误事件，让 LLM 从 kb 里选 top-3（输出候选**索引**）。"""
    stats = {"top1": 0, "top3": 0, "predictions": 0, "latency_ms": [], "api_failures": 0}
    if not kb:
        stats["api_failures"] = len(test_errors)
        return stats | {"kb_size": 0}
    listing = "\n".join(f"[{i}] tool={k.get('scope_tool', '')} rule={k['rule']}"
                        for i, k in enumerate(kb))
    # ⚠️ key 必须含 kb 指纹：LLM 返回的是 listing 里的**位置索引**，而 listing 随
    # --limit-extract 变化。若 key 只有 event_id，冒烟跑（kb 小）缓存的索引会在全量跑
    # --resume 时命中，指向**另一条**经验 —— 静默污染，且指标看起来完全正常。
    kb_fp = hashlib.sha256(listing.encode("utf-8")).hexdigest()[:12]
    for ev in test_errors:
        key = f"{tag}::retrieve::{kb_fp}::{ev['event_id']}"
        hit = ckpt.get(key)
        if hit is not None:
            res = hit
            counters["cache_hits"] += 1
        else:
            t0 = time.perf_counter()
            res = caller([{"role": "system", "content": SYS_RANK},
                          {"role": "user",
                           "content": f"新失败事件：\n{event_brief(ev)}\n\n候选经验：\n{listing}"}])
            res["latency_ms"] = (time.perf_counter() - t0) * 1000
            ckpt.put(key, res)
            counters["llm_calls"] += 1
            counters["in_tok"] += res.get("in_tok", 0)
            counters["out_tok"] += res.get("out_tok", 0)
            time.sleep(CALL_GAP)
        stats["latency_ms"].append(res.get("latency_ms", 0.0))
        if not res.get("ok"):
            # ⚠️ 关键：绝不转成空答案当 miss，而是计入 api_failures 走双口径
            stats["api_failures"] += 1
            counters["err_classes"][res.get("err_class", "?")] += 1
            continue
        idx = parse_ranked(res.get("content", ""), len(kb))
        if idx is None:
            stats["api_failures"] += 1
            counters["parse_failures"] += 1
            continue
        preds: list[str] = []
        for i in idx:
            for fp in kb[i]["fix_patterns"]:
                if fp not in preds:
                    preds.append(fp)
            if len(preds) >= 3:
                break
        _tally(stats, preds[:3], ev.get("fix_pattern"))
    return stats | {"kb_size": len(kb)}


def _tally(stats: dict, preds: list[str], gt: Any) -> None:
    if preds:
        stats["predictions"] += 1
        stats["top1"] += int(preds[0] == gt)
        stats["top3"] += int(gt in preds)


# ---------- 方法 2：简化 ERL（逐事件 reflection → heuristic）----------

def erl_units(train: list[dict], limit: int) -> list[dict]:
    evs = seedable(train)[:limit] if limit else seedable(train)
    # key = event_id + 提示词内容指纹：内容变了就必须重新调用，不得复用陈旧规则
    return [{"key": f"{e['event_id']}::{_fp(event_brief(e))}",
             "fix_patterns": [e["fix_pattern"]], "ev": e} for e in evs]


def erl_prompt(u: dict) -> str:
    return (f"以下是一条工具调用失败后重试成功的记录：\n{event_brief(u['ev'])}\n\n"
            '请输出严格 JSON：{"rule": "<一句话可复用规则>", "scope_tool": "<适用工具名>"}')


def run_erl(caller, ckpt, train, test_errors, limit, counters) -> dict:
    kb = llm_extract(caller, ckpt, "erl", erl_units(train, limit), erl_prompt, counters)
    return llm_retrieve(caller, ckpt, "erl", kb, test_errors, counters)


# ---------- 方法 3：ExpeL 式（跨会话聚合 → insight）----------

def expel_units(train: list[dict], limit: int) -> list[dict]:
    """按 session 聚合 —— 这是 ExpeL「跨会话 insight」的最小忠实实现。"""
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for e in seedable(train):
        by_sess[str(e.get("session_id", ""))].append(e)
    keys = sorted(by_sess)
    if limit:
        keys = keys[:limit]
    units = []
    for s in keys:
        evs = by_sess[s]
        # insight 携带该会话内**去重后**的 fix_pattern 列表（保持时间顺序）
        fps: list[str] = []
        for e in evs:
            if e["fix_pattern"] not in fps:
                fps.append(e["fix_pattern"])
        # ⚠️ key 必须用**完整** session_id + 内容指纹。原写 `sess_{s[:24]}`：实测有 6 个
        # 长 98 字符的 session_id 共享前 24 字符前缀，导致 5 个单元命中彼此的缓存，
        # kb 里出现「A 会话的规则 + B 会话的 fix_patterns」错配（污染 ExpeL 一方指标）。
        # 指纹取的材料与 expel_prompt 完全一致（前 8 条 brief + 条数）。
        body = f"{len(evs)}|" + "|".join(event_brief(x) for x in evs[:8])
        units.append({"key": f"sess_{s}::{_fp(body)}", "fix_patterns": fps, "evs": evs})
    return units


def expel_prompt(u: dict) -> str:
    body = "\n\n".join(f"({i + 1}) {event_brief(e)}" for i, e in enumerate(u["evs"][:8]))
    return (f"以下是**同一会话内**的 {len(u['evs'])} 条工具失败记录：\n{body}\n\n"
            "请归纳一条跨事件的一般化经验（而非复述单条）。"
            '输出严格 JSON：{"rule": "<一句话一般化规则>", "scope_tool": "<主要适用工具名>"}')


def run_expel(caller, ckpt, train, test_errors, limit, counters) -> dict:
    kb = llm_extract(caller, ckpt, "expel", expel_units(train, limit), expel_prompt, counters)
    return llm_retrieve(caller, ckpt, "expel", kb, test_errors, counters)


# ══════════════════════════════════════════════════════════════════════════
# 四、指标汇总（双口径）
# ══════════════════════════════════════════════════════════════════════════

def summarize(stats: dict, n_test_errors: int) -> dict:
    """严格沿用 exp9 `summarize_stats` 的公式，另加双口径与失败披露。"""
    fails = stats.get("api_failures", 0)

    def rate(v: float, denom: int) -> float:
        return round(v / denom * 100, 2) if denom else 0.0

    fp = max(0, stats["predictions"] - stats["top3"])
    out = {
        # ---- strict：分母恒为 n_test_errors，与 exp9/§6.4 同口径可比 ----
        "recall_at_1": rate(stats["top1"], n_test_errors),
        "recall_at_3": rate(stats["top3"], n_test_errors),
        "precision": round(stats["top1"] / stats["predictions"] * 100, 2) if stats["predictions"] else 0.0,
        "coverage_rate": rate(stats["predictions"], n_test_errors),
        "prevention_rate": rate(stats["top3"], n_test_errors),
        "false_positive_rate": rate(fp, n_test_errors),
        "avg_latency_ms": round(sum(stats["latency_ms"]) / len(stats["latency_ms"]), 4)
                          if stats["latency_ms"] else 0.0,
        # ---- 失败披露（绝不静默）----
        "api_failures": fails,
        "n_test_errors": n_test_errors,
        "kb_size": stats.get("kb_size", 0),
        "llm_calls": stats.get("llm_calls", 0),
        "top1_hits": stats["top1"], "top3_hits": stats["top3"],
        "predictions": stats["predictions"],
    }
    valid = n_test_errors - fails
    out["valid_only_denominator"] = valid
    out["valid_only_recall_at_1"] = rate(stats["top1"], valid)
    out["valid_only_prevention_rate"] = rate(stats["top3"], valid)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 五、负对照 / 正对照
# ══════════════════════════════════════════════════════════════════════════

def run_controls(say) -> int:
    """返回失败数。这些对照证明判据**能主动失败**，而非恒真。

    ⚠️ 必须走调用方的 `say`（同时进控制台与报告文件）。首版把对照只 append
    到报告列表，控制台只看到一个 `失败=0` 的汇总数字 —— 那就是**空转**：
    无法从输出确认对照真的跑了、真的有检测力。
    """
    fails = 0

    def chk(label: str, got, want) -> None:
        nonlocal fails
        ok = got == want
        fails += 0 if ok else 1
        say(f"  [{'ok   ' if ok else 'BLOCK'}] {label:56s} got={got!r} want={want!r}")

    say("")
    say("═══ 一、负对照：错误分类器（否则「0 限流」只是没测出来）═══")

    class E429(Exception):
        status_code = 429

    class E500(Exception):
        status_code = 500

    chk("合成 429 → RATE_LIMIT（可重试）", classify(E429("429 rate limit")), "RATE_LIMIT")
    chk("合成 500 → SERVER_ERROR（可重试）", classify(E500("500 internal")), "SERVER_ERROR")
    chk("合成 404 → MODEL_NOT_FOUND（不可重试）",
        classify(Exception("Error code: 404 - not found the model")), "MODEL_NOT_FOUND")
    chk("合成 invalid temperature → TEMP_UNSUPPORTED",
        classify(Exception("Error code: 400 - invalid temperature")), "TEMP_UNSUPPORTED")
    chk("合成 401 → AUTH", classify(Exception("Error code: 401 - incorrect api key")), "AUTH")
    chk("RATE_LIMIT 在可重试集合内", "RATE_LIMIT" in RETRYABLE, True)
    chk("MODEL_NOT_FOUND 不在可重试集合内（立即放弃）", "MODEL_NOT_FOUND" in RETRYABLE, False)

    say("")
    say("═══ 二、负对照：排序解析器（自由文本必须被拒，索引越界必须被丢）═══")
    chk("合法 JSON", parse_ranked('{"ranked": [2, 0, 1]}', 5), [2, 0, 1])
    chk("代码围栏", parse_ranked('```json\n{"ranked": [1]}\n```', 5), [1])
    chk("散文被拒（返回 None 而非空答案）", parse_ranked("我认为选第 2 条", 5), None)
    chk("越界索引被丢弃", parse_ranked('{"ranked": [9, 1]}', 5), [1])
    chk("重复索引去重", parse_ranked('{"ranked": [1, 1, 2]}', 5), [1, 2])
    chk("空列表 → None", parse_ranked('{"ranked": []}', 5), None)
    chk("全部越界 → None", parse_ranked('{"ranked": [7, 8]}', 5), None)

    say("")
    say("═══ 三、负对照：insight 解析器 ═══")
    chk("合法", _parse_insight('{"rule": "重试前规范化城市名", "scope_tool": "weather"}')
        is not None, True)
    chk("缺 rule 被拒", _parse_insight('{"scope_tool": "x"}'), None)
    chk("空 rule 被拒", _parse_insight('{"rule": "  "}'), None)
    chk("非 JSON 被拒", _parse_insight("规则是……"), None)

    say("")
    say("═══ 四、负对照：API 失败绝不污染指标（本脚本的核心效度保障）═══")
    # 模拟 10 个 test 事件，其中 4 个 API 失败、6 个成功且命中 3 个
    fake = {"top1": 3, "top3": 4, "predictions": 6, "latency_ms": [1.0] * 6,
            "api_failures": 4, "kb_size": 5, "llm_calls": 10}
    s = summarize(fake, 10)
    chk("strict 分母恒为 N（失败算 miss）", s["n_test_errors"], 10)
    chk("strict recall@1 = 3/10 = 30%", s["recall_at_1"], 30.0)
    chk("valid_only 分母 = N - failures = 6", s["valid_only_denominator"], 6)
    chk("valid_only recall@1 = 3/6 = 50%", s["valid_only_recall_at_1"], 50.0)
    chk("api_failures 被显式披露", s["api_failures"], 4)
    chk("precision 只按真实应答计算 = 3/6", s["precision"], 50.0)
    # 反例：若沿用 canonical 的 __ERROR__→空答案模式，失败会混入 predictions
    bad = dict(fake, predictions=10, top1=3, top3=4)
    chk("反例·失败被当空答案时 precision 被拉低到 30%", summarize(bad, 10)["precision"], 30.0)
    # 全失败不得产生 NaN / ZeroDivision
    zero = {"top1": 0, "top3": 0, "predictions": 0, "latency_ms": [],
            "api_failures": 10, "kb_size": 0, "llm_calls": 10}
    sz = summarize(zero, 10)
    chk("全失败·strict 指标为 0 而非 NaN", sz["recall_at_1"], 0.0)
    chk("全失败·valid_only 分母为 0 且不崩", sz["valid_only_denominator"], 0)
    chk("全失败·valid_only recall 为 0（除零已防护）", sz["valid_only_recall_at_1"], 0.0)

    say("")
    say("═══ 五、负对照：dry-run 调用器绝不发网络请求 ═══")
    d = make_caller("x", dry=True)([{"role": "user", "content": "hi"}])
    chk("dry-run 返回 ok=False", d["ok"], False)
    chk("dry-run 错误类为 DRY_RUN", d["err_class"], "DRY_RUN")
    chk("dry-run attempts=0（未尝试网络）", d["attempts"], 0)

    say("")
    say("═══ 六、正对照：summarize 必须能从 §6.4 六数字反推出整数计数 ═══")
    # §6.4：recall@1=71.19% recall@3=76.27% precision=89.36% coverage=79.66% FP=3.39%
    # 反推：N=59 ⇒ top1=42, top3=45, predictions=47
    pub = {"top1": 42, "top3": 45, "predictions": 47, "latency_ms": [0.5] * 47,
           "api_failures": 0, "kb_size": 137, "llm_calls": 0}
    sp = summarize(pub, 59)
    for k, v in PUBLISHED_EBEAC.items():
        chk(f"§6.4 反推复现 {k}", sp[k], v)

    say("")
    say("═══ 七、负对照：kb 指纹必须让陈旧缓存失效（防跨 limit 的索引污染）═══")

    def _ct() -> dict:
        return {"llm_calls": 0, "cache_hits": 0, "in_tok": 0, "out_tok": 0,
                "extract_failures": 0, "parse_failures": 0, "err_classes": Counter()}

    ev1 = [{"event_id": "nc_e1", "fix_pattern": "fp0"}]
    kb_small = [{"scope_tool": "t", "rule": "r0", "fix_patterns": ["fp0"]}]
    kb_big = kb_small + [{"scope_tool": "t", "rule": f"r{i}", "fix_patterns": [f"fp{i}"]}
                         for i in range(1, 5)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nc.jsonl"
        c1, ct1 = Checkpoint(p, resume=False), _ct()
        llm_retrieve(make_caller("x", dry=True), c1, "erl", kb_small, ev1, ct1)
        c1.close()
        chk("首轮真调用并写入缓存（llm_calls=1）", ct1["llm_calls"], 1)
        c2, ct2 = Checkpoint(p, resume=True), _ct()
        llm_retrieve(make_caller("x", dry=True), c2, "erl", kb_big, ev1, ct2)
        c2.close()
        chk("kb 变大后旧缓存失效（cache_hits=0）", ct2["cache_hits"], 0)
        chk("kb 变大后重新调用（llm_calls=1）", ct2["llm_calls"], 1)
        c3, ct3 = Checkpoint(p, resume=True), _ct()
        llm_retrieve(make_caller("x", dry=True), c3, "erl", kb_big, ev1, ct3)
        c3.close()
        chk("kb 不变时仍命中（防「永远 miss」式假修复）", ct3["cache_hits"], 1)
        chk("kb 不变时零新调用（llm_calls=0）", ct3["llm_calls"], 0)

    say("")
    say("═══ 八、负对照：提取单元 key 必须无碰撞且随内容变化（防跨会话规则错配）═══")
    # 用**实测的真实碰撞形态**造合成样本：3 个 session_id 共享前 24 字符、仅尾部时间戳不同
    PFX = "pwa_68e2619e-a46d-42d5-a"
    TAILS = ("1f1_remote_1780150057811", "1f1_remote_1780509711464", "1f1_remote_1780984105063")
    evs3 = [{"event_id": f"nc_ev{i}", "session_id": PFX + t, "status": "error",
             "fix_pattern": f"nc_fp{i}", "tool_name": "t", "action": "a",
             "error_type": "e", "metadata": {"query": f"q{i}", "diagnosis": f"d{i}"}}
            for i, t in enumerate(TAILS)]
    chk("旧 key（截断 24）确实碰撞 —— 证明该缺陷真实存在而非臆想",
        len({f"sess_{e['session_id'][:24]}" for e in evs3}), 1)
    chk("新 key 下三个同前缀会话互不碰撞", len({u["key"] for u in expel_units(evs3, 0)}), 3)
    chk("erl 三个事件亦互不碰撞", len({u["key"] for u in erl_units(evs3, 0)}), 3)
    chk("expel key 内容变化 ⇒ key 变化（指纹生效）",
        {u["key"] for u in expel_units(evs3[:1], 0)}
        == {u["key"] for u in expel_units([dict(evs3[0], fix_pattern="nc_fpX")], 0)}, False)
    chk("expel key 内容不变 ⇒ key 不变（缓存仍可用，非「永远 miss」）",
        [u["key"] for u in expel_units(evs3, 0)] == [u["key"] for u in expel_units(evs3, 0)], True)
    chk("erl key 内容变化 ⇒ key 变化",
        erl_units(evs3[:1], 0)[0]["key"]
        == erl_units([dict(evs3[0], fix_pattern="nc_fpX")], 0)[0]["key"], False)
    chk("erl key 内容不变 ⇒ key 不变",
        erl_units(evs3[:1], 0)[0]["key"] == erl_units(evs3[:1], 0)[0]["key"], True)

    say(f"  负对照/正对照合计失败 = {fails}")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 六、主流程
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="零 API：只跑 EBEAC + Raw-Log + 全部对照，验证复现 §6.4")
    ap.add_argument("--methods", default="raw_log,ebeac,erl,expel",
                    help="逗号分隔；dry-run 下 LLM 方法自动跳过")
    ap.add_argument("--limit-extract", type=int, default=0, help="冒烟：提取阶段单元上限")
    ap.add_argument("--limit-test", type=int, default=0, help="冒烟：test 事件上限")
    ap.add_argument("--resume", action="store_true", help="从 checkpoint 断点续跑")
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)
        print(s)

    say("=" * 100)
    say("Experiment 12: 同轨迹四方对照（EBEAC / 简化 ERL / ExpeL 式 insight / Raw-Log）")
    say(f"  model={a.model}  dry_run={a.dry_run}  methods={a.methods}")
    say(f"  limit_extract={a.limit_extract or '全量'}  limit_test={a.limit_test or '全量'}")
    say("=" * 100)

    ctrl_fails = 0 if a.skip_controls else run_controls(say)

    # ---------------- 数据装载 ----------------
    say("")
    say("═══ 数据装载 ═══")
    ds_path = a.dataset if Path(a.dataset).exists() else None
    name, version, events = load_dataset_items(ds_path, BUILTIN_LOG_REPLAY_SAMPLE)
    say(f"  dataset={name}  version={version}  items={len(events)}")
    builtin = ds_path is None
    if builtin:
        say("  ⚠️ 使用内置 6 条样例（真实数据集不在预期路径）—— 仅用于结构自检")

    train = [e for e in events if e.get("split") == "train"]
    test = [e for e in events if e.get("split") == "test"]
    test_errors = [e for e in test if e.get("status") == "error"]
    if a.limit_test:
        test_errors = test_errors[:a.limit_test]
    n_te = len(test_errors)
    say(f"  train={len(train)}  test={len(test)}  n_test_errors(分母)={n_te}")
    say(f"  seedable(train)={len(seedable(train))}  "
        f"error_types={len(count_by_key([e for e in events if e.get('error_type')], 'error_type'))} 类")
    say(f"  sessions={len({e.get('session_id') for e in events})}")

    # 真实数据集下必须与 §6.4 的计数一致，否则同源可比性不成立。
    # ⚠️ n_test_errors 在 --limit-test 冒烟模式下**按上限折算**而非跳过：
    #   跳过 → 冒烟跑恒 BLOCK，无法验收 LLM 通路；
    #   免检 → 丢掉「限流器真的生效」这一层（取全量 59 也不会被发现）。
    # 折算后两种模式都有判据：全量要 59，冒烟要 limit_test。
    data_fails = 0
    if not builtin:
        for k, want in PUBLISHED_COUNTS.items():
            got = {"n_train": len(train), "n_test": len(test), "n_test_errors": n_te}[k]
            note = ""
            if k == "n_test_errors" and a.limit_test:
                want = min(want, a.limit_test)
                note = f"（冒烟折算：limit_test={a.limit_test}）"
            ok = got == want
            data_fails += 0 if ok else 1
            say(f"  [{'ok   ' if ok else 'BLOCK'}] {k} 与 §6.4 一致  got={got} want={want}{note}")

    # ---------------- EBEAC：直接复用 exp9 代码路径 ----------------
    results: dict[str, dict] = {}
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    if "ebeac" in methods:
        say("")
        say("═══ 方法 EBEAC（ours）：复用 exp9 `evaluate_experience_store_memory` ═══")
        t0 = time.perf_counter()
        # min_similarity=0.6 与 exp9 L351 完全一致
        summ, _det = evaluate_experience_store_memory(train, test_errors, min_similarity=0.6)
        el = time.perf_counter() - t0
        summ = dict(summ)
        summ.update({"api_failures": 0, "n_test_errors": n_te, "llm_calls": 0,
                     "wall_seconds": round(el, 3),
                     "code_path": "exp9_ebeac_log_replay.evaluate_experience_store_memory"})
        results["ebeac"] = summ
        say(f"  recall@1={summ['recall_at_1']}%  recall@3={summ['recall_at_3']}%  "
            f"precision={summ['precision']}%  prevention={summ['prevention_rate']}%  "
            f"coverage={summ['coverage_rate']}%  FP={summ['false_positive_rate']}%")
        say(f"  avg_latency={summ['avg_latency_ms']}ms  wall={el:.2f}s  LLM calls=0")
        if not builtin and not a.limit_test:
            mism = [k for k, v in PUBLISHED_EBEAC.items() if abs(summ[k] - v) > 0.005]
            ok = not mism
            data_fails += 0 if ok else 1
            say(f"  [{'ok   ' if ok else 'BLOCK'}] 精确复现 §6.4 六个已发表数字"
                f"{'' if ok else ' —— 不一致项: ' + str(mism)}")

    # ---------------- Raw Error Log ----------------
    if "raw_log" in methods:
        say("")
        say("═══ 方法 Raw Error Log（零抽象基线，原 Table 3 该行的可复算实现）═══")
        st = run_raw_log(train, test_errors)
        results["raw_log"] = summarize(st, n_te) | {"llm_calls": 0, "wall_seconds": 0.0}
        r = results["raw_log"]
        say(f"  kb_size={r['kb_size']}  recall@1={r['recall_at_1']}%  "
            f"recall@3={r['recall_at_3']}%  prevention={r['prevention_rate']}%  "
            f"precision={r['precision']}%")
        say(f"  ⚠️ 原 Table 3 该行写 4 / 7.0% / 5.3%（全仓零出处）；本次可复算值见上")

    # ---------------- LLM 方法 ----------------
    counters = {"llm_calls": 0, "cache_hits": 0, "in_tok": 0, "out_tok": 0,
                "extract_failures": 0, "parse_failures": 0, "err_classes": Counter()}
    llm_methods = [m for m in methods if m in ("erl", "expel")]
    if llm_methods and not a.dry_run:
        ckpt = Checkpoint(CHECKPOINT_PATH, a.resume)
        caller = make_caller(a.model, dry=False)
        try:
            for m in llm_methods:
                say("")
                say(f"═══ 方法 {m}（our simplified re-implementation）═══")
                t0 = time.perf_counter()
                # ⚠️ counters 是**跨方法累计**的共享字典，直接取值会让后跑的方法
                # 记上前面方法的调用数（实测 erl=10、expel=20，而合计正是 20）。
                # 必须记**差值**，否则 Table 3b 的 LLM-calls 列会谎报成本。
                calls_before = counters["llm_calls"]
                st = (run_erl(caller, ckpt, train, test_errors, a.limit_extract, counters)
                      if m == "erl" else
                      run_expel(caller, ckpt, train, test_errors, a.limit_extract, counters))
                el = time.perf_counter() - t0
                st["llm_calls"] = counters["llm_calls"] - calls_before
                results[m] = summarize(st, n_te) | {"wall_seconds": round(el, 3)}
                r = results[m]
                say(f"  kb_size={r['kb_size']}  recall@1={r['recall_at_1']}%  "
                    f"recall@3={r['recall_at_3']}%  prevention={r['prevention_rate']}%")
                say(f"  precision={r['precision']}%  coverage={r['coverage_rate']}%  "
                    f"api_failures={r['api_failures']}  wall={el:.1f}s")
                if r["api_failures"]:
                    say(f"  🔴 有 API 失败 ⇒ 同时看 valid_only 口径: "
                        f"denom={r['valid_only_denominator']} "
                        f"recall@1={r['valid_only_recall_at_1']}% "
                        f"prevention={r['valid_only_prevention_rate']}%")
        finally:
            ckpt.close()
        say("")
        say(f"  LLM 调用合计={counters['llm_calls']}（checkpoint 命中 {counters['cache_hits']}）"
            f"  token in/out={counters['in_tok']}/{counters['out_tok']}")
        say(f"  提取失败={counters['extract_failures']}  解析失败={counters['parse_failures']}  "
            f"错误分类={dict(counters['err_classes'])}")
        # 归属不变量：各方法 llm_calls 之和必须等于累计总数。
        # 检测力已由修复前的**真实冒烟跑**实证：erl=10 + expel=20 = 30 ≠ 合计 20 ⇒ 必 BLOCK。
        per_sum = sum(results[m].get("llm_calls", 0) for m in llm_methods if m in results)
        ok = per_sum == counters["llm_calls"]
        data_fails += 0 if ok else 1
        say(f"  [{'ok   ' if ok else 'BLOCK'}] 各方法 llm_calls 之和 == 累计总数  "
            f"got={per_sum} want={counters['llm_calls']}")
    elif llm_methods and a.dry_run:
        say("")
        say(f"═══ dry-run：跳过 LLM 方法 {llm_methods}（零 API）═══")

    # ---------------- 汇总 ----------------
    say("")
    say("═" * 100)
    say("四方对照汇总（同一 196 条真实轨迹、同一分母、同一 fix_pattern 精确匹配判据）")
    say("═" * 100)
    hdr = f"{'method':<12}{'LLM':>5}{'kb':>6}{'R@1':>8}{'R@3':>8}{'Prec':>8}{'Prev':>8}{'Cov':>8}{'FP':>7}{'fail':>6}"
    say(hdr)
    say("-" * len(hdr))
    for m in ("raw_log", "erl", "expel", "ebeac"):
        if m not in results:
            continue
        r = results[m]
        say(f"{m:<12}{r.get('llm_calls', 0):>5}{r.get('kb_size', 0):>6}"
            f"{r['recall_at_1']:>8.2f}{r['recall_at_3']:>8.2f}{r['precision']:>8.2f}"
            f"{r['prevention_rate']:>8.2f}{r['coverage_rate']:>8.2f}"
            f"{r['false_positive_rate']:>7.2f}{r.get('api_failures', 0):>6}")

    payload = {
        "experiment": "exp12_same_traces_baselines",
        "status": "dry_run" if a.dry_run else "evaluated",
        "dataset": name, "dataset_version": version,
        "backend": {"model": a.model, "base_url": BASE_URL, "temperature": TEMPERATURE,
                    "max_retries": MAX_RETRIES, "sdk_max_retries": 0,
                    "selection_basis": "全模型稳定性探测：排除限流/下架/不支持 temperature=0 的模型"},
        "protocol": {
            "n_train": len(train), "n_test": len(test), "n_test_errors": n_te,
            "ground_truth_field": "fix_pattern", "match_rule": "exact string equality",
            "top_k": 3, "metric_formulas": "identical to exp9 summarize_stats",
            "dual_reporting": ["strict (denominator = n_test_errors)",
                               "valid_only (denominator = n_test_errors - api_failures)"],
            "error_policy": "API failure is NEVER converted into a wrong answer",
        },
        "counters": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in counters.items()},
        "results": results,
        "published_reference": {"section": "6.4 (E13)", **PUBLISHED_EBEAC, **PUBLISHED_COUNTS},
        "control_failures": ctrl_fails,
        "data_failures": data_fails,
    }
    if not a.dry_run or results:
        write_json(a.output, payload)
        say("")
        say(f"Saved output file: {Path(a.output).name}")

    report = OUTPUT_DIR / "exp12_same_traces_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    say(f"Saved report: {report.name}")

    total_fails = ctrl_fails + data_fails
    say("")
    say("=" * 100)
    say(f"结论: 对照失败={ctrl_fails}  数据一致性失败={data_fails}  合计={total_fails}")
    say("=" * 100)
    return 1 if total_fails else 0


if __name__ == "__main__":
    sys.exit(main())
