#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 4 (`tab:ablation_rcr`) 的独立复算证据 —— 填补 B1-2「14.6 → zero 无复算证据」缺口。

背景
----
批次一 B1-2（回应 R1-7）只更正了 Table 4 的**口径**（caption 明确 API Errors =
「mean number of HTTP-400 responses per conversation caused by orphaned tool calls」），
但 `14.6 → zero` 这个**数值本身**在批次二没有独立复算证据：`analysis_stats_revision.py`
与其报告对 `HTTP-400`／`Table 4`／`Watchdog` 全部零命中（见 REVISION_LOG §一 B1-2、§七）。

本脚本从冻结的原始数据 `raw_data/exp3_results.json`（`exp3_rcr_pipeline.py` 的确定性输出：
`RANDOM_SEED=42`、30 场合成对话、`Pure code measurement — no LLM calls`）**独立复算**
Table 4 的每一个单元格，并做四方交叉验证：

    逐对话原始列表  →  复算汇总  →  exp3_summary.json  →  两稿 tex Table 4 + 邻近正文

第四方（tex 交叉验证）只在**论文工作区**可用：`array_submission/` 是投稿源目录，其中
`paper_array.tex` 带作者身份，不随 artifact bundle 发布（发布即破双盲）。故本脚本按
`array_submission/` 的存在性自动选择运行模式（paper／bundle／broken，见下方 `TEX_MODE`）：
bundle 模式下 S3/S4 显式标 SKIPPED 而非静默略过，S1/S2 的数据复算与 S5 的负对照
（`cell_ok` 检测力证据）照常全量执行 —— 审稿人在克隆内一跑仍得确定性的 EXIT=0 与
逐字节可重现的报告，且不损失对「检查器空转」的防护。

诚实边界（写入报告，与论文 caption 口径一致）
------------------------------------------
Table 4 的「API Errors」是 exp3 在**合成对话**中按 `ORPHAN_RATE=0.09` 程序化注入的
**orphan tool-call 计数**（一个 orphan call 发给真实 OpenAI 兼容端点必然触发一次
HTTP-400，故为**确定性代理**），**不是**从生产 API 日志统计得到的 400 响应数。
`14.6 = 439 个注入 orphan / 30 场对话`。RCR 的 S1（`validate_message_structure`）为
每个 orphan 补 placeholder result，故 S4 起降到 0；LLMLingua 按困惑度删消息、破坏
call/result 配对，反升到 53.9。

带负对照（`artifact_registry.md` §7.7 教训 7）：合成偏离值，断言对照逻辑能抓到，
证明「全部吻合」不是检查器空转。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent                 # .../benchmarks
RAW = HERE / "raw_data"
SUB = HERE.parent / "array_submission"
ARCH = SUB / "paper_array.tex"
ANON = SUB / "paper_anonymous" / "paper_array_anonymous.tex"
REPORT = RAW / "exp3_table4_recompute_report.txt"

# tex 交叉校验（S3/S4）需要论文投稿源目录 array_submission/。该目录**不属于** artifact
# bundle —— 匿名仓只发布代码与冻结数据，不发布投稿源文件（paper_array.tex 含作者身份，
# 发布即破双盲）。故 bundle 内运行时 S3/S4 优雅自跳过，并在报告中显式标 SKIPPED。
# 守卫分三级，确保「跳过」不退化成「静默失效」（artifact_registry §7.7 教训 7）：
#   paper  —— array_submission/ 与两稿 tex 均在场 → S3/S4 全量执行（论文工作区）
#   bundle —— array_submission/ 整个目录不存在   → S3/S4 SKIP，不计 fail（bundle 语境）
#   broken —— 目录在场但 tex 缺失               → S3/S4 FAIL，不得降级为 SKIP
if not SUB.is_dir():
    TEX_MODE = "bundle"
elif ARCH.is_file() and ANON.is_file():
    TEX_MODE = "paper"
else:
    TEX_MODE = "broken"

out: list[str] = []
say = out.append
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    say(("  [ok  ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def skip(msg: str) -> None:
    """记录一次「已知不可用故不检查」—— 不进 fails，但在报告里留痕，禁止静默。"""
    say("  [skip] " + msg)


def tex_guard(section: str) -> bool:
    """True = tex 在场，该节全量执行；False = 已就地记录 skip（bundle）或 fail（broken）。"""
    if TEX_MODE == "paper":
        return True
    if TEX_MODE == "broken":
        check(False, f"{section}: array_submission/ 在场但两稿 tex 缺失 —— 论文工作区被破坏，"
                     f"不得降级为 SKIP")
    else:
        skip(f"{section}: array_submission/ 不在 artifact bundle 内 —— tex 交叉对照 SKIPPED"
             f"（S1/S2 数据复算与 S5 负对照照常执行，检测力未降低）")
    return False


# --------------------------------------------------------------------------- #
# 载入冻结原始数据
# --------------------------------------------------------------------------- #
res = json.loads((RAW / "exp3_results.json").read_text(encoding="utf-8"))
summ = json.loads((RAW / "exp3_summary.json").read_text(encoding="utf-8"))
meta = res["metadata"]
per = res["per_conversation"]
N = meta["num_conversations"]

CFG = ["raw", "pruning", "s1_s4", "s1_s5", "full_rcr", "llmlingua"]
# (tex 列名, results 里的逐对话列表键, summary 里的汇总键, 是否百分比列)
COLS = [("API Errors", "api_errors", "api_errors_per_conv", False),
        ("Jitter", "jitter", "jitter_events_per_conv", False),
        ("Prefix Hit", "prefix_hit", "prefix_hit_rate", True),
        ("Savings", "savings", "compression_savings", False)]


def mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# S1　逐对话原始列表 → 复算均值 → 对照 exp3_summary.json（不信 summary，独立重算）
# --------------------------------------------------------------------------- #
say("=" * 88)
say("S1　逐对话原始列表 → 复算均值 → 对照 exp3_summary.json")
say("=" * 88)
recomputed: dict[str, dict[str, float]] = {}
for cfg in CFG:
    recomputed[cfg] = {}
    for colname, listkey, summkey, is_pct in COLS:
        xs = per[cfg][listkey]
        check(len(xs) == N, f"{cfg}.{listkey} 列表长度 = {len(xs)}（应 = {N} 场对话）")
        val = round(mean(xs) * 100, 0) if is_pct else round(mean(xs), 1)
        recomputed[cfg][summkey] = val
        sv = summ[cfg][summkey]
        check(abs(val - sv) < 1e-9, f"{cfg}.{summkey}: 复算 {val} == summary {sv}")

# --------------------------------------------------------------------------- #
# S2　原始数据自洽（orphan 守恒 + RCR 修复语义）
# --------------------------------------------------------------------------- #
say("")
say("=" * 88)
say("S2　原始数据自洽（orphan 守恒 + RCR 修复语义）")
say("=" * 88)
raw_sum = sum(per["raw"]["api_errors"])
prun_sum = sum(per["pruning"]["api_errors"])
check(meta["total_orphans"] == raw_sum,
      f"metadata.total_orphans {meta['total_orphans']} == sum(raw.api_errors) {raw_sum}")
check(raw_sum == prun_sum,
      f"sum(raw) {raw_sum} == sum(pruning) {prun_sum}（S3 剪枝不动 call/result 配对）")
check(per["raw"]["api_errors"] == per["pruning"]["api_errors"],
      "raw 与 pruning 的逐对话 api_errors 列表完全相同")
for cfg in ("s1_s4", "s1_s5", "full_rcr"):
    check(all(v == 0 for v in per[cfg]["api_errors"]),
          f"{cfg}.api_errors 全 0（S1 validate_message_structure 为每个 orphan 补 placeholder）")
check(round(raw_sum / N, 1) == 14.6,
      f"14.6 = {raw_sum} orphans / {N} conv = {raw_sum / N:.4f} → round(_,1) = 14.6")
check(round(sum(per["llmlingua"]["api_errors"]) / N, 1) == 53.9,
      f"53.9 = sum(llmlingua.api_errors) {sum(per['llmlingua']['api_errors'])} / {N} → 53.9")

# --------------------------------------------------------------------------- #
# S3　两稿 tex Table 4 逐格对照（含两稿镜像一致）
# --------------------------------------------------------------------------- #
say("")
say("=" * 88)
say("S3　两稿 tex Table 4 (tab:ablation_rcr) 逐格对照复算值")
say("=" * 88)
PREFIX = [("No pipeline", "raw"), ("Pruning only", "pruning"),
          ("S1--S3 + S4", "s1_s4"), ("S1--S5", "s1_s5"),
          ("Full RCR", "full_rcr"), ("LLMLingua", "llmlingua")]


def parse_cell(s: str) -> tuple[str, float | None]:
    s = s.strip()
    if s in ("---", "--", "—"):
        return ("dash", None)
    s = s.replace(r"\%", "").replace("%", "").strip()
    return ("num", float(s))


LABEL = r"\label{tab:ablation_rcr}"


def tabular_window(lines: list[str]) -> tuple[int, int] | None:
    r"""定位 `tab:ablation_rcr` 所在 tabular 环境的行窗口 [bs, be)（0-based）。

    **必须限定作用域**：相关工作对比表（R2-7/R2-15）含行首
    `LLMLingua~\cite{jiang2023llmlingua}`，与 Table 4 的行名前缀**撞车**；旧实现
    全文扫描且后写覆盖前写，会取到对比表的 `$\times$` 单元格并在 `parse_cell`
    里抛 ValueError。返回 None 表示未找到（调用方须报错，不得退而猜表）。
    """
    li = next((i for i, s in enumerate(lines) if LABEL in s), None)
    if li is None:
        return None
    bs = next((i for i in range(li, len(lines))
               if lines[i].strip().startswith(r"\begin{tabular}")), None)
    if bs is None:
        return None
    be = next((i for i in range(bs + 1, len(lines))
               if lines[i].strip().startswith(r"\end{tabular}")), None)
    if be is None:
        return None
    return (bs, be)


def _scan_rows(lines: list[str]) -> dict[str, list[str]]:
    """在给定的行序列里按行名前缀抽数据行（不做作用域判断）。"""
    rows: dict[str, list[str]] = {}
    for line in lines:
        s = line.strip()
        if not s.endswith(r"\\") or "&" not in s:
            continue
        cells = [c.strip() for c in s[:-2].split("&")]
        if len(cells) != 5:
            continue
        name = cells[0]
        for pref, key in PREFIX:
            if name.startswith(pref):
                rows[key] = cells[1:]
                break
    return rows


def parse_table4(tex: str) -> dict[str, list[str]]:
    """提取 tab:ablation_rcr 的 6 个数据行 → {cfg_key: [4 个原始单元格串]}（**仅限该表作用域**）。"""
    lines = tex.splitlines()
    win = tabular_window(lines)
    if win is None:
        return {}
    bs, be = win
    return _scan_rows(lines[bs:be])


def table4_found(tex: str) -> bool:
    """两稿是否真能定位到 tab:ablation_rcr（定位失败要报 FAIL，不能静默得空字典）。"""
    return tabular_window(tex.splitlines()) is not None


def cell_ok(kind: str, val: float | None, recomp: float) -> bool:
    """tex 单元格与复算值是否一致。`---` 表示该值为 0（排版省略）。"""
    if kind == "dash":
        return abs(recomp) < 0.05
    return abs((val or 0.0) - recomp) < 0.05


tex_arch = tex_anon = ""
n_cells = 0
if tex_guard("S3"):
    tex_arch = ARCH.read_text(encoding="utf-8")
    tex_anon = ANON.read_text(encoding="utf-8")
    t4_arch = parse_table4(tex_arch)
    t4_anon = parse_table4(tex_anon)

    check(table4_found(tex_arch), "ARCH 定位到 tab:ablation_rcr 的 tabular 作用域")
    check(table4_found(tex_anon), "ANON 定位到 tab:ablation_rcr 的 tabular 作用域")
    check(set(t4_arch) == set(CFG), f"ARCH Table 4 解析出 {len(t4_arch)} 行 = {sorted(t4_arch)}")
    check(set(t4_anon) == set(CFG), f"ANON Table 4 解析出 {len(t4_anon)} 行 = {sorted(t4_anon)}")
    check(t4_arch == t4_anon, "两稿 Table 4 的 6 行 × 4 列逐格镜像一致")

    for cfg in CFG:
        for i, (colname, _, summkey, _) in enumerate(COLS):
            cell = t4_arch[cfg][i]
            kind, val = parse_cell(cell)
            recomp = recomputed[cfg][summkey]
            n_cells += 1
            check(cell_ok(kind, val, recomp),
                  f"[{cfg}] {colname}: tex「{cell}」 vs 复算 {recomp}")

# --------------------------------------------------------------------------- #
# S4　邻近正文数值对照
# --------------------------------------------------------------------------- #
say("")
say("=" * 88)
say("S4　Table 4 邻近正文数值对照（两稿）")
say("=" * 88)


def prose_line(tex: str) -> str:
    for ln in tex.splitlines():
        if "eliminates all API errors" in ln:
            return ln
    return ""


PROSE_PATTERNS = [
    (r"from a mean of ([\d.]+) to zero", "api_errors_per_conv", "raw", "正文 14.6（raw→zero 起点）"),
    (r"mean of ([\d.]+) such errors", "api_errors_per_conv", "llmlingua", "正文 53.9（LLMLingua）"),
    (r"achieves a (\d+)\\% prompt-cache hit rate", "prefix_hit_rate", "full_rcr", "正文 60%（Full RCR prefix hit）"),
    (r"vs\.\\ (\d+)\\% for LLMLingua", "prefix_hit_rate", "llmlingua", "正文 13%（LLMLingua prefix hit）"),
    (r"Full RCR achieves ([\d.]+)\\% compression", "compression_savings", "full_rcr", "正文 48.5%（Full RCR savings）"),
    (r"pruning-only's ([\d.]+)\\%", "compression_savings", "pruning", "正文 49.9%（pruning-only savings）"),
]
if tex_guard("S4"):
    for tag, tex in (("ARCH", tex_arch), ("ANON", tex_anon)):
        pl = prose_line(tex)
        check(bool(pl), f"{tag} 定位到 Table 4 邻近正文行")
        for pat, summkey, cfg, desc in PROSE_PATTERNS:
            m = re.search(pat, pl)
            check(bool(m), f"{tag} 正文匹配 {desc}")
            if m:
                tv = float(m.group(1))
                check(abs(tv - recomputed[cfg][summkey]) < 0.05,
                      f"{tag} {desc}: tex {tv} == 复算 {recomputed[cfg][summkey]}")
# 正文 "to zero"：S4 起 api_errors 应为 0 —— 纯数据断言，与 tex 无关，bundle 内照常执行
check(recomputed["s1_s4"]["api_errors_per_conv"] == 0.0, "正文「to zero」: s1_s4 api_errors 复算 = 0.0")

# --------------------------------------------------------------------------- #
# S5　负对照（证明 cell_ok / 正文对照有检测力，非空转）
# --------------------------------------------------------------------------- #
say("")
say("=" * 88)
say("S5　负对照（合成偏离值 + 作用域守卫，对照逻辑必须报错）")
say("=" * 88)
negs: list[str] = []


def neg(cond: bool, msg: str) -> None:
    """负对照专用：计数以便结论行自证「非空转」，判据与 check 完全一致。"""
    negs.append(msg)
    check(cond, msg)


neg(not cell_ok("num", 14.7, 14.6), "负对照①: tex=14.7 vs 复算=14.6 → 判不一致（检测力在）")
neg(cell_ok("num", 14.6, 14.6), "负对照②: tex=14.6 vs 复算=14.6 → 判一致")
neg(not cell_ok("dash", None, 5.0), "负对照③: tex=--- 但复算=5.0(≠0) → 判不一致")
neg(cell_ok("dash", None, 0.0), "负对照④: tex=--- 且复算=0.0 → 判一致")
neg(not cell_ok("num", 60.0, 13.0), "负对照⑤: tex=60 vs 复算=13 → 判不一致")
# ⑥⑦ 针对本轮新发现的作用域撞车（对比表 L604 `LLMLingua~\cite{...}`）：
# 合成一份「表内一行 + 表外同名撞车行」的文档，守卫必须只取表内行。
_synth = "\n".join([
    r"\begin{table*}",
    r"\caption{Synthetic.\label{tab:ablation_rcr}}",
    r"\begin{tabular}{lcccc}",
    r"Full RCR (S1--S6) & 0 & 0 & 60\% & 48.5\%\\",
    r"\end{tabular}",
    r"\end{table*}",
    r"\begin{tabular}{lcccc}",
    r"LLMLingua~\cite{jiang2023llmlingua} & --- & --- & --- & $\times$\\",
    r"\end{tabular}",
])
_synth_rows = parse_table4(_synth)
neg(list(_synth_rows) == ["full_rcr"],
    f"负对照⑥: 作用域外同名撞车行未被取到（解析出 {list(_synth_rows)}，应 = ['full_rcr']）")
neg(parse_table4("no label here") == {},
    "负对照⑦: 无 tab:ablation_rcr 标签 → 解析出空字典（不猜表，交由 table4_found 报 FAIL）")

# --------------------------------------------------------------------------- #
# 结论 + 报告
# --------------------------------------------------------------------------- #
say("")
say("=" * 88)
if fails:
    say(f"结论: FAIL 计数 = {len(fails)}")
    for f in fails:
        say("   - " + f)
elif TEX_MODE == "paper":
    say(f"结论: ALL OK —— Table 4 全 {n_cells} 格 + 两稿正文各 {len(PROSE_PATTERNS)} 处，"
        f"均由 exp3_results.json 逐对话列表独立复算吻合；14.6 = {raw_sum}/{N}，S4 起 → 0。")
else:
    say(f"结论: ALL OK（bundle 模式）—— S1 全 {len(CFG) * len(COLS)} 格汇总值 + S2 orphan 守恒，"
        f"均由 exp3_results.json 逐对话列表独立复算吻合；14.6 = {raw_sum}/{N}，S4 起 → 0。"
        f"S3/S4 tex 交叉对照 SKIPPED（array_submission/ 不入 bundle），S5 负对照 {len(negs)}/{len(negs)} 仍有效。")
say("=" * 88)
say("")
say("运行模式: " + {
    "paper": "paper —— array_submission/ 与两稿 tex 在场，S3/S4 tex 交叉对照已全量执行",
    "bundle": "bundle —— array_submission/ 不在 artifact bundle 内，S3/S4 已 SKIPPED（非静默失效）",
    "broken": "broken —— array_submission/ 在场但 tex 缺失，已计 FAIL（不得降级为 SKIP）",
}[TEX_MODE])
say("诚实声明：API Errors 为**合成对话**中按 ORPHAN_RATE=0.09 注入的 orphan tool-call 计数")
say("（HTTP-400 的确定性代理），非生产 API 日志统计；exp3 纯离线、seed=42、可复现。")
say("更严格的 RCR 质量与变异性证据（≥60 长会话 + 失败注入 + mean±SD + LLM judge）属 E14。")

REPORT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("\n".join(out[-6:]))
print("report →", REPORT, "; fails =", len(fails))
sys.exit(1 if fails else 0)
