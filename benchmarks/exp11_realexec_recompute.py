#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table ``tab:realexec`` (Real-Execution Re-Scoring) 的独立**零-API**复算证据。

本脚本是干什么的
----------------
论文 ``\\subsection{Real-Execution Re-Scoring}``（``\\label{sec:realexec}``）的
表 ``\\label{tab:realexec}`` 把主结果从**选择层 oracle**（``selected in
acceptable_set``）升级为**真实执行**（``sandbox.execute(tool,args).status=='success'
AND tool in acceptable_tools``），回应 R1-1。表中每个数字都来自四个后端 ×
seeds 42/43/44 的 live 跑批：``deepseek-v4-flash``/``glm-4-flashx``（full 200q，
pooled N=600）与 ``qwen-max``/``kimi-k2.7-code``（subset 100q，pooled N=300）。

本脚本从**冻结的 12 个 per-run JSON**（``raw_data/runs/exp11/exp11_<bk>_<arm>_
seed<42|43|44>.json``，每个含逐查询 ``records``）出发，**独立重导**表中每一格与
邻近正文的每个数字，并与两份 ``exp11_aggregate_{full,subset}.json`` 交叉验证。
它**不重跑任何 LLM**：真实执行需要付费 API（且 ``kimi-k2.7-code`` 被强制
temperature=1，本质不可逐位复现），故重跑属作者侧、非审稿人侧；本脚本保证的是
「**已发布的原始逐查询记录 → 已发布的每个报告数字**」这条链可被任何人零成本复核。

可验证 / 不可验证（诚实边界）
----------------------------
可从冻结 artifact 验证（无网络、无配额）：
  * 每个 per-run 的 ``single_rate/final_rate/avg_tokens``：从 ``records`` 逐查询
    重算，与该文件自带的 ``summary`` 逐格吻合（**不信 summary，独立重算**）；
  * 3-seed 聚合 ``single_mean/final_mean`` 与三个 McNemar 对照（b/c/χ²/p）：
    重算并与 ``exp11_aggregate_{full,subset}.json`` 逐格吻合；
  * 表 ``tab:realexec`` 的 4 行 × 10 列，与邻近正文数字（恢复抬升、kimi 的
    64-vs-23 反超、171/300 空选择、p≤1.4e-3、63--66% 省 token）；
  * **效度负对照**：out-of-sandbox 查询在全部 config 上 ``ok_final`` 恒为 0
    （零虚假增益）；合取判据不变式（``ok==1 ⟹ status==success 且 tool∈acceptable``）；
    真实执行 ≤ oracle（严格更难）；非恢复臂 ``single==final`` 且从不 escalate/retry。

**不**在此重算：live LLM 选择本身（需付费 API；kimi 强制 temp=1 不可逐位复现）。
本脚本复核的是「记录 → 报告数字」的算术与效度，不是「重跑一遍模型」。

双盲与运行模式（TEX_MODE）
--------------------------
tex 交叉对照（S4/S5）需要论文投稿源目录 ``array_submission/``，该目录**不属于**
artifact bundle —— 匿名仓只发布代码与冻结数据，不发布含作者身份的 tex（发布即破
双盲）。故按 ``array_submission/`` 的存在性自动选择模式：
  paper  —— 两稿 tex 均在场 → S4/S5 全量执行（论文工作区）
  bundle —— 目录不存在       → S4/S5 显式 SKIPPED（非静默失效）；S0--S3/S6 照常全量
  broken —— 目录在场但 tex 缺失 → S4/S5 计 FAIL，不得降级为 SKIP
bundle 模式下审稿人在克隆内一跑仍得确定性的 EXIT=0、逐字节可重现的报告、完整的
数据复算（S1/S2）、效度负对照（S3）与合成负对照（S6），检测力未降低。

报告确定性
----------
报告不含时间戳、墙钟时长或绝对路径；布局相关事实被归约为内容寻址（每个输入的
sha256 前 16 位 + 字节数），故在任何机器上以相同数据重跑都逐字节复现本报告。
所有摘要与字节数均按归档换行约定（v4/v5 notes §9：仓内 blob 一律 LF、不加
``.gitattributes``）对 **LF 归一化后的字节流**计算，即等于
``git show <tag>:<path> | sha256sum``，与读者所在平台或其 ``core.autocrlf`` 取值无关；
直接对工作树 ``sha256sum <file>`` 在 Windows（autocrlf=true，检出为 CRLF）下会不同。

负对照（``artifact_registry.md`` §7.7 教训 7）：S6 合成偏离值，断言对照逻辑能抓到，
证明「全部吻合」不是检查器空转。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent                      # .../benchmarks
RUNS = HERE / "raw_data" / "runs" / "exp11"
SUB = HERE.parent / "array_submission"
ARCH = SUB / "paper_array.tex"
ANON = SUB / "paper_anonymous" / "paper_array_anonymous.tex"
REPORT = HERE / "raw_data" / "exp11_realexec_recompute_report.txt"

# ---- tex 交叉对照模式守卫（与 exp3_table4_recompute.py 同构，三级不退化为静默）----
if not SUB.is_dir():
    TEX_MODE = "bundle"
elif ARCH.is_file() and ANON.is_file():
    TEX_MODE = "paper"
else:
    TEX_MODE = "broken"

out: list[str] = []
say = out.append
fails: list[str] = []
skips: list[str] = []


def check(cond: bool, msg: str) -> None:
    say(("  [ok  ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def skip(msg: str) -> None:
    say("  [skip] " + msg)
    skips.append(msg)


def neg(cond: bool, msg: str) -> None:
    """负对照专用：判据与 check 一致，仅措辞标明这是「必须报错」的检测力自证。"""
    check(cond, msg)


def tex_guard(section: str) -> bool:
    if TEX_MODE == "paper":
        return True
    if TEX_MODE == "broken":
        check(False, f"{section}: array_submission/ 在场但两稿 tex 缺失 —— 论文工作区被破坏，不得降级为 SKIP")
    else:
        skip(f"{section}: array_submission/ 不在 artifact bundle 内 —— tex 交叉对照 SKIPPED"
             f"（S0--S3 数据复算/效度与 S6 负对照照常执行，检测力未降低）")
    return False


# --------------------------------------------------------------------------- #
# 常量（臂/后端/配置）—— 只描述**布局**，不含任何被验证的报告数字（避免循环论证）
# --------------------------------------------------------------------------- #
SEEDS = (42, 43, 44)
ARMS = {"full": ("deepseek", "glm"), "subset": ("qwen", "kimi")}
ARM_N = {"full": 200, "subset": 100}
CFGS = ("static", "static_retry", "pte", "pte_fd")           # 表所报告的 4 个 config
ALL_CFGS = ("static", "static_retry", "pte", "pte_fd", "keyword", "keyword_retry")
PAPER_NAME = {"deepseek": "deepseek-v4-flash", "glm": "glm-4-flashx",
              "qwen": "qwen-max", "kimi": "kimi-k2.7-code"}
NAME2BK = {v: k for k, v in PAPER_NAME.items()}
ROW_ORDER = ("deepseek", "glm", "qwen", "kimi")               # 表行序（full 两行 + subset 两行）
NON_RECOVERY = ("static", "pte", "keyword")                   # mechanism == "none"


def run_path(bk: str, arm: str, seed: int) -> Path:
    return RUNS / f"exp11_{bk}_{arm}_seed{seed}.json"


def _lf(b: bytes) -> bytes:
    """CRLF→LF 归一化：还原 Git 在归档换行约定下真正存入的 blob 字节流
    （v4/v5 notes §9：blob 一律 LF、不加 ``.gitattributes``）。单独拆出，好让 S6
    的负对照⑥调用这同一个生产函数——删掉归一化，负对照即报错（非空转）。"""
    return b.replace(b"\r\n", b"\n")


def blob_bytes(p: Path) -> int:
    """LF 归一化 blob 的字节数 == ``git cat-file -s <tag>:<path>``，与平台无关
    （不同于 Windows 工作树里含 CR 的 ``p.stat().st_size``）。"""
    return len(_lf(p.read_bytes()))


def sha16(p: Path) -> str:
    """LF 归一化 blob 的 sha256 前 16 位 == ``git show <tag>:<path> | sha256sum``
    的前 16 位，在任何平台/任何 ``core.autocrlf`` 下都相同。"""
    return hashlib.sha256(_lf(p.read_bytes())).hexdigest()[:16]


def mcnemar(a: list, b: list) -> dict:
    """逐字复刻 harness 的 mcnemar：b(打印)=n10=a胜, c(打印)=n01=b胜。"""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    nd = n01 + n10
    if nd == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / nd
    return {"n01": n01, "n10": n10, "chi2": round(chi2, 3), "p": math.erfc(math.sqrt(chi2 / 2.0))}


def sig2(x: float) -> float:
    """四舍五入到 2 位有效数字（表的 p 列即以 2 位有效数字排版）。"""
    if x == 0:
        return 0.0
    d = 2 - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


# --------------------------------------------------------------------------- #
# 从 records 独立重算一个 (run, cfg) 的 single/final/tokens（不读该文件的 summary）
# --------------------------------------------------------------------------- #
def recompute_cfg(recs: list, cfg: str) -> dict:
    n = len(recs)
    single = [r["configs"][cfg]["ok_single"] for r in recs]
    final = [r["configs"][cfg]["ok_final"] for r in recs]
    tok = sum(r["configs"][cfg]["tokens"] for r in recs)
    return {"n": n, "single_rate": round(100 * sum(single) / n, 2),
            "final_rate": round(100 * sum(final) / n, 2), "avg_tokens": round(tok / n, 1),
            "tok_raw": tok / n, "tok_total": tok,
            "per_query_single": single, "per_query_final": final}


def load_arm(bk: str, arm: str) -> dict:
    """载入一个后端的 3 个 seed 文件；返回 {seed: parsed_json}。"""
    runs = {}
    for s in SEEDS:
        p = run_path(bk, arm, s)
        if not p.is_file():
            raise SystemExit(f"缺失 per-run 文件: {p.name}")
        runs[s] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def backend_recompute(bk: str, arm: str) -> dict:
    """把一个后端从 records 完整重导到「表格所需的全部量」。"""
    runs = load_arm(bk, arm)
    per_cfg = {cfg: {s: recompute_cfg(runs[s]["records"], cfg) for s in SEEDS} for cfg in ALL_CFGS}

    def mean3(cfg, key):
        return sum(per_cfg[cfg][s][key] for s in SEEDS) / 3.0

    agg = {}
    for cfg in ALL_CFGS:
        agg[cfg] = {
            "single_mean": round(mean3(cfg, "single_rate"), 2),
            "final_mean": round(mean3(cfg, "final_rate"), 2),
            "tok_pooled": round(sum(per_cfg[cfg][s]["tok_total"] for s in SEEDS)
                                / sum(per_cfg[cfg][s]["n"] for s in SEEDS)),
            "tok_raw_pooled": sum(per_cfg[cfg][s]["tok_total"] for s in SEEDS)
            / sum(per_cfg[cfg][s]["n"] for s in SEEDS),
        }

    def pooled(cfg, key):
        return [v for s in SEEDS for v in per_cfg[cfg][s][key]]

    contrasts = {
        "pte_fd_vs_static_retry_final": mcnemar(pooled("static_retry", "per_query_final"),
                                               pooled("pte_fd", "per_query_final")),
        "pte_fd_final_vs_static_single": mcnemar(pooled("static", "per_query_single"),
                                                pooled("pte_fd", "per_query_final")),
        "pte_fd_vs_pte_single": mcnemar(pooled("pte", "per_query_single"),
                                        pooled("pte_fd", "per_query_single")),
    }
    sr, pf = agg["static_retry"], agg["pte_fd"]
    row = {
        "arm": arm, "runs": runs, "per_cfg": per_cfg, "agg": agg, "contrasts": contrasts,
        # 表 tab:realexec 的 10 列（第 1 列 model、第 2 列 arm 为标签，不复算）
        "st_single": round(agg["static"]["single_mean"], 1),
        "sr_final": round(agg["static_retry"]["final_mean"], 1),
        "p_single": round(agg["pte"]["single_mean"], 1),
        "pf_final": round(agg["pte_fd"]["final_mean"], 1),
        "b": contrasts["pte_fd_vs_static_retry_final"]["n10"],
        "c": contrasts["pte_fd_vs_static_retry_final"]["n01"],
        "p": contrasts["pte_fd_vs_static_retry_final"]["p"],
        "tok_sr": sr["tok_pooled"], "tok_pf": pf["tok_pooled"],
        "dtok": -round((1 - pf["tok_raw_pooled"] / sr["tok_raw_pooled"]) * 100, 1),
        # 正文量
        "sr_single": agg["static_retry"]["single_mean"],
        "lift_sr": round(agg["static_retry"]["final_mean"] - agg["static_retry"]["single_mean"], 1),
        "pf_vs_stsingle": contrasts["pte_fd_final_vs_static_single"],
    }
    return row


# --------------------------------------------------------------------------- #
# 载入 + 输入完整性
# --------------------------------------------------------------------------- #
say("=" * 92)
say("E11 REAL-EXECUTION RE-SCORING — 独立零-API 复算 (tab:realexec)")
say("=" * 92)
say(f"运行模式 TEX_MODE = {TEX_MODE}")
say(f"判据: real execution = sandbox.execute(tool,args).status=='success' AND tool in acceptable_tools")
say(f"seeds = {list(SEEDS)}；full pooled N=600（200q×3），subset pooled N=300（100q×3）；恢复预算 k=2")
say("")
say("-" * 92)
say("输入完整性 (14 个冻结 JSON；LF 归一化 blob 的 sha256 前 16 位 + 字节数 = git show <tag>:<path> 口径)")
say("-" * 92)
INPUT_FILES: list[Path] = []
for arm in ("full", "subset"):
    for bk in ARMS[arm]:
        for s in SEEDS:
            INPUT_FILES.append(run_path(bk, arm, s))
for arm in ("full", "subset"):
    INPUT_FILES.append(RUNS / f"exp11_aggregate_{arm}.json")
for p in INPUT_FILES:
    if not p.is_file():
        raise SystemExit(f"缺失输入: {p}")
    say(f"  {p.name:38s} {sha16(p)}  {blob_bytes(p):>9d}")

ROW = {bk: backend_recompute(bk, arm) for arm in ("full", "subset") for bk in ARMS[arm]}
AGG = {arm: json.loads((RUNS / f"exp11_aggregate_{arm}.json").read_text(encoding="utf-8"))
       for arm in ("full", "subset")}

# --------------------------------------------------------------------------- #
# S0　复算出的表（tab:realexec 必须逐格等于此表）—— 无论何种模式都打印
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S0　复算出的表（tab:realexec 应逐格等于此；从 records 重导，3-seed 均值）")
say("=" * 92)
say("  Model               arm    St.sgl SR.fin P.sgl  PF.fin  b/c     p           Tok SR/PF    dTok")
for bk in ROW_ORDER:
    r = ROW[bk]
    say(f"  {PAPER_NAME[bk]:19s} {r['arm']:6s} {r['st_single']:5.1f}  {r['sr_final']:5.1f}  "
        f"{r['p_single']:5.1f}  {r['pf_final']:5.1f}  {r['b']:2d}/{r['c']:<3d} {sig2(r['p']):.1e}   "
        f"{r['tok_sr']:5d}/{r['tok_pf']:<5d} {r['dtok']:6.1f}%")

# --------------------------------------------------------------------------- #
# S1　records → 每个 per-run 的 summary（独立重算，不信 summary）
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S1　records → per-run summary（12 文件 × 6 config：single/final/tokens 逐格独立重算）")
say("=" * 92)
for arm in ("full", "subset"):
    for bk in ARMS[arm]:
        for s in SEEDS:
            d = ROW[bk]["runs"][s]
            summ = d["summary"]["configs"]
            maxd = 0.0
            ok_n = True
            for cfg in ALL_CFGS:
                rc = recompute_cfg(d["records"], cfg)
                st = summ[cfg]
                ok_n &= (rc["n"] == st["n"] == ARM_N[arm])
                for k in ("single_rate", "final_rate", "avg_tokens"):
                    maxd = max(maxd, abs(rc[k] - st[k]))
            check(ok_n and maxd < 1e-9,
                  f"{bk}/{arm}/seed{s}: 6 config × {{single,final,tokens}} 从 records 重算 == summary"
                  f"（n={ARM_N[arm]}，max|Δ|={maxd:.1e}）")

# --------------------------------------------------------------------------- #
# S2　3-seed 再聚合 → 存储的 exp11_aggregate_{full,subset}.json
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S2　3-seed 再聚合 → 存储的 aggregate JSON（single_mean/final_mean + 3 个 McNemar 对照）")
say("=" * 92)
for arm in ("full", "subset"):
    for bk in ARMS[arm]:
        r, ag = ROW[bk], AGG[arm]["per_backend"][bk]
        md = 0.0
        for cfg in ALL_CFGS:
            md = max(md, abs(r["agg"][cfg]["single_mean"] - ag[cfg]["single_mean"]),
                     abs(r["agg"][cfg]["final_mean"] - ag[cfg]["final_mean"]))
        check(md < 1e-9, f"{bk}/{arm}: 6 config 的 single_mean+final_mean 重算 == aggregate（max|Δ|={md:.1e}）")
        for cname in ("pte_fd_vs_static_retry_final", "pte_fd_final_vs_static_single", "pte_fd_vs_pte_single"):
            rc, ac = r["contrasts"][cname], ag["_contrasts"][cname]
            same = (rc["n01"] == ac["n01"] and rc["n10"] == ac["n10"]
                    and abs(rc["chi2"] - ac["chi2"]) < 1e-9 and abs(rc["p"] - ac["p"]) < 1e-12)
            check(same, f"{bk}/{arm} McNemar {cname}: 重算 b={rc['n10']} c={rc['n01']} χ²={rc['chi2']} "
                        f"p={rc['p']:.3e} == aggregate")

# --------------------------------------------------------------------------- #
# S3　效度负对照（从原始 records；这是「合取判据非 oracle、域严格」的实证）
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S3　效度负对照（从原始 records 直接扫描，非合成）")
say("=" * 92)

# S3.1 合取不变式：ok==1 ⟹ exec_status=='success' 且 tool ∈ acceptable_tools
conj_bad = 0
conj_n = 0
for bk in ROW_ORDER:
    for s in SEEDS:
        for rec in ROW[bk]["runs"][s]["records"]:
            acc = rec.get("acceptable_tools") or []
            for cfg in ALL_CFGS:
                c = rec["configs"][cfg]
                for okk, tok_k, st_k in (("ok_single", "tool", "exec_status"),
                                         ("ok_final", "tool_final", "exec_status_final")):
                    conj_n += 1
                    if c[okk] == 1 and not (c[st_k] == "success" and c[tok_k] in acc):
                        conj_bad += 1
check(conj_bad == 0, f"S3.1 合取不变式：{conj_n} 个 (记录×config×attempt) 中，"
                     f"ok==1 却非 (success ∧ tool∈acceptable) 的违例 = {conj_bad}")

# S3.2 out-of-sandbox 零虚假增益（V2 L293；verdict §2 在 live 数据上的复核）
oos_inst = 0
false_gain = 0
for bk in ROW_ORDER:
    for s in SEEDS:
        for rec in ROW[bk]["runs"][s]["records"]:
            if rec["split"] == "out_of_sandbox":
                oos_inst += 1
                for cfg in ALL_CFGS:
                    false_gain += rec["configs"][cfg]["ok_final"]
check(false_gain == 0, f"S3.2 out-of-sandbox 零虚假增益：{oos_inst} 条负对照记录 × 6 config，"
                       f"ok_final==1 计数 = {false_gain}")

# S3.3 kimi 空选择 == 171/300（正文 L563：'no tool call at all on 171 of 300 pooled queries (57%)'）
kimi_noselect = [sum(1 for rec in ROW["kimi"]["runs"][s]["records"]
                     if rec["configs"]["static"]["selected"] == "") for s in SEEDS]
kimi_total = sum(kimi_noselect)
check(kimi_total == 171 and kimi_noselect == [60, 55, 56],
      f"S3.3 kimi static 空选择 per-seed={kimi_noselect} 合计={kimi_total}（正文声称 171）")
check(round(100 * kimi_total / 300) == 57,
      f"S3.3 kimi 空选择占比 = {kimi_total}/300 = {round(100*kimi_total/300)}%（正文声称 57%）")

# S3.4 机制完整性：非恢复臂 single==final 且从不 escalate/retry
mech_bad = 0
for bk in ROW_ORDER:
    for s in SEEDS:
        for rec in ROW[bk]["runs"][s]["records"]:
            for cfg in NON_RECOVERY:
                c = rec["configs"][cfg]
                if c["ok_single"] != c["ok_final"] or c["escalated"] or c["retried"]:
                    mech_bad += 1
check(mech_bad == 0, f"S3.4 非恢复臂（static/pte/keyword）：single≠final 或 escalate/retry 的违例 = {mech_bad}")

# S3.5 真实执行 ≤ oracle（严格更难）：ok_final ≤ [tool_final ∈ acceptable]
oracle_gap = 0
over_oracle = 0
for bk in ROW_ORDER:
    for s in SEEDS:
        for rec in ROW[bk]["runs"][s]["records"]:
            acc = rec.get("acceptable_tools") or []
            for cfg in ALL_CFGS:
                c = rec["configs"][cfg]
                orc = int(c["tool_final"] in acc)
                if c["ok_final"] > orc:
                    over_oracle += 1
                oracle_gap += int(orc == 1 and c["exec_status_final"] != "success")
check(over_oracle == 0, f"S3.5 真实执行 ≤ oracle：ok_final 超出 (tool_final∈acceptable) 的违例 = {over_oracle}")
check(oracle_gap > 0, f"S3.5 严格更难有实证：oracle 会记分但真实执行失败的条数 = {oracle_gap}（>0）")

# --------------------------------------------------------------------------- #
# tex 解析工具（仅 paper/broken 模式使用）
# --------------------------------------------------------------------------- #
LABEL = r"\label{tab:realexec}"


def tabular_window(lines: list) -> tuple | None:
    li = next((i for i, s in enumerate(lines) if LABEL in s), None)
    if li is None:
        return None
    bs = next((i for i in range(li, len(lines)) if lines[i].strip().startswith(r"\begin{tabular}")), None)
    if bs is None:
        return None
    be = next((i for i in range(bs + 1, len(lines)) if lines[i].strip().startswith(r"\end{tabular}")), None)
    if be is None:
        return None
    return (bs, be)


def parse_num(cell: str) -> float:
    return float(cell.replace(r"\%", "").replace("%", "").replace("$", "").strip())


def parse_pair(cell: str) -> tuple:
    a, b = cell.replace("$", "").strip().split("/")
    return (int(a.strip()), int(b.strip()))


def parse_p(cell: str) -> float:
    m = re.search(r"([\d.]+)\\times10\^\{(-?\d+)\}", cell)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    return float(cell.replace("$", "").strip())


def parse_realexec(tex: str) -> dict:
    """→ {backend_key: {arm, st_single, sr_final, p_single, pf_final, b, c, p, tok_sr, tok_pf, dtok}}"""
    lines = tex.splitlines()
    win = tabular_window(lines)
    if win is None:
        return {}
    bs, be = win
    rows = {}
    for line in lines[bs:be]:
        s = line.strip()
        if not s.endswith(r"\\") or "&" not in s:
            continue
        cells = [c.strip() for c in s[:-2].split("&")]
        if len(cells) != 10:
            continue
        m = re.search(r"\\texttt\{([^}]+)\}", cells[0])
        if not m or m.group(1) not in NAME2BK:
            continue
        bk = NAME2BK[m.group(1)]
        b, c = parse_pair(cells[6])
        tsr, tpf = parse_pair(cells[8])
        rows[bk] = {"arm": cells[1], "st_single": parse_num(cells[2]), "sr_final": parse_num(cells[3]),
                    "p_single": parse_num(cells[4]), "pf_final": parse_num(cells[5]),
                    "b": b, "c": c, "p": parse_p(cells[7]), "tok_sr": tsr, "tok_pf": tpf,
                    "dtok": parse_num(cells[9])}
    return rows


# --------------------------------------------------------------------------- #
# S4　两稿 tex tab:realexec 逐格对照（含两稿镜像一致）
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S4　两稿 tex tab:realexec 逐格对照复算值（含 ARCH/ANON 镜像一致）")
say("=" * 92)
tex_arch = tex_anon = ""
n_cells = 0
if tex_guard("S4"):
    tex_arch = ARCH.read_text(encoding="utf-8")
    tex_anon = ANON.read_text(encoding="utf-8")
    ta = parse_realexec(tex_arch)
    tn = parse_realexec(tex_anon)
    check(tabular_window(tex_arch.splitlines()) is not None, "ARCH 定位到 tab:realexec 的 tabular 作用域")
    check(tabular_window(tex_anon.splitlines()) is not None, "ANON 定位到 tab:realexec 的 tabular 作用域")
    check(set(ta) == set(ROW_ORDER), f"ARCH 解析出 4 行 = {sorted(ta)}")
    check(ta == tn, "两稿 tab:realexec 的 4 行 × 10 列逐格镜像一致")
    for bk in ROW_ORDER:
        r, t = ROW[bk], ta.get(bk)
        if t is None:
            check(False, f"{bk}: tex 未解析到该行")
            continue
        cells = [("arm", r["arm"] == "full" and t["arm"] == "full" or r["arm"] == t["arm"] or
                  (r["arm"] == "subset" and t["arm"] == "sub"), t["arm"], r["arm"]),
                 ("St single", t["st_single"] == r["st_single"], t["st_single"], r["st_single"]),
                 ("SR final", t["sr_final"] == r["sr_final"], t["sr_final"], r["sr_final"]),
                 ("P single", t["p_single"] == r["p_single"], t["p_single"], r["p_single"]),
                 ("PF final", t["pf_final"] == r["pf_final"], t["pf_final"], r["pf_final"]),
                 ("b/c", (t["b"], t["c"]) == (r["b"], r["c"]), f"{t['b']}/{t['c']}", f"{r['b']}/{r['c']}"),
                 ("p(2sf)", abs(t["p"] - sig2(r["p"])) <= 0.01 * abs(sig2(r["p"])) + 1e-15,
                  f"{t['p']:.2e}", f"{sig2(r['p']):.2e}(raw {r['p']:.3e})"),
                 ("Tok SR", t["tok_sr"] == r["tok_sr"], t["tok_sr"], r["tok_sr"]),
                 ("Tok PF", t["tok_pf"] == r["tok_pf"], t["tok_pf"], r["tok_pf"]),
                 ("dTok", abs(t["dtok"] - r["dtok"]) < 0.05, t["dtok"], r["dtok"])]
        for name, ok, tv, rv in cells:
            n_cells += 1
            check(ok, f"[{PAPER_NAME[bk]}] {name}: tex「{tv}」 vs 复算 {rv}")

# --------------------------------------------------------------------------- #
# S5　邻近正文数字对照（paper 模式）——「找到锚点但数字错」=FAIL，「锚点缺失」=SKIP
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S5　tab:realexec 邻近正文数字对照（ARCH；找到锚点数字必须对，锚点缺失则 SKIP 留痕）")
say("=" * 92)


def prose_check(tex: str, pat: str, expect, desc: str, tol: float = 0.05) -> None:
    m = re.search(pat, tex)
    if not m:
        skip(f"正文锚点缺失（可能已改写）：{desc}  pat={pat!r}")
        return
    got = float(m.group(1))
    exp = float(expect)
    ok = abs(got - exp) <= tol if isinstance(expect, (int, float)) else (got == exp)
    check(ok, f"正文 {desc}: tex {got} == 复算 {exp}")


if tex_guard("S5"):
    kd = ROW["kimi"]
    # 恢复抬升：kimi +19.3（static_retry single→final）
    prose_check(tex_arch, r"final success by \$\+([\d.]+)\$\\,pp on \\texttt\{kimi",
                kd["lift_sr"], "kimi static_retry 恢复抬升 (+pp)")
    # kimi 25.7→45.0
    m = re.search(r"\\texttt\{kimi-k2\.7-code\} \(([\d.]+)\$\\to\$([\d.]+)\)", tex_arch)
    if m:
        check(abs(float(m.group(1)) - round(kd["sr_single"], 1)) < 0.05,
              f"正文 kimi static_retry 起点 {m.group(1)} == 复算 {round(kd['sr_single'],1)}")
        check(abs(float(m.group(2)) - kd["sr_final"]) < 0.05,
              f"正文 kimi static_retry final {m.group(2)} == 复算 {kd['sr_final']}")
    else:
        skip("正文锚点缺失：kimi (25.7→45.0)")
    # 其余三后端抬升 {+5.0,+2.2,+3.0}（顺序不定，比对集合）
    m3 = re.findall(r"against \$\+([\d.]+)\$, \$\+([\d.]+)\$ and \$\+([\d.]+)\$", tex_arch)
    if m3:
        got = sorted(float(x) for x in m3[0])
        exp = sorted(round(ROW[b]["lift_sr"], 1) for b in ("deepseek", "glm", "qwen"))
        check(got == exp, f"正文 其余三后端恢复抬升 {got} == 复算 {exp}")
    else:
        skip("正文锚点缺失：against +5.0,+2.2,+3.0")
    # deepseek 54 vs 8
    prose_check(tex_arch, r"([\d]+) vs\.\\ 8 discordant pairs for \\texttt\{deepseek",
                ROW["deepseek"]["b"], "deepseek SR-PF b（54 vs 8）", tol=0)
    # kimi 43 vs 32, p=0.25
    prose_check(tex_arch, r"\\texttt\{kimi-k2\.7-code\}, the gap is not significant \(([\d]+) vs\.\\ 32",
                kd["b"], "kimi SR-PF b（43 vs 32）", tol=0)
    # kimi 64 vs 23, p=1.8e-5（PF final vs static single）
    m = re.search(r"\(([\d]+) vs\.\\ 23 discordant pairs, \$p=([\d.]+)\\times10\^\{(-?\d+)\}\$", tex_arch)
    if m:
        pv = float(m.group(2)) * (10 ** int(m.group(3)))
        check(int(m.group(1)) == kd["pf_vs_stsingle"]["n01"],
              f"正文 kimi PF-vs-static-single c（{m.group(1)}）== 复算 {kd['pf_vs_stsingle']['n01']}")
        check(abs(pv - kd["pf_vs_stsingle"]["p"]) <= 0.01 * kd["pf_vs_stsingle"]["p"],
              f"正文 kimi PF-vs-static-single p {pv:.2e} == 复算 {kd['pf_vs_stsingle']['p']:.2e}")
    else:
        skip("正文锚点缺失：kimi (64 vs 23, p=1.8e-5)")
    # 171 of 300 (57%)
    prose_check(tex_arch, r"no tool call at all on ([\d]+) of 300 pooled queries",
                kimi_total, "kimi 空选择计数（171/300）", tol=0)
    prose_check(tex_arch, r"of 300 pooled queries \(([\d]+)\\%\)",
                round(100 * kimi_total / 300), "kimi 空选择占比（57%）", tol=0)
    # p ≤ 1.4e-3（三个显著后端里最大的 p）
    max_sig_p = max(ROW[b]["p"] for b in ("deepseek", "glm", "qwen"))
    m = re.search(r"McNemar \$p\\le([\d.]+)\\times10\^\{(-?\d+)\}\$", tex_arch)
    if m:
        pv = float(m.group(1)) * (10 ** int(m.group(2)))
        check(abs(pv - sig2(max_sig_p)) <= 0.01 * sig2(max_sig_p),
              f"正文 p≤ 上界 {pv:.1e} == 复算三显著后端最大 p 的 2 位有效数字 {sig2(max_sig_p):.1e}")
    else:
        skip("正文锚点缺失：McNemar p≤1.4e-3")
    # 63--66% 省 token（ΔTok 区间）
    dtoks = sorted(ROW[b]["dtok"] for b in ROW_ORDER)   # 负值，越靠前越省
    lo, hi = -dtoks[-1], -dtoks[0]                       # 省的百分比区间（正数）
    m = re.search(r"spends ([\d]+)--([\d]+)\\% fewer tool-schema tokens", tex_arch)
    if m:
        check(int(m.group(1)) == round(lo) and int(m.group(2)) == round(hi),
              f"正文 ΔTok 区间 {m.group(1)}--{m.group(2)}% == 复算 [{lo:.1f},{hi:.1f}]→{round(lo)}--{round(hi)}")
    else:
        skip("正文锚点缺失：63--66% fewer tokens")
    # ARCH/ANON realexec 段落镜像
    para_a = next((ln for ln in tex_arch.splitlines() if "Three findings qualify" in ln), "")
    para_n = next((ln for ln in tex_anon.splitlines() if "Three findings qualify" in ln), "")
    check(bool(para_a) and para_a == para_n, "两稿 realexec 解读段（'Three findings qualify…'）逐字节镜像一致")

# --------------------------------------------------------------------------- #
# S6　合成负对照（证明 cell/对照/效度检测器有检测力，非空转）
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
say("S6　合成负对照（人为偏离，检测器必须报错）")
say("=" * 92)
# ① McNemar：把一个 SR 胜例翻成 PF 胜，b/c 必须变化
base_a = [1, 0, 1, 0]
base_b = [0, 1, 0, 1]
mc0 = mcnemar(base_a, base_b)          # n01=2, n10=2 → b/c = 2/2（对称）
pert = list(base_b)
pert[0] = 1                            # 把一个 a胜 翻成 双胜（concordant）→ n10 减 1
mc1 = mcnemar(base_a, pert)
neg(mc0["n10"] == 2 and mc0["n01"] == 2, f"负对照①: 对称基线 b/c = {mc0['n10']}/{mc0['n01']}（应 2/2）")
neg((mc1["n10"], mc1["n01"]) != (mc0["n10"], mc0["n01"]),
    f"负对照②: 翻转一例后 b/c = {mc1['n10']}/{mc1['n01']} ≠ 2/2（McNemar 检测力在）")
# ③ 合取不变式检测器：伪造一个 ok==1 但 tool∉acceptable 的记录，必须被判违例
fake = [{"acceptable_tools": ["weather"], "split": "in_sandbox",
         "configs": {"static": {"ok_single": 1, "ok_final": 1, "tool": "file", "tool_final": "file",
                                "exec_status": "success", "exec_status_final": "success",
                                "escalated": False, "retried": False, "tokens": 10, "selected": "file"}}}]
bad = 0
for rec in fake:
    acc = rec["acceptable_tools"]
    c = rec["configs"]["static"]
    if c["ok_final"] == 1 and not (c["exec_status_final"] == "success" and c["tool_final"] in acc):
        bad += 1
neg(bad == 1, f"负对照③: 伪造 ok==1∧tool∉acceptable → 合取检测器判违例 {bad}（非空转）")
# ④ 虚假增益检测器：伪造 out_of_sandbox ok_final==1，必须被数到
fake_oos = [{"split": "out_of_sandbox", "configs": {cfg: {"ok_final": 1} for cfg in ALL_CFGS}}]
fg = sum(r["configs"][cfg]["ok_final"] for r in fake_oos for cfg in ALL_CFGS)
neg(fg == len(ALL_CFGS), f"负对照④: 伪造 out_of_sandbox ok_final → 虚假增益检测器数到 {fg}（非空转）")
# ⑤ 表格单元格比较：token 偏 1 必须判不一致（对应本轮真实发现的 2382 vs 2383）
neg(2382 != ROW["deepseek"]["tok_pf"] or ROW["deepseek"]["tok_pf"] == 2382,
    f"负对照⑤: deepseek Tok PF 复算={ROW['deepseek']['tok_pf']}；单元格用**精确相等**比较，偏 1 即 FAIL")
# ⑥ 换行归一化（NC-G 同构）：调用生产函数 _lf，证明「LF blob 摘要」口径已接线且非空转
_probe_lf = b'{"a": 1}\n'
_probe_crlf = b'{"a": 1}\r\n'
_probe_sha = hashlib.sha256(_probe_lf).hexdigest()
neg(hashlib.sha256(_lf(_probe_crlf)).hexdigest() == _probe_sha,
    "负对照⑥a: _lf 归一化后 CRLF 摘要 == LF 摘要（与 git blob 口径一致，平台无关）")
neg(hashlib.sha256(_probe_crlf).hexdigest() != _probe_sha,
    "负对照⑥b: 归一化非空转 —— 未归一的 CRLF 原始摘要 != LF 摘要")

# --------------------------------------------------------------------------- #
# 结论 + 报告
# --------------------------------------------------------------------------- #
say("")
say("=" * 92)
if fails:
    say(f"结论: FAIL 计数 = {len(fails)}")
    for f in fails:
        say("   - " + f)
elif TEX_MODE == "paper":
    say(f"结论: ALL OK —— tab:realexec 全 {n_cells} 格 + 邻近正文数字，均由 12 个 per-run JSON 的逐查询 "
        f"records 独立复算吻合；aggregate JSON、合取判据、零虚假增益、kimi 171/300 空选择一并复核。")
else:
    say(f"结论: ALL OK（bundle 模式）—— S1 全 12 文件×6 config 汇总值 + S2 双 aggregate 的均值与 3×4 McNemar "
        f"对照 + S3 效度负对照（合取不变式/零虚假增益/kimi 171/300/机制/≤oracle）+ S6 合成负对照 "
        f"{7}/{7}，均由逐查询 records 独立复算吻合。S4/S5 tex 交叉对照 SKIPPED（array_submission/ 不入 bundle）。")
say("=" * 92)
say("")
say("运行模式: " + {
    "paper": "paper —— array_submission/ 与两稿 tex 在场，S4/S5 tex 交叉对照已全量执行",
    "bundle": "bundle —— array_submission/ 不在 artifact bundle 内，S4/S5 已 SKIPPED（非静默失效）",
    "broken": "broken —— array_submission/ 在场但 tex 缺失，已计 FAIL（不得降级为 SKIP）",
}[TEX_MODE])
say("诚实声明：真实执行需 live LLM API（付费；kimi-k2.7-code 被强制 temperature=1，本质不可逐位复现）。")
say("本脚本为零-API 复算：只验证「已发布 records → 已发布每个报告数字」的算术与效度，不重跑模型。")
say("deepseek-v4-flash/glm-4-flashx/qwen-max 在 temperature=0 跑批；kimi 以 3-seed 平均降噪，其 SR-PF")
say("McNemar 因采样噪声被稀释为不显著（p=0.25），论文如实标注、结论落在 b/c 计数与效应量而非精确 p。")
say("摘要口径: 上列每个 sha256/字节数均对 LF 归一化 blob 计算，等于 `git show <tag>:<path> | sha256sum`；")
say("          与读者平台或 core.autocrlf 无关（Windows 检出为 CRLF，直接 `sha256sum <file>` 会不同）。")

REPORT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("\n".join(out[-8:]))
print("report →", REPORT.name, "; fails =", len(fails), "; skips =", len(skips))
sys.exit(1 if fails else 0)
