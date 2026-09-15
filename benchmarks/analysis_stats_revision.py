#!/usr/bin/env python3
"""Statistical re-analysis for the ARRAY revision of the WeClaw paper (reviewer R1, comment 4).

WHAT THIS SCRIPT DOES
---------------------
Recomputes, from the committed per-run raw data alone, every paired accuracy
comparison, p-value and confidence interval quoted in the paper, and then audits
those quoted numbers against the recomputation (see ``--claims``).

WHY THE ORIGINAL ANALYSIS WAS REPLACED
--------------------------------------
The submitted manuscript argued significance from the non-overlap of two
marginal Wilson intervals and quoted uncorrected McNemar p-values computed on
the pooled three-seed data. Both steps are unsound for this design.

1. Marginal intervals. In a paired design two marginal 95% intervals can
   overlap while the paired difference is significant, and can fail to overlap
   while it is not. Non-overlap is neither necessary nor sufficient evidence.
   Every inferential claim must be made on the paired difference.

2. Independence. ``exp1_pte_ablation.generate_test_queries(n=100, seed=s)``
   allocates ``100 // 12 = 8`` queries per intent across 12 intents -- hence 96
   queries, not 100 -- by sampling from a fixed pool of 12 x 10 = 120 query
   templates. The three seeds therefore do NOT produce disjoint query sets: the
   pooled N = 288 observations per back-end contain only 118 distinct query
   texts (10 occurring in one seed, 46 in two, 62 in all three). Observations
   sharing a query text are positively correlated, so McNemar's test, the paired
   Wald interval and Newcombe's (1998) paired-difference interval alike -- all of
   which assume N mutually independent paired observations -- understate the
   standard error of the paired difference. On this dataset the understatement
   is 18-58% of the standard error, which inflates significance by two to three
   orders of magnitude in the extreme cells.

WHAT IS REPORTED INSTEAD
------------------------
For every paired comparison this script emits:
  * the continuity-corrected McNemar chi-square and its p-value, i.e. exactly
    the quantity the original submission quoted, reproduced for traceability;
  * the exact binomial McNemar p-value (two-sided, integer arithmetic);
  * a cluster-robust Wald statistic taking the query text as the cluster, with
    the G/(G-1) small-sample correction of Cameron, Gelbach & Miller (2008),
    "Bootstrap-based improvements for inference with clustered errors",
    The Review of Economics and Statistics 90(3), 414-427;
  * a cluster bootstrap percentile interval resampling whole queries, which
    assumes no sampling distribution and remains well defined when a discordant
    cell is empty (c = 0), as happens for glm-4-flash and for the n = 500
    PTE-FD-vs-PTE comparison.
The paper quotes cluster-robust Wald p-values and cluster bootstrap intervals.

For the n = 500 evaluation (exp10b) every query is distinct, so G = N = 500, no
within-cluster correlation exists, and the cluster-robust estimator reduces to
the ordinary paired Wald estimator; self-check S2 asserts that identity.

SELF-CHECKS (every line must read PASS)
---------------------------------------
S1  with exactly one observation per cluster, the cluster-robust variance equals
    the closed-form paired Wald variance [(b+c) - (b-c)^2 / N] / N^2 scaled by
    G/(G-1);
S2  the same identity on the real n = 500 data, where G = N = 500;
S3  the uncorrected chi-square and p reproduce the eight ``_mcnemar_*`` entries
    already committed in ``runs/recanon/recanon_aggregate.json``;
S4  the cluster structure is internally consistent:
    sum(cluster size x number of clusters of that size) = pooled N = 288;
S5  every statistical number printed in the manuscript is reproduced.

USAGE
-----
    python analysis_stats_revision.py                  # full report + audit
    python analysis_stats_revision.py --claims-only    # audit only
    python analysis_stats_revision.py --boot 2000      # faster, coarser bootstrap
    python analysis_stats_revision.py --json out.json  # machine-readable dump

DEPENDENCIES: Python >= 3.8 standard library only. No third-party packages, so
the numbers a reviewer obtains do not depend on a BLAS or numpy version.
Bootstrap draws are seeded per comparison from a CRC32 of the comparison label,
which makes every interval reproducible and independent of iteration order.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RAW = BENCH / "raw_data"
RECANON = RAW / "runs" / "recanon"

BACKENDS = ("deepseek", "qwen", "kimi", "glm")
PAPER_NAMES = {
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen-max",
    "kimi": "moonshot-v1-8k",
    "glm": "glm-4-flash",
}
SEEDS = (42, 43, 44)
CONFIGS = ("static", "pte", "pte_fd", "keyword")
PAIRS = (("pte_fd", "pte"), ("pte_fd", "static"), ("pte", "static"), ("pte_fd", "keyword"))
N500_PAIRS = (("pte_fd", "pte"), ("pte_fd", "static"), ("pte", "static"))
N500_RUNS = (
    ("static", "20260608_n500_static", "exp10b_static_results.json", "static"),
    ("pte", "20260608_n500_pte", "exp10b_pte_results.json", "pte"),
    ("pte_fd", "20260608_n500_ptefd", "exp10b_pte-fd_results.json", "pte-fd"),
)

Z95 = 1.959963984540054
BOOT_DEFAULT = 10000
RNG_SEED = 20260914
LINE = "=" * 100


# --------------------------------------------------------------------------- #
# elementary statistics
# --------------------------------------------------------------------------- #
def two_sided_p_from_z(z: float) -> float:
    """Two-sided normal p-value, via erfc so that tiny tails stay accurate."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a single proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return ((centre - half) * 100.0, (centre + half) * 100.0)


def mcnemar_chi2_cc(b: int, c: int) -> float:
    """Continuity-corrected McNemar chi-square on the discordant cells."""
    nd = b + c
    return ((abs(b - c) - 1) ** 2 / nd) if nd else 0.0


def mcnemar_p_chi2_cc(b: int, c: int) -> float:
    nd = b + c
    if not nd:
        return 1.0
    return math.erfc(math.sqrt(mcnemar_chi2_cc(b, c) / 2.0))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value in exact integer arithmetic.

    Under H0 the smaller discordant cell is Binomial(b+c, 1/2), so the two-sided
    p is 2 * sum_{i<=min(b,c)} C(b+c, i) / 2^(b+c), capped at 1. Done with
    integers this is exact for b+c of any realistic size.
    """
    nd = b + c
    if nd == 0:
        return 1.0
    tail = sum(math.comb(nd, i) for i in range(min(b, c) + 1))
    return min(1.0, tail / 2 ** (nd - 1))


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile, identical to numpy's default method."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def sig_fig(x: float, n: int = 2) -> float:
    """Round to n significant figures, for comparing against printed values."""
    if x == 0:
        return 0.0
    d = n - 1 - math.floor(math.log10(abs(x)))
    return round(x, d)


def holm(items: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Holm-Bonferroni step-down. Returns [(label, raw p, adjusted p)] sorted."""
    ordered = sorted(items, key=lambda kv: kv[1])
    m = len(ordered)
    out: list[tuple[str, float, float]] = []
    prev = 0.0
    for j, (lab, p) in enumerate(ordered):
        adj = min(1.0, max(prev, (m - j) * p))
        prev = adj
        out.append((lab, p, adj))
    return out


# --------------------------------------------------------------------------- #
# the paired-difference estimator
# --------------------------------------------------------------------------- #
def cluster_infer(u: list[float], cid: list[int], n_boot: int, rng_seed: int) -> dict:
    """Inference on theta = mean(u), where u_j = correct_A(j) - correct_B(j).

    u_j lies in {-1, 0, +1}; cid_j identifies the cluster (distinct query text)
    of observation j. theta-hat is the paired difference in proportions, and
    (b, c) are the two discordant cells of the McNemar table.

    Variances:
        naive   Var = sum_j (u_j - theta)^2 / N^2
        cluster Var = G/(G-1) * sum_q (U_q - n_q * theta)^2 / N^2
    where U_q = sum_{j in q} u_j and n_q = |cluster q|. The cluster form is the
    sandwich estimator for a mean with cluster-level score contributions; when
    every cluster holds one observation the two differ only by G/(G-1).

    The bootstrap resamples clusters with replacement. Because theta* for a
    resample depends only on the per-cluster aggregates, each draw costs O(G):
        theta* = sum_{q picked} U_q / sum_{q picked} n_q
    """
    N = len(u)
    if N == 0:
        raise ValueError("empty comparison")
    theta = math.fsum(u) / N
    G = max(cid) + 1

    Uq = [0.0] * G
    nq = [0] * G
    for uj, q in zip(u, cid):
        Uq[q] += uj
        nq[q] += 1

    b = sum(1 for x in u if x > 0)
    c = sum(1 for x in u if x < 0)

    var_naive = math.fsum((x - theta) ** 2 for x in u) / N ** 2
    if G > 1:
        var_clus = (math.fsum((Uq[q] - nq[q] * theta) ** 2 for q in range(G))
                    / N ** 2 * (G / (G - 1)))
    else:
        var_clus = var_naive
    se_naive = math.sqrt(var_naive)
    se_clus = math.sqrt(var_clus)

    res: dict = {
        "N": N, "G": G, "theta_pp": theta * 100.0,
        "b": b, "c": c, "n_discordant": b + c,
        "var_naive": var_naive, "var_clus": var_clus,
        "se_naive_pp": se_naive * 100.0, "se_clus_pp": se_clus * 100.0,
        "z_naive": theta / se_naive if se_naive > 0 else float("inf"),
        "z_clus": theta / se_clus if se_clus > 0 else float("inf"),
        "chi2_cc": mcnemar_chi2_cc(b, c),
        "p_chi2_cc": mcnemar_p_chi2_cc(b, c),
        "p_exact": mcnemar_exact_p(b, c),
        "se_inflation": (se_clus / se_naive) if se_naive > 0 else float("nan"),
    }
    res["p_naive"] = two_sided_p_from_z(res["z_naive"]) if se_naive > 0 else 0.0
    res["p_clus"] = two_sided_p_from_z(res["z_clus"]) if se_clus > 0 else 0.0
    res["ci_naive_pp"] = ((theta - Z95 * se_naive) * 100.0, (theta + Z95 * se_naive) * 100.0)
    res["ci_clus_pp"] = ((theta - Z95 * se_clus) * 100.0, (theta + Z95 * se_clus) * 100.0)

    if n_boot > 0 and G > 1:
        rng = random.Random(rng_seed)
        pool = list(range(G))
        picks = [rng.choices(pool, k=G) for _ in range(n_boot)]
        boots = sorted(
            math.fsum(Uq[q] for q in pick) / math.fsum(nq[q] for q in pick)
            for pick in picks
        )
        res["ci_boot_pp"] = (percentile(boots, 2.5) * 100.0, percentile(boots, 97.5) * 100.0)
        bmean = math.fsum(boots) / len(boots)
        res["boot_mean_pp"] = bmean * 100.0
        # ddof=1: the bootstrap distribution is itself a sample.
        res["boot_se_pp"] = math.sqrt(
            math.fsum((x - bmean) ** 2 for x in boots) / (len(boots) - 1)
        ) * 100.0
        n_le = sum(1 for x in boots if x <= 0.0)
        n_ge = sum(1 for x in boots if x >= 0.0)
        # +1 in numerator and denominator: a bootstrap p can never be exactly 0,
        # so the resolution floor is 2/(B+1) and must be quoted as "p < 2/(B+1)".
        res["p_boot"] = min(1.0, 2.0 * min((n_le + 1) / (n_boot + 1), (n_ge + 1) / (n_boot + 1)))
        res["boot_resolution"] = 2.0 / (n_boot + 1)
    return res


# --------------------------------------------------------------------------- #
# rebuilding the cluster structure
# --------------------------------------------------------------------------- #
_NEEDED_CONSTS = {"INTENT_GROUND_TRUTH", "QUERY_TEMPLATES", "TOTAL_QUERIES", "RANDOM_SEED"}
_SAFE_IMPORTS = {"random", "json", "math", "re", "collections", "itertools"}


def load_query_texts(n: int, seeds) -> dict[int, list[str]]:
    """Rebuild the per-seed query texts without importing exp1_pte_ablation.

    That module pulls in ``openai``, ``dotenv`` and the whole ``src.core``
    package at import time, none of which is needed to reproduce statistics, and
    requiring them would make this script unrunnable for a reviewer who only
    wants to check the numbers. We therefore parse the module and execute just
    the query generator plus the constant tables it reads.

    Note: those constants are *annotated* assignments (``X: dict = {...}``), i.e.
    ``ast.AnnAssign``. Collecting only ``ast.Assign`` yields an empty namespace
    and a confusing NameError, so both node types must be handled.
    """
    src = (BENCH / "exp1_pte_ablation.py").read_text(encoding="utf-8")
    keep: list[ast.stmt] = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import) and all(a.name in _SAFE_IMPORTS for a in node.names):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            if any(getattr(t, "id", "") in _NEEDED_CONSTS for t in node.targets):
                keep.append(node)
        elif isinstance(node, ast.AnnAssign):
            if getattr(node.target, "id", "") in _NEEDED_CONSTS:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "generate_test_queries":
            keep.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<extracted>", "exec"), ns)
    missing = (_NEEDED_CONSTS | {"generate_test_queries"}) - set(ns)
    if missing:
        raise SystemExit(
            "could not extract the query generator from exp1_pte_ablation.py; "
            f"missing {sorted(missing)}"
        )
    return {s: [q["query"] for q in ns["generate_test_queries"](n=n, seed=s)] for s in seeds}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse(n_boot: int) -> tuple[dict, dict, dict, dict]:
    qtext = load_query_texts(n=100, seeds=SEEDS)
    uniq = sorted({t for v in qtext.values() for t in v})
    qidx = {t: i for i, t in enumerate(uniq)}
    mult = Counter(t for v in qtext.values() for t in v)
    clusters = {
        "G": len(uniq),
        "N": sum(len(v) for v in qtext.values()),
        "per_seed": {s: len(v) for s, v in qtext.items()},
        "size_distribution": dict(sorted(Counter(mult.values()).items())),
        "pairwise_overlap": {
            f"{a}-{b}": len(set(qtext[a]) & set(qtext[b]))
            for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]
        },
    }

    # pooled observation order per back-end: seed 42 block, then 43, then 44
    cid = [qidx[t] for s in SEEDS for t in qtext[s]]

    results: dict[tuple[str, str, str], dict] = {}
    per_seed_docs = {
        (be, s): load_json(RECANON / f"{be}_seed{s}_n100.json")
        for be in BACKENDS for s in SEEDS
    }
    agg = load_json(RECANON / "recanon_aggregate.json")["per_model"]

    for be in BACKENDS:
        vec = {cfg: [] for cfg in CONFIGS}
        for s in SEEDS:
            pq = per_seed_docs[(be, s)]["per_query"]
            for cfg in CONFIGS:
                vec[cfg].extend(float(x) for x in pq[cfg])
        if any(len(v) != len(cid) for v in vec.values()):
            raise SystemExit(f"{be}: per_query length != pooled cluster id length")
        for A, B in PAIRS:
            u = [x - y for x, y in zip(vec[A], vec[B])]
            label = f"n96x3|{be}|{A}_vs_{B}"
            results[(be, A, B)] = cluster_infer(
                u, cid, n_boot, RNG_SEED ^ zlib.crc32(label.encode())
            )

    # ---- n = 500 (exp10b): each query distinct, so cluster == observation
    vec500: dict[str, list[float]] = {}
    ids500: list[str] = []
    for cfg, run_dir, fname, key in N500_RUNS:
        doc = load_json(RAW / "runs" / run_dir / fname)
        details = doc["details"]
        vec500[cfg] = [float(bool(x["results"][key]["correct"])) for x in details]
        cur = [x["query_id"] for x in details]
        if ids500 and cur != ids500:
            raise SystemExit("exp10b: query_id order differs between runs; cannot pair")
        ids500 = cur
    if len(set(ids500)) != len(ids500):
        raise SystemExit("exp10b: query_id values are not unique; clustering assumption broken")
    cid500 = list(range(len(ids500)))
    n500_meta = {"n_items": len(ids500), "unique_ids": len(set(ids500))}
    for A, B in N500_PAIRS:
        u = [x - y for x, y in zip(vec500[A], vec500[B])]
        label = f"n500|{A}_vs_{B}"
        results[("n500", A, B)] = cluster_infer(
            u, cid500, n_boot, RNG_SEED ^ zlib.crc32(label.encode())
        )

    # ---- marginal accuracies, Wilson intervals and token reduction
    marginals: dict[str, dict] = {}
    for be in BACKENDS:
        marginals[be] = {}
        for cfg in CONFIGS:
            pooled = sum(float(x) for s in SEEDS for x in per_seed_docs[(be, s)]["per_query"][cfg])
            N = len(cid)
            accs = [
                sum(float(x) for x in per_seed_docs[(be, s)]["per_query"][cfg])
                / len(per_seed_docs[(be, s)]["per_query"][cfg]) * 100.0
                for s in SEEDS
            ]
            mean = sum(accs) / len(accs)
            sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
            lo, hi = wilson_ci(int(pooled), N)
            marginals[be][cfg] = {
                "pooled_correct": int(pooled), "pooled_N": N,
                "acc_mean": mean, "acc_sd": sd,
                "wilson_lo": lo, "wilson_hi": hi,
                "avg_tokens_mean": agg[be][cfg]["avg_tokens_mean"],
            }
    for cfg, run_dir, fname, key in N500_RUNS:
        doc = load_json(RAW / "runs" / run_dir / fname)
        corr = [bool(x["results"][key]["correct"]) for x in doc["details"]]
        toks = [x["results"][key]["tokens"] for x in doc["details"]]
        lo, hi = wilson_ci(sum(corr), len(corr))
        marginals.setdefault("n500", {})[cfg] = {
            "pooled_correct": sum(corr), "pooled_N": len(corr),
            "acc_mean": sum(corr) / len(corr) * 100.0,
            "wilson_lo": lo, "wilson_hi": hi,
            "avg_tokens_mean": math.fsum(toks) / len(toks),
            "summary_accuracy": doc["summary"][key]["accuracy"],
            "summary_wilson": doc["summary"][key]["wilson_ci_95"],
            "summary_tokens": doc["summary"][key]["avg_tokens_per_query"],
        }

    tokens: dict[str, dict] = {}
    for be in BACKENDS:
        st = marginals[be]["static"]["avg_tokens_mean"]
        tokens[be] = {
            cfg: (1.0 - marginals[be][cfg]["avg_tokens_mean"] / st) * 100.0
            for cfg in CONFIGS if cfg != "static"
        }
    st5 = marginals["n500"]["static"]["avg_tokens_mean"]
    tokens["n500"] = {
        cfg: (1.0 - marginals["n500"][cfg]["avg_tokens_mean"] / st5) * 100.0
        for cfg in ("pte", "pte_fd")
    }

    return results, clusters, marginals, tokens


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #
def self_checks(results: dict, clusters: dict, n_boot: int) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    # S1: synthetic data, one observation per cluster
    rng = random.Random(1)
    u = [float(rng.randint(-1, 1)) for _ in range(300)]
    cid_single = list(range(300))
    r = cluster_infer(u, cid_single, 0, 0)
    b, c, N = r["b"], r["c"], 300
    closed = ((b + c) - (b - c) ** 2 / N) / N ** 2
    checks.append(("S1a cluster Var -> closed-form paired Wald Var (G = N)",
                   abs(closed - r["var_naive"]) < 1e-18,
                   f"closed={closed:.12e} computed={r['var_naive']:.12e}"))
    ratio = r["var_clus"] / r["var_naive"]
    checks.append(("S1b G/(G-1) small-sample factor applied to the VARIANCE",
                   abs(ratio - 300 / 299) < 1e-12,
                   f"ratio={ratio:.9f} expected={300 / 299:.9f}"))

    # S2: the real n=500 data has G = N = 500
    for key in (("n500", "pte_fd", "pte"), ("n500", "pte_fd", "static"), ("n500", "pte", "static")):
        rr = results[key]
        ratio = rr["var_clus"] / rr["var_naive"]
        checks.append((f"S2 n=500 {key[1]} vs {key[2]}: G = N, Var ratio = G/(G-1)",
                       rr["G"] == rr["N"] == 500 and abs(ratio - 500 / 499) < 1e-12,
                       f"G={rr['G']} N={rr['N']} ratio={ratio:.9f} expected={500 / 499:.9f}"))

    # S3: reproduce the eight committed McNemar entries
    agg = load_json(RECANON / "recanon_aggregate.json")["per_model"]
    for be in BACKENDS:
        for other in ("pte", "static"):
            committed = agg[be].get(f"_mcnemar_fd_vs_{other}")
            if not committed:
                continue
            rr = results[(be, "pte_fd", other)]
            ok = (committed["n01"] == rr["b"] and committed["n10"] == rr["c"]
                  and abs(committed["chi2"] - rr["chi2_cc"]) < 2e-3
                  and abs(committed["p"] - rr["p_chi2_cc"]) <= 1e-2 * max(committed["p"], 1e-300))
            checks.append((f"S3 {be}: FD vs {other} reproduces committed McNemar",
                           ok,
                           f"committed n01={committed['n01']} n10={committed['n10']} "
                           f"chi2={committed['chi2']:.3f} p={committed['p']:.6g} | "
                           f"computed b={rr['b']} c={rr['c']} chi2={rr['chi2_cc']:.3f} "
                           f"p={rr['p_chi2_cc']:.6g}"))

    # S4: cluster structure internal consistency
    recon = sum(k * v for k, v in clusters["size_distribution"].items())
    checks.append(("S4 sum(cluster size x count) = pooled N",
                   recon == clusters["N"] == 288,
                   f"reconstructed={recon} pooled N={clusters['N']}"))
    checks.append(("S4b every seed yields 96 queries (not the requested 100)",
                   all(v == 96 for v in clusters["per_seed"].values()),
                   str(clusters["per_seed"])))

    # exact vs chi-square sanity on an empty discordant cell
    checks.append(("S4c exact McNemar p at c = 0 equals 2 * 0.5^(b+c)",
                   abs(mcnemar_exact_p(29, 0) - 2 * 0.5 ** 29) < 1e-18,
                   f"exact={mcnemar_exact_p(29, 0):.6e} expected={2 * 0.5 ** 29:.6e}"))
    if n_boot:
        checks.append((f"S4d bootstrap p resolution floor is 2/(B+1) = {2 / (n_boot + 1):.6g}",
                       all(abs(results[k].get("boot_resolution", 2 / (n_boot + 1))
                               - 2 / (n_boot + 1)) < 1e-15
                           for k in results if "boot_resolution" in results[k]),
                       f"B={n_boot}"))
    return checks


# --------------------------------------------------------------------------- #
# S5: audit every statistical number printed in the manuscript
# --------------------------------------------------------------------------- #
def audit_paper_claims(results: dict, clusters: dict, marginals: dict,
                       tokens: dict) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def num(what: str, got: float, want: float, dp: int) -> None:
        checks.append((what, round(got, dp) == want, f"computed={round(got, dp)} paper={want}"))

    def sci(what: str, got: float, want: float) -> None:
        checks.append((what, sig_fig(got, 2) == want,
                       f"computed={sig_fig(got, 2):.6g} paper={want:.6g}"))

    def bound(what: str, got: float, ceiling: float) -> None:
        checks.append((what, got < ceiling, f"computed={got:.6g} paper claims < {ceiling:g}"))

    def ci(what: str, got: tuple[float, float], want: tuple[float, float]) -> None:
        ok = round(got[0], 1) == want[0] and round(got[1], 1) == want[1]
        checks.append((what, ok,
                       f"computed=[{got[0]:+.2f},{got[1]:+.2f}] paper=[{want[0]:+.1f},{want[1]:+.1f}]"))

    def cell(what: str, got: tuple[int, int], want: tuple[int, int]) -> None:
        checks.append((what, got == want, f"computed b,c={got} paper={want}"))

    ds = results[("deepseek", "pte_fd", "pte")]
    dss = results[("deepseek", "pte_fd", "static")]
    dsp = results[("deepseek", "pte", "static")]

    # --- Table 2 (tab:ablation_pte) caption footnotes
    bound("Table 2 fn *: cluster-robust McNemar FD vs PTE p < 1e-4", ds["p_clus"], 1e-4)
    bound("Table 2 fn dag: cluster-robust McNemar FD vs Static p < 1e-3", dss["p_clus"], 1e-3)

    # --- Table 2 body: marginal accuracy, SD and pooled Wilson CI
    for cfg, want_acc, want_sd, want_ci in (
        ("static", 74.0, 1.0, (68.6, 78.7)),
        ("pte", 74.7, 1.6, (69.3, 79.3)),
        ("pte_fd", 86.8, 2.4, (82.4, 90.2)),
        ("keyword", 59.7, 1.6, (54.0, 65.2)),
    ):
        m = marginals["deepseek"][cfg]
        num(f"Table 2 row {cfg}: accuracy mean", m["acc_mean"], want_acc, 1)
        num(f"Table 2 row {cfg}: SD across seeds", m["acc_sd"], want_sd, 1)
        ci(f"Table 2 row {cfg}: pooled Wilson 95% CI",
           (m["wilson_lo"], m["wilson_hi"]), want_ci)

    # --- Section "Accuracy and token efficiency"
    sci("Ablation text: FD vs PTE cluster-robust p = 2.0e-5", ds["p_clus"], 2.0e-5)
    sci("Ablation text: FD vs Static cluster-robust p = 9.0e-4", dss["p_clus"], 9.0e-4)
    num("Ablation text: FD vs PTE paired difference (pp)", ds["theta_pp"], 12.2, 1)
    num("Ablation text: FD vs Static paired difference (pp)", dss["theta_pp"], 12.8, 1)
    ci("Ablation text: FD vs PTE 95% cluster bootstrap CI", ds["ci_boot_pp"], (6.8, 17.8))
    ci("Ablation text: FD vs Static 95% cluster bootstrap CI", dss["ci_boot_pp"], (5.5, 20.6))
    num("Ablation text: PTE vs Static paired difference (pp)", dsp["theta_pp"], 0.7, 1)
    ci("Ablation text: PTE vs Static 95% CI", dsp["ci_boot_pp"], (-7.9, 9.1))
    num("Ablation text: PTE vs Static cluster-robust p = 0.87", dsp["p_clus"], 0.87, 2)
    sci("Ablation text: uncorrected McNemar FD vs PTE = 5.2e-8", ds["p_chi2_cc"], 5.2e-8)
    sci("Ablation text: uncorrected McNemar FD vs Static = 5.7e-6", dss["p_chi2_cc"], 5.7e-6)

    # --- Table 9 (tab:crossmodel) FD-P / FD-St columns: exact paired differences
    want_gain = {
        "deepseek": (12.2, 12.8), "qwen": (2.8, 10.8),
        "kimi": (5.2, 14.2), "glm": (2.8, 8.7),
    }
    for be, (wp, wst) in want_gain.items():
        num(f"Table 9 {PAPER_NAMES[be]}: FD-P gain (pp)",
            results[(be, "pte_fd", "pte")]["theta_pp"], wp, 1)
        num(f"Table 9 {PAPER_NAMES[be]}: FD-St gain (pp)",
            results[(be, "pte_fd", "static")]["theta_pp"], wst, 1)
    want_acc9 = {
        "deepseek": (74.0, 74.7, 86.8, 59.7), "qwen": (87.2, 95.1, 97.9, 76.4),
        "kimi": (78.8, 87.8, 93.1, 78.1), "glm": (87.8, 93.8, 96.5, 74.7),
    }
    for be, wants in want_acc9.items():
        for cfg, w in zip(CONFIGS, wants):
            num(f"Table 9 {PAPER_NAMES[be]} {cfg}: accuracy mean",
                marginals[be][cfg]["acc_mean"], w, 1)

    # The Table 9 caption promises that the exact paired gain can exceed the
    # difference of the two rounded accuracy columns by up to 0.1 pp. That gap
    # is the whole reason the caption names the estimator: a reader who simply
    # subtracts the printed accuracies would otherwise get e.g. 86.8-74.7=12.1
    # where the paired value is 35/288 = 12.153 -> 12.2.
    worst = 0.0
    for be in BACKENDS:
        fd_acc = round(marginals[be]["pte_fd"]["acc_mean"], 1)
        for cfg in ("pte", "static"):
            paired = results[(be, "pte_fd", cfg)]["theta_pp"]
            worst = max(worst, abs(paired - (fd_acc - round(marginals[be][cfg]["acc_mean"], 1))))
    checks.append(("Table 9 caption: |paired gain - difference of rounded accuracies| <= 0.1 pp",
                   worst <= 0.1 + 1e-9, f"worst discrepancy = {worst:.3f} pp"))

    # --- Holm families
    fam_i = holm([(be, results[(be, "pte_fd", "static")]["p_clus"]) for be in BACKENDS])
    fam_ii = holm([(be, results[(be, "pte_fd", "pte")]["p_clus"]) for be in BACKENDS]
                  + [("n500", results[("n500", "pte_fd", "pte")]["p_clus"])])
    bound("Family (i) FD-vs-Static: every Holm-adjusted p < 0.001",
          max(a for _, _, a in fam_i), 0.001)
    sci("Family (i): largest Holm-adjusted p = 9.5e-4", max(a for _, _, a in fam_i), 9.5e-4)
    bound("Family (ii) FD-vs-PTE: every Holm-adjusted p < 0.05",
          max(a for _, _, a in fam_ii), 0.05)
    sci("Family (ii): largest Holm-adjusted p = 3.8e-2", max(a for _, _, a in fam_ii), 3.8e-2)
    sci("Family (ii): qwen-max uncorrected cluster-robust p = 3.0e-2",
        results[("qwen", "pte_fd", "pte")]["p_clus"], 3.0e-2)
    bound("n=500 FD vs PTE: Holm-adjusted p < 1e-6",
          dict((l, a) for l, _, a in fam_ii)["n500"], 1e-6)

    # --- Table 7 (tab:exp10b) and the n=500 paragraph
    want7 = {"static": (73.4, (69.36, 77.08), 5207), "pte": (71.2, (67.08, 75.00), 3543),
             "pte_fd": (77.0, (73.11, 80.47), 4531)}
    for cfg, (wacc, wci, wtok) in want7.items():
        m = marginals["n500"][cfg]
        num(f"Table 7 {cfg}: accuracy", m["acc_mean"], wacc, 1)
        checks.append((f"Table 7 {cfg}: Wilson CI matches the committed summary",
                       [round(m["wilson_lo"], 2), round(m["wilson_hi"], 2)] == list(wci)
                       and m["summary_wilson"] == list(wci),
                       f"computed=[{m['wilson_lo']:.2f},{m['wilson_hi']:.2f}] "
                       f"summary={m['summary_wilson']} paper={list(wci)}"))
        num(f"Table 7 {cfg}: tokens/query", m["avg_tokens_mean"], wtok, 0)
    n_fd_pte = results[("n500", "pte_fd", "pte")]
    n_fd_st = results[("n500", "pte_fd", "static")]
    n_pte_st = results[("n500", "pte", "static")]
    cell("n=500 FD vs PTE discordant cells b=29, c=0", (n_fd_pte["b"], n_fd_pte["c"]), (29, 0))
    cell("n=500 FD vs Static discordant cells b=49, c=31", (n_fd_st["b"], n_fd_st["c"]), (49, 31))
    num("n=500 FD vs PTE paired difference (pp)", n_fd_pte["theta_pp"], 5.8, 1)
    ci("n=500 FD vs PTE 95% bootstrap CI", n_fd_pte["ci_boot_pp"], (3.8, 8.0))
    sci("n=500 FD vs PTE exact binomial McNemar p = 3.7e-9", n_fd_pte["p_exact"], 3.7e-9)
    sci("n=500 FD vs PTE continuity-corrected McNemar p = 2.0e-7", n_fd_pte["p_chi2_cc"], 2.0e-7)
    num("n=500 FD vs Static paired difference (pp)", n_fd_st["theta_pp"], 3.6, 1)
    ci("n=500 FD vs Static 95% bootstrap CI", n_fd_st["ci_boot_pp"], (0.2, 7.0))
    num("n=500 FD vs Static cluster-robust Wald p = 0.044", n_fd_st["p_clus"], 0.044, 3)
    num("n=500 FD vs Static continuity-corrected McNemar p = 0.057", n_fd_st["p_chi2_cc"], 0.057, 3)
    num("n=500 FD vs Static bootstrap p = 0.047", n_fd_st["p_boot"], 0.047, 3)
    num("n=500 PTE vs Static paired difference (pp)", n_pte_st["theta_pp"], -2.2, 1)
    ci("n=500 PTE vs Static 95% bootstrap CI", n_pte_st["ci_boot_pp"], (-5.4, 0.8))
    num("n=500 PTE vs Static cluster-robust p = 0.17", n_pte_st["p_clus"], 0.17, 2)

    # --- token-reduction claims
    num("n=500 text: PTE-FD cuts tokens by 13.0%", tokens["n500"]["pte_fd"], 13.0, 1)
    fd_red = {be: tokens[be]["pte_fd"] for be in BACKENDS}
    num("Abstract/L530/L542 lower bound 46% (PTE-FD vs Static)", min(fd_red.values()), 46.0, 0)
    num("Abstract/L530/L542 upper bound 55% (PTE-FD vs Static)", max(fd_red.values()), 55.0, 0)
    others = {be: v for be, v in fd_red.items() if be != "deepseek"}
    num("L534 lower bound 52% (other three back-ends)", min(others.values()), 52.0, 0)
    num("L534 upper bound 55% (other three back-ends)", max(others.values()), 55.0, 0)
    num("Ablation text: deepseek PTE-FD ~46% fewer tokens", fd_red["deepseek"], 46.0, 0)
    num("Ablation text: deepseek PTE ~55% fewer tokens", tokens["deepseek"]["pte"], 55.0, 0)
    checks.append(("Token claim: the 57% endpoint belongs to PTE, not PTE-FD",
                   max(tokens[be]["pte"] for be in BACKENDS) > 57.0 - 0.5
                   and max(fd_red.values()) < 56.0,
                   f"PTE max={max(tokens[be]['pte'] for be in BACKENDS):.2f}% "
                   f"PTE-FD max={max(fd_red.values()):.2f}%"))

    # --- clustering claims
    num("Cluster count G = 118 distinct query texts", float(clusters["G"]), 118.0, 0)
    checks.append(("Cluster size distribution {1:10, 2:46, 3:62}",
                   clusters["size_distribution"] == {1: 10, 2: 46, 3: 62},
                   str(clusters["size_distribution"])))
    infl = [r["se_inflation"] for k, r in results.items()
            if k[0] != "n500" and not math.isnan(r["se_inflation"])]
    num("SE understatement lower bound 18%", (min(infl) - 1) * 100, 18.0, 0)
    num("SE understatement upper bound 58%", (max(infl) - 1) * 100, 58.0, 0)
    return checks


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def emit_comparison(lines: list[str], title: str, r: dict) -> None:
    lines.append(f"  * {title}")
    lines.append(f"      cells      b={r['b']:4d} (A right, B wrong)   c={r['c']:4d} "
                 f"(A wrong, B right)   discordant={r['n_discordant']:4d}   "
                 f"N={r['N']}   clusters G={r['G']}")
    lines.append(f"      theta-hat  {r['theta_pp']:+7.3f} pp")
    lines.append(f"      McNemar chi2 (continuity corrected) = {r['chi2_cc']:9.3f}   "
                 f"p = {r['p_chi2_cc']:.6g}    <- what the submission quoted")
    lines.append(f"      McNemar exact binomial                p = {r['p_exact']:.6g}")
    lines.append(f"      naive Wald (ignores clustering)  SE={r['se_naive_pp']:6.3f} pp  "
                 f"z={r['z_naive']:7.3f}  p={r['p_naive']:.6g}  "
                 f"95% CI=[{r['ci_naive_pp'][0]:+7.2f},{r['ci_naive_pp'][1]:+7.2f}] pp")
    lines.append(f"      cluster-robust Wald  [REPORTED p]  SE={r['se_clus_pp']:6.3f} pp  "
                 f"z={r['z_clus']:7.3f}  p={r['p_clus']:.6g}  "
                 f"95% CI=[{r['ci_clus_pp'][0]:+7.2f},{r['ci_clus_pp'][1]:+7.2f}] pp")
    if "ci_boot_pp" in r:
        lines.append(f"      cluster bootstrap    [REPORTED CI] SE={r.get('boot_se_pp', float('nan')):6.3f} pp  "
                     f"95% CI=[{r['ci_boot_pp'][0]:+7.2f},{r['ci_boot_pp'][1]:+7.2f}] pp  "
                     f"p={r['p_boot']:.6g}  boot mean={r['boot_mean_pp']:+7.2f} pp")
    lines.append(f"      SE inflation from clustering = {r['se_inflation']:.3f}x")
    lines.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boot", type=int, default=BOOT_DEFAULT,
                    help=f"bootstrap resamples (default {BOOT_DEFAULT}; 0 disables)")
    ap.add_argument("--claims-only", action="store_true",
                    help="run the manuscript audit only, skip the full report")
    ap.add_argument("--json", type=Path, default=None, help="write a machine-readable dump")
    args = ap.parse_args()

    results, clusters, marginals, tokens = analyse(args.boot)
    checks = self_checks(results, clusters, args.boot)
    checks += audit_paper_claims(results, clusters, marginals, tokens)

    lines: list[str] = []
    if not args.claims_only:
        lines.append(LINE)
        lines.append("1. Cluster structure induced by the seeded query generator")
        lines.append(LINE)
        lines.append(f"  distinct query texts (clusters) G = {clusters['G']}")
        lines.append(f"  pooled observations N            = {clusters['N']}")
        lines.append(f"  queries per seed                 = {clusters['per_seed']}")
        lines.append(f"  cluster size distribution        = {clusters['size_distribution']}")
        lines.append(f"  pairwise seed overlap            = {clusters['pairwise_overlap']}")
        lines.append("  => observations sharing a query text are positively correlated;")
        lines.append("     methods assuming N independent pairs understate the SE.")
        lines.append("")
        for be in BACKENDS:
            lines.append(LINE)
            lines.append(f"2.{BACKENDS.index(be) + 1} {PAPER_NAMES[be]} "
                         f"(n=96 per seed x 3 seeds, pooled N=288, G={clusters['G']})")
            lines.append(LINE)
            for A, B in PAIRS:
                emit_comparison(lines, f"{A} vs {B}", results[(be, A, B)])
        lines.append(LINE)
        lines.append("3. n=500 bilingual evaluation (exp10b): every query distinct, G = N = 500")
        lines.append(LINE)
        for A, B in N500_PAIRS:
            emit_comparison(lines, f"{A} vs {B}", results[("n500", A, B)])
        lines.append(LINE)
        lines.append("4. Holm-Bonferroni within each declared family (cluster-robust p)")
        lines.append(LINE)
        for name, members in (
            ("(i) PTE-FD vs Static, four back-ends",
             [(be, results[(be, "pte_fd", "static")]["p_clus"]) for be in BACKENDS]),
            ("(ii) PTE-FD vs PTE, four back-ends plus n=500 (five tests)",
             [(be, results[(be, "pte_fd", "pte")]["p_clus"]) for be in BACKENDS]
             + [("n500", results[("n500", "pte_fd", "pte")]["p_clus"])]),
        ):
            lines.append(f"  family {name}")
            for lab, praw, padj in holm(members):
                lines.append(f"      {lab:10s} raw p={praw:12.6g}   Holm-adjusted={padj:12.6g}   "
                             f"{'significant' if padj < 0.05 else 'NOT significant'}")
            lines.append("")
        lines.append(LINE)
        lines.append("5. Marginal accuracies, Wilson intervals and token reduction")
        lines.append(LINE)
        for be in BACKENDS:
            lines.append(f"  {PAPER_NAMES[be]}")
            for cfg in CONFIGS:
                m = marginals[be][cfg]
                red = tokens[be].get(cfg)
                lines.append(f"      {cfg:8s} pooled {m['pooled_correct']:3d}/{m['pooled_N']}  "
                             f"acc={m['acc_mean']:6.2f} +/- {m['acc_sd']:4.2f}  "
                             f"Wilson=[{m['wilson_lo']:6.2f},{m['wilson_hi']:6.2f}]  "
                             f"tokens/q={m['avg_tokens_mean']:8.1f}"
                             + (f"  reduction vs static={red:6.2f}%" if red is not None else ""))
            lines.append("")
        lines.append("  n=500 (exp10b)")
        for cfg in ("static", "pte", "pte_fd"):
            m = marginals["n500"][cfg]
            red = tokens["n500"].get(cfg)
            lines.append(f"      {cfg:8s} {m['pooled_correct']:3d}/{m['pooled_N']}  "
                         f"acc={m['acc_mean']:6.2f}  "
                         f"Wilson=[{m['wilson_lo']:6.2f},{m['wilson_hi']:6.2f}]  "
                         f"tokens/q={m['avg_tokens_mean']:8.2f}"
                         + (f"  reduction vs static={red:6.3f}%" if red is not None else ""))
        lines.append("")

    lines.append(LINE)
    lines.append("6. Self-checks and manuscript audit")
    lines.append(LINE)
    n_fail = 0
    for what, ok, detail in checks:
        if not ok:
            n_fail += 1
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {what}")
        lines.append(f"           {detail}")
    lines.append("")
    lines.append(f"  {len(checks) - n_fail}/{len(checks)} checks passed"
                 + ("  -- ALL OK" if n_fail == 0 else f"  -- {n_fail} FAILURE(S)"))

    report = "\n".join(lines) + "\n"
    out_path = BENCH / "raw_data" / "analysis_stats_revision_report.txt"
    out_path.write_text(report, encoding="utf-8", newline="\n")
    sys.stdout.write(report)
    print(f"\nreport written to {out_path}")

    if args.json:
        dump = {
            "clusters": clusters,
            "comparisons": {"|".join(k): v for k, v in results.items()},
            "marginals": marginals,
            "token_reduction_pct": tokens,
            "checks": [{"what": w, "pass": bool(o), "detail": d} for w, o, d in checks],
        }
        args.json.write_text(json.dumps(dump, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8", newline="\n")
        print(f"json written to {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
