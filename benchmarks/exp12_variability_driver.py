# -*- coding: utf-8 -*-
"""Verification driver for the same-trace comparison table of the ARRAY submission.

What this script is for
-----------------------
Table 3 of the manuscript (``\\label{tab:ablation_ebe}``, subsection
``sec:ablation_cost``) reports four experience-capture mechanisms evaluated on one
shared set of 196 production traces.  Two of the four rows (ERL and ExpeL) are
``mean +/- SD`` over **n = 5 independent end-to-end runs**, because the hosted
back-end is not bit-reproducible even at ``temperature=0``.  This driver is the
single entry point for checking that every number in that table, and every number
in the prose that interprets it, follows from the five shipped result files.

It is deliberately **zero-API by default**: everything below recomputes from
artifacts that ship in this repository.  Re-running the five live repetitions
costs about 328 hosted LLM calls per repetition and is available behind
``--reps`` for authors, not for reviewers.

What is verifiable here, and what is not
----------------------------------------
Verifiable from the shipped artifacts (no network, no quota):

* the ``mean +/- SD`` of all four reported metrics for both LLM-based rows;
* the two deterministic rows (Raw Error Log, EBEAC), which must have SD exactly 0;
* the six EBEAC figures quoted in Section 6.4, per repetition and after aggregation;
* the LLM-call budget per method and the attribution invariant (sum == total);
* the library (KB) sizes, **including EBEAC's**, which ``exp12`` itself cannot
  report -- see the note below;
* the derived quantities used in the prose: one event = 1.69 pp, the SD range
  0.93--1.52 pp, the full range up to 3.39 pp, the precision advantage
  14.8--19.9 pp, the false-positive ratio 6.3--7.5x, and the FP band 21.35--25.42;
* the extraction-drift percentages 83.94% and 91.78% and the five distinct
  library fingerprints, from ``variability_summary.json``.

**Not** recomputable here: the drift percentages are recorded counts
(``n_diff / n_units``) measured while the five repetitions ran.  Recomputing them
from scratch would require the per-call extraction units, i.e. the LLM
checkpoints, which are deliberately **not** shipped (about 100 KB of raw
prompt/response traffic per repetition).  The arithmetic that turns the recorded
counts into the published percentages *is* checked.

Why EBEAC's library size needs this driver
------------------------------------------
``exp12_ebeac_baselines_same_traces.py`` summarises every method through
``summarize()``, which reads ``stats.get("kb_size", 0)``.  The EBEAC path reuses
``exp9_ebeac_log_replay.evaluate_experience_store_memory``, whose
``summarize_stats()`` returns no ``kb_size`` key at all, so EBEAC's library size
is reported as **0** in ``exp12``'s own console summary and in the five shipped
``rep*_report.txt`` files.  That 0 is a *reporting* default, not a measurement.
Section E below measures the real value the honest way: it seeds an actual
``ExperienceStore`` through the shipped ``exp9`` code path and counts the rows
written, exactly as ``_enforce_max_experiences`` does.  The measured value is
what the manuscript prints.

Determinism of this report
--------------------------
The report contains no timestamp, no wall-clock duration and no absolute path.
Layout-dependent facts are reduced to content-addressed ones (a ``layout`` tag
plus the sha256 of every artifact), so re-running this script on any machine with
the same layout reproduces the report byte for byte.  Two sections are
layout-scoped by nature and say so out loud instead of failing silently: section E
prefers the published anonymized snapshot, which is the only copy the artifact
bundle carries, and section F needs the article source, which the bundle
deliberately does not ship.

Language of the shipped artifacts
---------------------------------
The five ``rep*_report.txt`` transcripts are the raw console output of
``exp12_ebeac_baselines_same_traces.py``, whose author-side working language is
Chinese.  They are shipped **unedited**: rewriting a recorded run would destroy
exactly the evidential value that justifies shipping it.  This driver is the
English entry point -- every number a reader needs is recomputed and printed here
in English, so no Chinese has to be read to verify the table.

Usage
-----
    python benchmarks/exp12_variability_driver.py              # verify (zero API)
    python benchmarks/exp12_variability_driver.py --nc-only    # synthetic controls only
    python benchmarks/exp12_variability_driver.py --reps 5     # authors: re-run live (API cost)

Exit codes: 0 = all checks passed, 1 = a check BLOCKED, 2 = the driver itself failed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parent
RAW = BENCH / "raw_data"
PLANNED = BENCH / "planned_artifacts"

#: Canonical in-repo location of the five repetitions.  The author tree and the
#: artifact bundle both use it, so the driver has exactly one layout to support.
REPS_DIR = RAW / "exp12_reps"
#: File names of repetition ``i`` inside ``REPS_DIR``.  An earlier revision also
#: accepted the author's private working-tree names and walked every ancestor
#: directory hunting for them.  Once the records were committed in-repo that walk
#: became dead code, and shipping a script that rummages through ancestor
#: directories is surprising behaviour in a public bundle -- so it is gone.
#: ``WECLAW_E12_REPS_DIR`` remains as an explicit, opt-in override for anyone who
#: wants to point the driver at their own re-runs.
REP_PATTERNS = ("rep{i}_results.json",)
REPORT_PATTERNS = ("rep{i}_report.txt",)
SUMMARY_NAMES = ("variability_summary.json",)

#: Dataset resolution.  The anonymized variant is the published one and is the
#: only copy present in the artifact bundle, so it is preferred unconditionally:
#: that keeps the sha256 line of section E identical everywhere.  The raw export
#: is not published (it carries un-anonymized free text); when it is present as
#: well, section E asserts both yield the same library size, which is the evidence
#: behind the manuscript's claim that anonymization leaves the metrics unchanged.
DATASET_ANON = PLANNED / "weclaw_ebeac_log_replay_realdb_20260808_anon.json"
DATASET_RAW = PLANNED / "weclaw_ebeac_log_replay_realdb_20260808.json"
DATASET_CANDIDATES = (DATASET_ANON, DATASET_RAW)

N_REPS = 5
METHODS = ("raw_log", "erl", "expel", "ebeac")
#: Methods that issue no LLM call must therefore be bit-identical across reps.
DETERMINISTIC = ("raw_log", "ebeac")
METRICS = ("recall_at_1", "recall_at_3", "precision", "coverage_rate",
           "prevention_rate", "false_positive_rate")
COUNTS = ("top1_hits", "top3_hits", "predictions")

#: The four metric columns of Table 3, in table order, with their results.json key.
TABLE3_METRIC_COLUMNS = (("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"),
                         ("Precision", "precision"), ("FP Rate", "false_positive_rate"))
#: Row label prefix in the manuscript -> method key in results.json.
TABLE3_ROWS = (("Raw Error Log", "raw_log"), ("ERL", "erl"), ("ExpeL", "expel"),
               ("EBEAC", "ebeac"))
TABLE3_LABEL = "tab:ablation_ebe"

#: Section 6.4 published EBEAC figures (the production-snapshot paragraph).
PUBLISHED_EBEAC = {"recall_at_1": 71.19, "recall_at_3": 76.27, "precision": 89.36,
                   "coverage_rate": 79.66, "prevention_rate": 76.27,
                   "false_positive_rate": 3.39}
#: LLM-call budget per method, and the library size each method builds.
PUBLISHED_CALLS = {"raw_log": 0, "erl": 196, "expel": 132, "ebeac": 0}
PUBLISHED_KB = {"raw_log": 137, "erl": 137, "expel": 73, "ebeac": 137}
#: Denominator shared by all four methods (the 59 test tool-failure events).
N_TEST_ERRORS = 59
#: Prose quantities printed at TWO decimals in the manuscript.
PROSE_DP2 = {"one_event_pp": "1.69", "sd_lo": "0.93", "sd_hi": "1.52",
             "range_hi": "3.39", "fp_band_lo": "21.35", "fp_band_hi": "25.42",
             "gap_r1_erl": "3.39", "gap_r1_expel": "1.01", "gap_r3": "2.38"}
#: Prose quantities printed at ONE decimal: the precision advantage band and the
#: false-positive ratio band are the only two the manuscript rounds to 0.1.
#: Comparing them at two decimals would report 14.78 against a printed 14.8 and
#: BLOCK on a rounding convention rather than on a real discrepancy.  NC-F1 in
#: section D2 proves this distinction is load-bearing, not decorative.
PROSE_DP1 = {"prec_lo": "14.8", "prec_hi": "19.9",
             "fp_ratio_lo": "6.3", "fp_ratio_hi": "7.5"}
#: Key holding the measured drift figures in variability_summary.json.  There is
#: deliberately no fallback: silently defaulting to the whole document would turn
#: a key-name error into a pile of per-method BLOCKs pointing at the wrong thing.
DRIFT_KEY = "root_cause_measured"

_lines: list[str] = []
_fails: list[str] = []


def say(s: str = "") -> None:
    _lines.append(s)
    print(s, flush=True)


def chk(label: str, got: Any, want: Any) -> None:
    """Single comparator for real assertions and negative controls alike.

    Negative controls must go through the *same* code path as the real checks,
    otherwise a broken comparator would silently pass both.
    """
    ok = got == want
    if not ok:
        _fails.append(label)
    say(f"  [{'ok   ' if ok else 'BLOCK'}] {label:<62s} got={got!r} want={want!r}")


def info(label: str, value: Any) -> None:
    say(f"  [info ] {label:<62s} {value!r}")


def rel(p: Path) -> str:
    """Path relative to ``benchmarks/`` -- never print an absolute path, or the
    report stops being machine-independent and starts leaking the local layout."""
    try:
        return str(p.resolve().relative_to(BENCH)).replace("\\", "/")
    except ValueError:
        return p.name


def _digest_lf(b: bytes) -> str:
    """sha256 over the LF-normalized byte stream -- exactly the blob Git stores.

    Every tracked file in this archive is ``i/lf``, but ``exp12`` wrote the five
    ``rep{i}_results.json`` records with CRLF, because that is what text-mode I/O
    does on Windows.  Digesting the raw on-disk bytes would make this report
    platform-dependent: identical on two Windows checkouts, different on a Linux
    one, where ``core.autocrlf`` leaves the blob's LF intact.  Normalizing removes
    that dependency and makes every digest printed below equal to
    ``git show <tag>:<path> | sha256sum`` on any platform.

    Nothing on disk is modified.  The records keep the bytes ``exp12`` actually
    produced; only the *digest* is taken over the archived form.

    Split out from ``blob_sha256`` on purpose: NC-G in section B calls this
    function directly, so deleting the ``.replace`` below turns NC-G red.  A
    control that only re-implements the rule in its own literal would keep passing
    while the production digest silently became platform-dependent.
    """
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def blob_sha256(p: Path) -> str:
    """Digest of ``p`` under the archive's end-of-line convention.  See ``_digest_lf``."""
    return _digest_lf(p.read_bytes())


# ======================================================================
# A. Layout resolution
# ======================================================================

def resolve_reps_dir() -> Path | None:
    """Find the directory holding the five repetition result files."""
    cands = [REPS_DIR]
    env = os.environ.get("WECLAW_E12_REPS_DIR", "").strip()
    if env:
        cands.insert(0, Path(env))
    for d in cands:
        if not d.is_dir():
            continue
        if all(_rep_file(d, i) for i in range(1, N_REPS + 1)):
            return d
    return None


def _rep_file(d: Path, i: int) -> Path | None:
    for pat in REP_PATTERNS:
        p = d / pat.format(i=i)
        if p.exists():
            return p
    return None


def _rep_report(d: Path, i: int) -> Path | None:
    for pat in REPORT_PATTERNS:
        p = d / pat.format(i=i)
        if p.exists():
            return p
    return None


def resolve_summary(reps_dir: Path | None) -> Path | None:
    cands = []
    if reps_dir:
        cands += [reps_dir / n for n in SUMMARY_NAMES]
    cands += [RAW / n for n in SUMMARY_NAMES]
    for p in cands:
        if p.exists():
            return p
    return None


def resolve_dataset() -> Path | None:
    for p in DATASET_CANDIDATES:
        if p.exists():
            return p
    return None


def resolve_tex() -> Path | None:
    """Find ``array_submission/paper_array.tex`` if this tree carries the article.

    The artifact bundle deliberately ships no article source, so in the bundle
    this returns None and section F announces a loud self-skip instead of failing
    silently -- a skipped check that looks like a passed check is worse than none.
    """
    for anc in [BENCH, *BENCH.parents][:8]:
        p = anc / "array_submission" / "paper_array.tex"
        if p.exists():
            return p
    return None


# ======================================================================
# B. Aggregator
# ======================================================================

def agg(vals: list[float]) -> dict:
    n = len(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return {"n": n, "mean": statistics.fmean(vals), "sd": sd,
            "min": min(vals), "max": max(vals), "range": max(vals) - min(vals),
            "values": list(vals)}


def aggregate(reps: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for m in METHODS:
        out[m] = {}
        for k in METRICS + COUNTS:
            vals = [r["results"][m][k] for r in reps if k in r["results"].get(m, {})]
            if vals:
                out[m][k] = agg(vals)
        out[m]["llm_calls"] = agg([float(r["results"][m].get("llm_calls", 0)) for r in reps])
        # kb_size is absent for EBEAC (exp12 cannot report it -- see section E), so
        # it is aggregated only where the key really exists.  Padding it with 0
        # would print a measured-looking KB of 0 for that row.
        kbs = [float(r["results"][m]["kb_size"]) for r in reps
               if "kb_size" in r["results"].get(m, {})]
        if kbs:
            out[m]["kb_size"] = agg(kbs)
    return out


def fmt2(x: float) -> str:
    """The manuscript prints two decimals; compare the printed form, not a float."""
    return f"{x:.2f}"


def fmt1(x: float) -> str:
    """One-decimal form, for the four prose quantities listed in ``PROSE_DP1``."""
    return f"{x:.1f}"


def synthetic(ratios: dict[str, dict[str, list]], n: int) -> list[dict]:
    reps = []
    for i in range(n):
        res = {}
        for m in METHODS:
            res[m] = {}
            for k in METRICS + COUNTS:
                seq = ratios.get(m, {}).get(k)
                res[m][k] = seq[i] if seq else 0
            res[m]["llm_calls"] = 0 if m in DETERMINISTIC else 1
        reps.append({"results": res})
    return reps


def nc_aggregator() -> None:
    """Synthetic controls proving the aggregator is not blind (zero API)."""
    say("")
    say("=== B. Negative controls on the aggregator (synthetic, zero API) ===")
    # NC-A: identical reps must yield SD and range exactly 0 -- proves the
    # aggregator cannot invent variability.
    same = synthetic({"erl": {"recall_at_1": [74.58] * 3}}, 3)
    a = aggregate(same)["erl"]["recall_at_1"]
    chk("NC-A identical reps -> SD exactly 0", fmt2(a["sd"]), "0.00")
    chk("NC-A identical reps -> range exactly 0", fmt2(a["range"]), "0.00")
    chk("NC-A identical reps -> mean is the common value", fmt2(a["mean"]), "74.58")
    # NC-B: a single-event drift must be measurable.  With a 59-event denominator
    # one event is 100/59 = 1.6949 pp, so (46,47,47)/59 must give range 1.69.
    # (the first draft of this control passed ``fmt2`` strings in, i.e. text where
    #  the aggregator expects numbers; the dead assignment is kept out and the
    #  reason recorded here so the mistake is not repeated)
    one = synthetic({"erl": {"recall_at_1": [round(100 * 46 / 59, 2), round(100 * 47 / 59, 2),
                                             round(100 * 47 / 59, 2)]}}, 3)
    b = aggregate(one)["erl"]["recall_at_1"]
    chk("NC-B one-event drift -> range is one event (1.69 pp)", fmt2(b["range"]), "1.69")
    chk("NC-B one-event drift -> SD is non-zero", b["sd"] > 0, True)
    # NC-C: tampering with one cell must change the aggregate -- proves the
    # aggregator reads data rather than returning constants.
    # (74.58 + 74.58 + 76.27) / 3 = 75.14333..., which prints as 75.14.  The first
    # draft of this control asserted 75.15; the BLOCK was the comparator being
    # right and the expectation being wrong, which is exactly what NC-C is for.
    tam = json.loads(json.dumps(synthetic({"erl": {"recall_at_1": [74.58] * 3}}, 3)))
    tam[2]["results"]["erl"]["recall_at_1"] = 76.27
    chk("NC-C tampering one cell changes the mean",
        fmt2(aggregate(tam)["erl"]["recall_at_1"]["mean"]), "75.14")
    chk("NC-C the untampered aggregate is still 74.58 (the tamper is localized)",
        fmt2(aggregate(synthetic({"erl": {"recall_at_1": [74.58] * 3}}, 3))
             ["erl"]["recall_at_1"]["mean"]), "74.58")
    # NC-D: dropping a repetition must be reported as n=2, never silently padded.
    chk("NC-D dropping one rep reports n=2 (no silent padding)",
        aggregate(synthetic({"erl": {"recall_at_1": [74.58] * 2}}, 2))["erl"]["recall_at_1"]["n"], 2)
    # NC-E: the printed form must round half away from the manuscript's values.
    chk("NC-E rounding 74.578 prints as 74.58", fmt2(74.578), "74.58")
    chk("NC-E rounding 1.1985 prints as 1.20", fmt2(1.1985), "1.20")
    # NC-G: the digest convention announced in the header is load-bearing, not
    # decorative.  exp12 wrote the five results records with CRLF while the
    # archive stores LF blobs, so digesting raw bytes would make this report
    # differ between a Windows and a Linux checkout.  Both halves must hold:
    # normalization makes the two forms agree, and it is not a no-op that would
    # agree on anything -- a control that can never fail proves nothing.
    lf = hashlib.sha256(b'{"a": 1}\n').hexdigest()
    chk("NC-G the production digest of a CRLF record equals its LF blob digest",
        _digest_lf(b'{"a": 1}\r\n'), lf)
    chk("NC-G normalization is not vacuous (the raw CRLF digest differs)",
        hashlib.sha256(b'{"a": 1}\r\n').hexdigest() == lf, False)


# ======================================================================
# C./D. The five shipped repetitions
# ======================================================================

def load_reps(reps_dir: Path) -> list[dict]:
    reps = []
    for i in range(1, N_REPS + 1):
        p = _rep_file(reps_dir, i)
        if p is None:
            say(f"  [BLOCK] rep{i}: no result file in {rel(reps_dir)}")
            _fails.append(f"rep{i} missing")
            continue
        reps.append(json.loads(p.read_text(encoding="utf-8")))
    return reps


def check_reps(reps: list[dict]) -> None:
    say("")
    say("=== C. Per-repetition integrity (each rep carries its own controls) ===")
    # No path in this label: pointing WECLAW_E12_REPS_DIR at private re-runs would
    # otherwise make the report differ between a fresh checkout and such a re-run.
    # Which location was used is announced once, in main().
    chk(f"exactly {N_REPS} repetitions resolved", len(reps), N_REPS)
    for i, r in enumerate(reps, 1):
        chk(f"rep{i} internal control_failures == 0", r.get("control_failures"), 0)
        chk(f"rep{i} internal data_failures == 0", r.get("data_failures"), 0)
        chk(f"rep{i} status == evaluated (not dry_run)", r.get("status"), "evaluated")
        chk(f"rep{i} api_failures == 0 for every method",
            [r["results"][m].get("api_failures") for m in METHODS], [0] * len(METHODS))
        chk(f"rep{i} denominator n_test_errors == {N_TEST_ERRORS} for every method",
            sorted({r["results"][m].get("n_test_errors") for m in METHODS}), [N_TEST_ERRORS])
        # Attribution invariant: the per-method budgets must sum to the run total,
        # otherwise a method is being credited with another method's calls.
        per = sum(r["results"][m].get("llm_calls", 0) for m in METHODS)
        chk(f"rep{i} per-method llm_calls sum == run total",
            per, r.get("counters", {}).get("llm_calls"))
        chk(f"rep{i} per-method llm_calls match the published budget",
            [r["results"][m].get("llm_calls") for m in METHODS],
            [PUBLISHED_CALLS[m] for m in METHODS])
        for m in ("raw_log", "erl", "expel"):
            chk(f"rep{i} {m} kb_size == {PUBLISHED_KB[m]}",
                r["results"][m].get("kb_size"), PUBLISHED_KB[m])
        for k, v in PUBLISHED_EBEAC.items():
            chk(f"rep{i} EBEAC {k} == published Section 6.4 value",
                r["results"]["ebeac"].get(k), v)
        # Do not trust this driver's constants blindly.  Every result file records
        # the Section 6.4 reference values it was checked against at run time, so a
        # constant that drifts away from the shipped artifacts is caught here.
        pr = r.get("published_reference", {})
        chk(f"rep{i} driver constants agree with the file's own published_reference",
            {k: pr.get(k) for k in PUBLISHED_EBEAC}, dict(PUBLISHED_EBEAC))
        chk(f"rep{i} published_reference n_test_errors == {N_TEST_ERRORS}",
            pr.get("n_test_errors"), N_TEST_ERRORS)


def check_aggregate(reps: list[dict]) -> dict:
    say("")
    say("=== D. Aggregation over the five repetitions ===")
    A = aggregate(reps)
    for m in METHODS:
        info(f"{m} n", A[m]["recall_at_1"]["n"])
    # Deterministic methods must not vary at all.
    for m in DETERMINISTIC:
        for k in METRICS:
            if k in A[m]:
                chk(f"{m} is deterministic across reps: SD({k}) == 0",
                    fmt2(A[m][k]["sd"]), "0.00")
    # LLM-based rows must show the variability the manuscript discloses.
    for m in ("erl", "expel"):
        chk(f"{m} LLM Calls is constant across reps", fmt2(A[m]["llm_calls"]["sd"]), "0.00")
    # KB column.  EBEAC is excluded here because exp12 cannot report its library
    # size at all; section E measures it and section F ties the printed cell to
    # that measurement rather than to a constant.
    for label, m in TABLE3_ROWS:
        if m == "ebeac":
            continue
        chk(f"{label}: kb_size recorded in all five runs", "kb_size" in A[m], True)
        if "kb_size" not in A[m]:
            continue
        chk(f"{label}: KB identical across runs", fmt2(A[m]["kb_size"]["sd"]), "0.00")
        chk(f"{label}: KB equals the published library size",
            int(A[m]["kb_size"]["mean"]), PUBLISHED_KB[m])
    say("")
    say("  Aggregated Table 3 values (mean +/- SD, as printed in the manuscript):")
    for label, m in TABLE3_ROWS:
        cells = []
        for _, key in TABLE3_METRIC_COLUMNS:
            a = A[m].get(key)
            if a is None:
                cells.append(f"{key}=MISSING")
                continue
            cells.append(f"{fmt2(a['mean'])}" + (f"+/-{fmt2(a['sd'])}" if m not in DETERMINISTIC
                                                 and a["sd"] > 0 else ""))
        if m == "ebeac":
            kb_s = f"{PUBLISHED_KB[m]}*"
        else:
            kb_s = str(int(A[m]["kb_size"]["mean"])) if "kb_size" in A[m] else "ABSENT"
        say(f"    {label:<16s} calls={int(A[m]['llm_calls']['mean']):>4d} kb={kb_s:>5s}  "
            + "  ".join(cells))
    say("    (* EBEAC's library size is not reported by exp12; section E measures it.)")
    return A


def check_prose_quantities(A: dict) -> None:
    say("")
    say("=== D2. Prose quantities in sec:ablation_cost must follow from the table ===")
    chk("one event on a 59-event denominator prints as 1.69 pp",
        fmt2(100 / N_TEST_ERRORS), PROSE_DP2["one_event_pp"])
    sds = [A[m][k]["sd"] for m in ("erl", "expel") for _, k in TABLE3_METRIC_COLUMNS]
    rngs = [A[m][k]["range"] for m in ("erl", "expel") for _, k in TABLE3_METRIC_COLUMNS]
    chk("lowest SD across the eight LLM cells prints as 0.93",
        fmt2(min(sds)), PROSE_DP2["sd_lo"])
    chk("highest SD across the eight LLM cells prints as 1.52",
        fmt2(max(sds)), PROSE_DP2["sd_hi"])
    chk("largest full range across the eight LLM cells prints as 3.39",
        fmt2(max(rngs)), PROSE_DP2["range_hi"])
    e1, x1, eb1 = (A["erl"]["recall_at_1"]["mean"], A["expel"]["recall_at_1"]["mean"],
                   A["ebeac"]["recall_at_1"]["mean"])
    e3, x3, eb3 = (A["erl"]["recall_at_3"]["mean"], A["expel"]["recall_at_3"]["mean"],
                   A["ebeac"]["recall_at_3"]["mean"])
    chk("recall@1 gap ERL over EBEAC prints as 3.39",
        fmt2(e1 - eb1), PROSE_DP2["gap_r1_erl"])
    chk("recall@1 gap ExpeL over EBEAC prints as 1.01",
        fmt2(x1 - eb1), PROSE_DP2["gap_r1_expel"])
    chk("recall@3 gap (both LLM rows share it) prints as 2.38",
        fmt2(e3 - eb3), PROSE_DP2["gap_r3"])
    chk("recall@3 gap for ExpeL is the same 2.38", fmt2(x3 - eb3), PROSE_DP2["gap_r3"])
    prec = A["ebeac"]["precision"]["mean"]
    others = [A[m]["precision"]["mean"] for m in ("erl", "expel", "raw_log")]
    adv = sorted(prec - o for o in others)
    chk("smallest precision advantage prints as 14.8 (one decimal)",
        fmt1(adv[0]), PROSE_DP1["prec_lo"])
    chk("largest precision advantage prints as 19.9 (one decimal)",
        fmt1(adv[-1]), PROSE_DP1["prec_hi"])
    # NC-F1: the one-decimal convention is load-bearing, not cosmetic.  The same
    # value prints as 14.78 at two decimals, which is NOT what the manuscript says,
    # so a driver that used fmt2 everywhere would BLOCK on a rounding convention.
    chk("NC-F1 the same advantage at two decimals prints 14.78, not 14.8",
        fmt2(adv[0]), "14.78")
    fps = sorted(A[m]["false_positive_rate"]["mean"] for m in ("erl", "expel", "raw_log"))
    eb_fp = A["ebeac"]["false_positive_rate"]["mean"]
    ratios = sorted(f / eb_fp for f in fps)
    chk("lowest FP ratio prints as 6.3 (one decimal)",
        fmt1(ratios[0]), PROSE_DP1["fp_ratio_lo"])
    chk("highest FP ratio prints as 7.5 (one decimal)",
        fmt1(ratios[-1]), PROSE_DP1["fp_ratio_hi"])
    chk("NC-F2 the same ratio at two decimals prints 6.30, not 6.3",
        fmt2(ratios[0]), "6.30")
    chk("lower end of the FP band prints as 21.35",
        fmt2(fps[0]), PROSE_DP2["fp_band_lo"])
    chk("upper end of the FP band prints as 25.42",
        fmt2(fps[-1]), PROSE_DP2["fp_band_hi"])
    chk("under the shared judge prevention_rate == recall_at_3 for every method "
        "(why the manuscript omits the column)",
        [fmt2(A[m]["prevention_rate"]["mean"]) == fmt2(A[m]["recall_at_3"]["mean"])
         for m in METHODS], [True] * len(METHODS))


def check_variability_summary(p: Path | None) -> None:
    say("")
    say("=== D3. Extraction drift behind the disclosed non-reproducibility ===")
    if p is None:
        say("  [BLOCK] no variability summary found -- the 83.94%/91.78% figures "
            "in the manuscript would rest on nothing shipped")
        _fails.append("variability summary missing")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    # Content-addressed, not name-addressed: an earlier revision of this driver also
    # accepted the author's private working-tree name for this file, so printing the
    # name would have made the report layout-dependent.  The digest does not.
    say(f"  sha256(summary) = {blob_sha256(p)}")
    chk("summary records five repetitions", d.get("n_reps"), N_REPS)
    chk("summary denominator is the 59 test failures", d.get("denominator"), N_TEST_ERRORS)
    chk("summary's own pp-per-event prints as the caption's 1.69",
        fmt2(d.get("pp_per_event")), PROSE_DP2["one_event_pp"])
    chk("summary backend is deepseek-chat", d.get("backend", {}).get("model"), "deepseek-chat")
    chk("summary backend temperature is 0, yet the five runs still differ",
        d.get("backend", {}).get("temperature"), 0.0)
    if DRIFT_KEY not in d:
        say(f"  [BLOCK] {DRIFT_KEY!r} is absent from the summary.  Refusing to fall back "
            "to the whole document: that would turn a key-name error into a pile of "
            "per-method BLOCKs pointing at the wrong thing.")
        _fails.append(f"{DRIFT_KEY} absent from the variability summary")
        return
    drift = d[DRIFT_KEY]
    for m, want_pct, want_units in (("erl", "83.94", 137), ("expel", "91.78", 73)):
        e = drift.get(m)
        if not e:
            say(f"  [BLOCK] no drift entry for {m}")
            _fails.append(f"drift entry {m} missing")
            continue
        n_units, n_diff = e.get("n_units"), e.get("n_diff")
        chk(f"{m} extraction units == {want_units}", n_units, want_units)
        chk(f"{m} drift percentage recomputes from n_diff/n_units",
            fmt2(100 * n_diff / n_units), want_pct)
        chk(f"{m} recorded drift_pct agrees with the recomputation",
            fmt2(e.get("drift_pct")), want_pct)
        chk(f"{m} five distinct library fingerprints", e.get("distinct_kb_fp"), 5)
        chk(f"{m} the five fingerprints are actually distinct",
            len(set(e.get("kb_fps", []))), 5)
    chk("input-token spread across the five runs is recorded (3717 tokens)",
        drift.get("in_tok", {}).get("range"), 3717)
    chk("that spread is non-zero, i.e. the back end was fed differently-sized prompts",
        drift.get("in_tok", {}).get("range", 0) > 0, True)


# ======================================================================
# E. EBEAC library size, measured (exp12 cannot report it)
# ======================================================================

def measure_ebeac_kb(dataset: Path | None) -> int | None:
    """Measure EBEAC's library size and return the row count.

    Returns None when it could not be measured.  Callers must treat None as a
    missing measurement and never as 0 -- conflating the two is precisely how the
    bogus ``kb=0`` in exp12's own summary came about.
    """
    say("")
    say("=== E. EBEAC library size, measured through the shipped exp9 code path ===")
    if dataset is None:
        say("  [BLOCK] neither the anonymized nor the raw snapshot is present; "
            "cannot measure the library size")
        _fails.append("dataset missing")
        return None
    info("dataset used", rel(dataset))
    say(f"  sha256(dataset) = {blob_sha256(dataset)}")
    sys.path.insert(0, str(BENCH))
    try:
        # exp9 first: exp12 inserts its own directory at sys.path[0] on import, and
        # if exp9 were not already in sys.modules it could resolve to a different
        # copy of exp9 than the one this repository ships.
        import exp9_ebeac_log_replay as X9
        import exp12_ebeac_baselines_same_traces as X12
    except Exception as e:                                   # pragma: no cover
        say(f"  [BLOCK] cannot import the shipped experiment modules: {e!r}")
        _fails.append("import exp9/exp12 failed")
        return None
    # Compare the resolved locations but print only the verdict.  Printing the paths
    # themselves would embed the author's absolute working-tree path -- including a
    # non-ASCII directory name -- into a report that ships in the public bundle,
    # which is both a layout leak and a break of machine-independence.
    chk("exp9 resolved inside this benchmarks directory",
        Path(X9.__file__).resolve().parent == BENCH, True)
    chk("exp12 resolved inside this benchmarks directory",
        Path(X12.__file__).resolve().parent == BENCH, True)

    items = json.loads(dataset.read_text(encoding="utf-8"))["items"]
    train = [e for e in items if e.get("split") == "train"]
    test = [e for e in items if e.get("split") == "test"]
    sd = X12.seedable(train)
    chk("snapshot carries 196 records", len(items), 196)
    chk("train split == 137", len(train), 137)
    chk("test split == 59", len(test), 59)
    chk("seedable(train) == 137 (the filter exp9's seeder applies)", len(sd), 137)

    n_rows, pats = _seed_and_count(X9, sd)
    chk("EBEAC store holds one experience per seedable record", n_rows, len(sd))
    chk("measured EBEAC library size == the KB cell printed in Table 3",
        n_rows, PUBLISHED_KB["ebeac"])
    info("distinct abstract_pattern values (recall-time dedup candidates)", len(set(pats)))
    say("    The KB column is the *library size* (rows written).  The recall path")
    say("    additionally de-duplicates by abstract_pattern, which is production")
    say("    behaviour and is what Section 6.4 reports; the two counts differ by")
    say(f"    design ({n_rows} stored vs {len(set(pats))} distinct patterns).")

    say("")
    say("  Negative controls: the row counter must track its input, not return 137.")
    chk("NC-E1 seeding 5 records stores 5 rows", _seed_and_count(X9, sd[:5])[0], 5)
    chk("NC-E2 blanking every fix_pattern stores 0 rows",
        _seed_and_count(X9, [dict(e, fix_pattern="") for e in sd])[0], 0)
    chk("NC-E3 marking every record status=ok stores 0 rows",
        _seed_and_count(X9, [dict(e, status="ok") for e in sd])[0], 0)
    chk("NC-E4 dropping the last record stores 136 rows",
        _seed_and_count(X9, sd[:-1])[0], 136)
    say("")
    say("  Root cause of the 0 that exp12's own summary prints:")
    src9 = Path(X9.__file__).read_text(encoding="utf-8")
    src12 = Path(X12.__file__).read_text(encoding="utf-8")
    body = re.search(r"def summarize_stats.*?\n\n", src9, re.S)
    chk("exp9 summarize_stats returns no kb_size key", "kb_size" in (body.group(0) if body else ""), False)
    chk("exp12 falls back to stats.get('kb_size', 0)", 'stats.get("kb_size", 0)' in src12, True)

    # The manuscript claims anonymization provably leaves the metrics unchanged.
    # Where the un-anonymized export is present as well (author tree only), measure
    # it too: that claim should rest on a second measurement, not on assertion.
    if DATASET_RAW.exists() and dataset.resolve() != DATASET_RAW.resolve():
        raw_items = json.loads(DATASET_RAW.read_text(encoding="utf-8"))["items"]
        n_raw, _ = _seed_and_count(X9, X12.seedable(
            [e for e in raw_items if e.get("split") == "train"]))
        chk("the un-anonymized export gives the same library size "
            "(evidence for 'anonymization leaves the metrics unchanged')", n_raw, n_rows)
    else:
        say(f"  [info ] only one of the two snapshots is present in this layout "
            f"({rel(dataset)}), so the raw-versus-anonymized equivalence could not be "
            "re-measured here; it is re-measured wherever both are present")
    return n_rows


def _seed_and_count(X9: Any, events: list[dict]) -> tuple[int, list[str]]:
    """Seed a real ExperienceStore in a temp directory and count the rows written.

    Uses the same SQL as ``ExperienceStore._enforce_max_experiences`` so the count
    is not a second, divergent definition of "library size".
    """
    with tempfile.TemporaryDirectory(prefix="exp12_kb_") as td:
        tp = Path(td)
        store = X9.SqliteOnlyExperienceStore(
            db_path=str(tp / "experiences.db"),
            vector_db_dir=str(tp / "experience_vectors"))
        try:
            asyncio.run(X9.seed_experience_store(store, events))
            n = store._db_conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            rows = store._db_conn.execute("SELECT abstract_pattern FROM experiences").fetchall()
        finally:
            store.close()
    return n, [r[0] for r in rows]


# ======================================================================
# F. Cross-check against the manuscript table (author tree only)
# ======================================================================

def parse_table3(tex: str) -> dict[str, dict[str, tuple[str, str | None]]]:
    """Parse the ``tab:ablation_ebe`` tabular into {method: {column: (mean, sd)}}.

    Scoped to the tabular environment that follows the label.  An unscoped search
    for ``&``-separated rows collides with other tables in this manuscript -- that
    collision was already hit once with a different table and is the reason the
    scope is explicit here.
    """
    lab = tex.find("\\label{" + TABLE3_LABEL + "}")
    if lab < 0:
        raise ValueError(f"label {TABLE3_LABEL} not found")
    begin = tex.find("\\begin{tabular}", lab)
    end = tex.find("\\end{tabular}", begin)
    if begin < 0 or end < 0:
        raise ValueError("tabular environment around the label not found")
    body = tex[begin:end]
    out: dict[str, dict[str, tuple[str, str | None]]] = {}
    for raw in body.split("\\\\"):
        line = raw.strip()
        # Splitting on the row terminator leaves the *next* row prefixed by whatever
        # rule followed the previous one -- here `\midrule` sits directly above the
        # Raw Error Log row.  Without stripping it, startswith() on the row label
        # fails and that row is silently dropped, so the parse would report three
        # rows out of four.  Strip the rules first.
        line = re.sub(r"^(?:\\(?:top|mid|bottom)rule\s*)+", "", line).strip()
        if "&" not in line or "\\textbf{Method}" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 7:
            continue
        head = cells[0]
        method = next((m for pre, m in TABLE3_ROWS if head.startswith(pre)), None)
        if method is None:
            continue
        row: dict[str, tuple[str, str | None]] = {"LLM Calls": (cells[1], None),
                                                  "KB": (cells[2], None)}
        for (col, _), cell in zip(TABLE3_METRIC_COLUMNS, cells[3:]):
            if "\\pm" in cell:
                mean, sd = cell.split("\\pm")
                row[col] = (mean.replace("\\,", "").replace("$", "").strip(),
                            sd.replace("\\,", "").replace("$", "").strip())
            else:
                row[col] = (cell.replace("\\,", "").replace("$", "").strip(), None)
        out[method] = row
    return out


def check_tex(A: dict, kb_ebeac: int | None, tex_path: Path | None) -> None:
    say("")
    say("=== F. Cross-check against the manuscript table ===")
    if tex_path is None:
        say("  [skip ] no array_submission/paper_array.tex in this tree.")
        say("          The artifact bundle ships no article source on purpose, so")
        say("          this section cannot run here.  It is NOT a passed check: the")
        say("          table-to-data comparison is exercised in the author's tree,")
        say("          and sections C/D/D2/D3/E above verify the same numbers")
        say("          against the published constants without needing the source.")
        return
    info("manuscript", rel(tex_path))
    tex = tex_path.read_text(encoding="utf-8")
    T = parse_table3(tex)
    chk("all four Table 3 rows parsed", sorted(T), sorted(m for _, m in TABLE3_ROWS))
    for label, m in TABLE3_ROWS:
        row = T[m]
        chk(f"{label}: LLM Calls cell", row["LLM Calls"][0], str(PUBLISHED_CALLS[m]))
        if m == "ebeac":
            # Tie the printed cell to the live measurement from section E rather than
            # to a constant: this is the only place where the manuscript's EBEAC KB
            # cell is checked against a number that was actually counted.
            kb_want = str(kb_ebeac) if kb_ebeac is not None else "<not measured>"
            chk(f"{label}: KB cell equals the library size measured in section E",
                row["KB"][0], kb_want)
        else:
            chk(f"{label}: KB cell equals the per-run kb_size",
                row["KB"][0],
                str(int(A[m]["kb_size"]["mean"])) if "kb_size" in A[m] else "<absent>")
        for col, key in TABLE3_METRIC_COLUMNS:
            mean_s, sd_s = row[col]
            chk(f"{label}: {col} mean cell", mean_s, fmt2(A[m][key]["mean"]))
            if m in DETERMINISTIC:
                chk(f"{label}: {col} carries no +/- because SD is 0", sd_s, None)
            else:
                chk(f"{label}: {col} SD cell", sd_s, fmt2(A[m][key]["sd"]))


# ======================================================================
# main
# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nc-only", action="store_true",
                    help="run only the synthetic negative controls (no data needed)")
    ap.add_argument("--reps", type=int, default=0,
                    help="authors only: re-run this many live repetitions (API cost)")
    ap.add_argument("--report", default=str(RAW / "exp12_variability_verification.txt"),
                    help="where to write the verification report")
    a = ap.parse_args()

    if a.reps:
        say("[BLOCK] --reps re-runs the experiment against a hosted LLM and is not "
            "implemented in the shipped driver; use the author-side runner.")
        return 1

    say("=" * 100)
    say("exp12 verification driver -- Table 3 (tab:ablation_ebe) from n=5 independent runs")
    say("=" * 100)
    say(f"  benchmarks dir : {rel(BENCH)}")
    say("  zero API       : True (no section below makes a network call)")
    say("  digest rule    : every sha256 below is over the LF-normalized byte stream,")
    say("                   i.e. the blob this archive stores, so it equals")
    say("                   `git show <tag>:<path> | sha256sum` on any platform and any")
    say("                   core.autocrlf value.  Five shipped records are CRLF on the")
    say("                   workstation that produced them, so a plain `sha256sum <file>`")
    say("                   there differs; NC-G below proves the rule is load-bearing.")

    nc_aggregator()
    if a.nc_only:
        return _finish(Path(a.report))

    reps_dir = resolve_reps_dir()
    if reps_dir is None:
        say("")
        say(f"  [BLOCK] no directory holds all {N_REPS} repetition result files "
            f"(looked in {rel(REPS_DIR)}; set WECLAW_E12_REPS_DIR to point the "
            "driver at your own re-runs)")
        _fails.append("reps dir unresolved")
        return _finish(Path(a.report))
    # A layout tag, not a path.  Printing either directory would make the report
    # differ between a fresh checkout and a re-run against private data, whereas
    # the sha256 values below are content-addressed and therefore identical
    # wherever the same bytes happen to live.  The tag says which of the two the
    # run was reading: `in-repo` for the shipped location, anything else only when
    # WECLAW_E12_REPS_DIR was set deliberately.
    say(f"  layout         : {'in-repo' if reps_dir == REPS_DIR else 'override'}")
    for i in range(1, N_REPS + 1):
        f, rp = _rep_file(reps_dir, i), _rep_report(reps_dir, i)
        say(f"    rep{i}: results={'FOUND' if f else 'MISSING'}"
            f"  report={'FOUND' if rp else 'MISSING'}"
            + (f"  sha256={blob_sha256(f)[:16]}" if f else ""))
        if rp is None:
            say(f"  [BLOCK] rep{i} report missing -- the shipped evidence would be "
                "results without the console transcript that produced them")
            _fails.append(f"rep{i} report missing")

    reps = load_reps(reps_dir)
    if len(reps) == N_REPS:
        check_reps(reps)
        A = check_aggregate(reps)
        check_prose_quantities(A)
        check_variability_summary(resolve_summary(reps_dir))
        kb_ebeac = measure_ebeac_kb(resolve_dataset())
        check_tex(A, kb_ebeac, resolve_tex())
    return _finish(Path(a.report))


def _finish(report: Path) -> int:
    say("")
    say("=" * 100)
    if _fails:
        say(f"RESULT: FAIL -- {len(_fails)} check(s) BLOCKED")
        for f in _fails:
            say(f"  - {f}")
    else:
        n_ok = sum(1 for ln in _lines if "[ok   ]" in ln)
        say(f"RESULT: ALL OK -- {n_ok} checks passed, 0 blocked")
    say("=" * 100)
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(_lines) + "\n", encoding="utf-8", newline="\n")
        say(f"report written: {rel(report)}")
    except Exception as e:                                   # pragma: no cover
        say(f"[warn] could not write the report: {e!r}")
    return 1 if _fails else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        say("[EXC] the driver itself failed:")
        say(traceback.format_exc())
        try:
            RAW.mkdir(parents=True, exist_ok=True)
            (RAW / "exp12_variability_verification.txt").write_text(
                "\n".join(_lines) + "\n", encoding="utf-8", newline="\n")
        except Exception:
            pass
        sys.exit(2)
