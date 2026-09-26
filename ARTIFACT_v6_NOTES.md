# ARTIFACT_v6_NOTES

## 1. Relationship between the published tags

| Tag | Commit | Content |
| --- | --- | --- |
| `paper-array-eval-v1`, `paper-array-eval-v2` | `1431ca31` (2026-06-15) | The **single frozen snapshot** from which every number reported in the submission was produced. The two tags are content-identical. Product version `5.30.1` (`src/__init__.py:3`). |
| `paper-array-eval-v3` | `3e9b4ce` (2026-09-14) | One commit on top of the frozen snapshot: a completeness and documentation supplement. No benchmark script, no raw-data file changed. |
| `paper-array-eval-v4` | `acd5c67` (2026-09-15) | One commit on top of v3: recomputation evidence for the revised statistical analysis and for two specific manuscript claims, plus the anonymized production snapshot named in its Section 6.4. |
| `paper-array-eval-v5` | `f340aee` (2026-09-23) | One additive commit on top of v4: the five-repetition variability study (E12) behind Table 3, the same-trajectory four-way comparison script, and a zero-API driver verifying every cell of that table. |
| `paper-array-eval-v6` | *(this tag)* | One additive commit on top of v5: the **real-execution re-scoring study (E11)** behind the `tab:realexec` table — the execution sandbox, the harness that produced the records, the frozen query set, the twelve per-run records plus two aggregates, and a zero-API driver that re-derives every cell of that table from the shipped records. |

v6 is a **strictly additive evidence supplement** relative to v1-v5. It modifies no
file published in v1-v5, changes no experimental input, and re-runs no experiment
that was already published. Every result reported in v1-v5 is therefore unchanged.
v6 delivers exactly the item that `ARTIFACT_v5_NOTES.md` Section 10 listed as
pending (see Section 10 below).

The Zenodo archive `10.5281/zenodo.22014311` captures the **v2** tree only. It does
not contain the v3, v4, v5 or v6 additions; for those, the Git tags are the
authoritative source.

## 2. What v6 adds

Twenty-nine files are added and nothing else: the twenty-eight content files listed
below plus this notes file. "Blob bytes" is the size of the object this repository
stores, i.e. what `git cat-file -s` reports and what the checksums in Section 9 are
taken over; "tree bytes" is the size of the same file in the working tree of the
workstation that produced it. The two differ for the twenty-seven files that were
written with CRLF line endings there (see Section 9); the report is LF-native
because the driver writes it with `newline="\n"`.

| Path | Blob bytes | Tree bytes | Role |
| --- | --- | --- | --- |
| `benchmarks/exp11_realexec_recompute.py` | 35,499 | 36,184 | **Zero-API verification driver — the reviewer-facing entry point.** Re-derives every `tab:realexec` cell, both aggregates and every validity invariant from the twelve frozen records. Standard library only. |
| `benchmarks/raw_data/exp11_realexec_recompute_report.txt` | 10,432 | 10,432 | The report the driver produces **in this bundle** (`bundle` mode, S4/S5 self-skipped): `ALL OK`, 42 checks passed, 0 failed, 2 loud skips; exit 0. |
| `benchmarks/exp11_real_execution.py` | 33,437 | 34,147 | The real-execution harness that produced the records. Provenance only: module-level `dotenv`/`openai`/`src.core.*`/`sandbox_env` imports and a live paid LLM API. **Not on the verification path.** |
| `benchmarks/e11_query_set.py` | 15,193 | 15,538 | Query-set builder: materializes the frozen benchmark queries from the public seed list. |
| `benchmarks/e11_public_queries.py` | 21,289 | 21,589 | The public benchmark queries themselves (Chinese by design — they are the experiment content). |
| `benchmarks/test_sandbox_env.py` | 12,486 | 12,782 | Test suite for the execution sandbox, including the negative-control assertions that the workspace rejects absolute-path escape. |
| `benchmarks/sandbox_env/__init__.py` | 1,105 | 1,146 | Sandbox package exports (`execute`, `SandboxWorkspace`, `SANDBOX_FAMILIES`, `coverage_report`). |
| `benchmarks/sandbox_env/executor.py` | 10,867 | 11,130 | The sandboxed executor: runs a tool call in a workspace and returns an `ExecOutcome`. |
| `benchmarks/sandbox_env/families.py` | 19,886 | 20,340 | Tool-family registry and dispatch. |
| `benchmarks/sandbox_env/families_tier1.py` | 39,992 | 40,764 | Tier-1 tool families (deterministic, in-sandbox). |
| `benchmarks/sandbox_env/families_tier2.py` | 31,071 | 31,751 | Tier-2 tool families (cassette-backed). |
| `benchmarks/sandbox_env/fixtures/sandbox_fixtures.json` | 14,059 | 14,216 | Frozen sandbox fixtures (logged system data). |
| `benchmarks/sandbox_env/fixtures/tier2_cassette.json` | 68,121 | 71,084 | Frozen tier-2 cassette (logged system data). |
| `benchmarks/raw_data/e11_query_set.json` | 106,589 | 111,059 | The frozen query set the harness consumed (Chinese queries = experiment content). |
| `benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed42.json` | 1,044,550 | 1,080,195 | deepseek-v4-flash, full arm (N=200), seed 42 — per-query records. |
| `benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed43.json` | 1,044,145 | 1,079,783 | deepseek-v4-flash, full, seed 43. |
| `benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed44.json` | 1,043,587 | 1,079,236 | deepseek-v4-flash, full, seed 44. |
| `benchmarks/raw_data/runs/exp11/exp11_glm_full_seed42.json` | 1,050,409 | 1,086,227 | glm-4-flashx, full, seed 42. |
| `benchmarks/raw_data/runs/exp11/exp11_glm_full_seed43.json` | 1,051,145 | 1,086,974 | glm-4-flashx, full, seed 43. |
| `benchmarks/raw_data/runs/exp11/exp11_glm_full_seed44.json` | 1,050,869 | 1,086,696 | glm-4-flashx, full, seed 44. |
| `benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed42.json` | 521,711 | 539,691 | qwen-max, subset arm (N=100), seed 42. |
| `benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed43.json` | 525,386 | 543,380 | qwen-max, subset, seed 43. |
| `benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed44.json` | 523,883 | 541,858 | qwen-max, subset, seed 44. |
| `benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed42.json` | 501,722 | 518,792 | kimi-k2.7-code, subset, seed 42. |
| `benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed43.json` | 499,901 | 516,987 | kimi-k2.7-code, subset, seed 43. |
| `benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed44.json` | 500,847 | 517,965 | kimi-k2.7-code, subset, seed 44. |
| `benchmarks/raw_data/runs/exp11/exp11_aggregate_full.json` | 4,148 | 4,364 | Full-arm 3-seed aggregate: per-config means and the three McNemar contrasts. |
| `benchmarks/raw_data/runs/exp11/exp11_aggregate_subset.json` | 4,155 | 4,371 | Subset-arm 3-seed aggregate. |
| `ARTIFACT_v6_NOTES.md` | — | — | This file. |

The twenty-seven CRLF-origin files account for a 332,197-byte difference between the
two columns, which is exactly the number of CR bytes Git drops on the way in. No
other byte differs.

## 3. No previously published file is modified

All twenty-eight content files are new paths — a new `sandbox_env/` package, a new
`raw_data/runs/exp11/` directory, and four new top-level scripts — so no existing
path is touched. Verify after checkout:

```bash
git diff --numstat paper-array-eval-v5 paper-array-eval-v6 -- benchmarks/raw_data/runs/exp11/ benchmarks/sandbox_env/
# -> only added lines: these directories do not exist at v5
git diff --numstat paper-array-eval-v5 paper-array-eval-v6 -- src/ scripts/
# -> empty: no published source module changed
git diff --numstat paper-array-eval-v5 paper-array-eval-v6
# -> twenty-nine added files, zero deleted lines, zero modified lines
```

The five checksums published in `ARTIFACT_v4_NOTES.md` Section 7 and the fourteen in
`ARTIFACT_v5_NOTES.md` Section 9 are unchanged by v6 and still verify.

## 4. Dependencies

**The verification path is standard-library only.** `exp11_realexec_recompute.py`
imports exactly `hashlib`, `json`, `math`, `re`, `sys` and `pathlib`. It reads the
twelve frozen records and re-derives every reported quantity; it calls no network,
no model and no third-party package. A reader needs nothing beyond a Python 3
interpreter to reproduce all 42 bundle-mode checks.

**The harness is provenance, not a dependency of the verification.**
`exp11_real_execution.py` executes, at module level, `from dotenv import load_dotenv`,
`from openai import OpenAI`, `from src.core.prompts import ...`,
`from src.core.tool_exposure import ...`, `import exp1_pte_ablation` and
`from sandbox_env import ...` (lines 79-89). Importing it therefore requires
`python-dotenv`, `openai` and the full `src.core` runtime, and *running* it requires
a live paid LLM API. The recompute driver imports none of these. Two disclosed
quirks, shipped byte-identical rather than patched:

1. `PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()`
   (line 72) resolves to the repository root in the authors' tree but **overshoots it
   in this bundle** (the file sits two levels deep here, not five). This is the same
   class of layout constant documented for `exp12` in `ARTIFACT_v5_NOTES.md` Section
   4. It affects only the harness, which is not on the verification path.
2. `load_dotenv(PROJECT_ROOT / ".env")` reads only; it never writes. No number in the
   submission depends on it.

`requirements.txt` already lists `openai>=1.0`, `python-dotenv>=1.0` and
`numpy>=1.24`. **v6 does not modify `requirements.txt`.**

## 5. Reproducing, and the executability boundary

### 5.1 The zero-API verification (runs anywhere)

```bash
python benchmarks/exp11_realexec_recompute.py
# -> exit 0; the report prints 42 [ok], 2 [skip], 0 [FAIL] and the line
#    "ALL OK (bundle mode)". The report is written to
#    benchmarks/raw_data/exp11_realexec_recompute_report.txt
```

The `[ok]`/`[skip]`/`[FAIL]` tokens and the exit code are ASCII, so the result is
readable regardless of the report's prose language (Section 8).

### 5.2 The shipped report is the bundle-mode variant, and that is deliberate

The driver selects its mode from the presence of `array_submission/`, the directory
that holds the manuscript sources. Because publishing those sources would break
double-anonymization, this bundle does not contain them, so the driver runs in
`bundle` mode and the two sections that cross-check the LaTeX (`S4`, the 40-cell
`tab:realexec` comparison, and `S5`, the adjacent prose numbers) **self-skip loudly**
— printed as `[skip]` with the reason, never as a silent pass. The three-level guard
is `paper` (manuscripts present → S4/S5 run), `bundle` (absent → S4/S5 skip),
`broken` (directory present but a `.tex` missing → S4/S5 **fail**, not skip).

| Mode | Where | `[ok]` | `[skip]` | What runs |
| --- | --- | --- | --- | --- |
| `paper` | authors' tree only (manuscripts present) | 99 | 0 | S0-S6 in full, including the 40-cell table and prose cross-check |
| `bundle` | **this repository** (shipped report) | 42 | 2 | S0-S3 + S6: per-run summaries, both aggregates, all McNemar contrasts, validity controls, synthetic controls |

The 57-check difference is exactly S4 (the 40 `tab:realexec` cells plus their
structural and mirror checks) and S5 (13 adjacent prose numbers) — the checks that
need the manuscript sources. This mirrors the already-public
`exp3_table4_recompute_report.txt`, which is likewise the bundle-mode variant. The
authors'-tree paper-mode report is **not** shipped, because a reader could not
reproduce it from this bundle; shipping it would publish a report the archive cannot
regenerate.

### 5.3 The real execution itself is not reproducible at zero cost

Producing the twelve records required live, paid LLM API calls across four backends,
three seeds, six configurations and up to 200 queries per arm. `kimi-k2.7-code` is
additionally forced to `temperature=1` by its provider, so its outputs are not
bit-reproducible even with a fixed seed. The recompute driver therefore verifies the
**arithmetic and the validity invariants** that lead from the frozen records to every
published number; it does **not** re-run any model, and nothing in v6 claims that the
raw generations can be regenerated for free.

## 6. What E11 measures, and what it found

E11 scores tool use under a **real-execution** criterion that is strictly harder than
the oracle/selection-in-ground-truth criterion used elsewhere in the paper:

> a configuration scores a success on a query only if
> `sandbox.execute(tool, args).status == "success"` **and** the executed tool is in
> that query's `acceptable_tools` set.

Four backends were run over three seeds (42/43/44) with a recovery budget of k=2:
`deepseek-v4-flash` and `glm-4-flashx` on the full arm (N=200 per seed, pooled
N=600), `qwen-max` and `kimi-k2.7-code` on the subset arm (N=100 per seed, pooled
N=300). The re-derived `tab:realexec` (3-seed means; SR = `static_retry`, PF =
`pte_fd`; Tok = pooled tokens; b/c = McNemar discordant pairs, SR-final vs PF-final):

```
Model              arm     St.sgl  SR.fin  P.sgl  PF.fin  b/c    p         Tok SR/PF   dTok
deepseek-v4-flash  full     59.5    64.5    55.7   56.8   54/8   1.1e-08   6528/2383   -63.5%
glm-4-flashx       full     56.3    58.7    53.3   54.7   38/14  1.4e-03   10926/3704  -66.1%
qwen-max           subset   62.0    65.0    54.7   57.3   23/0   4.5e-06   7372/2670   -63.8%
kimi-k2.7-code     subset   27.7    45.0    31.7   41.3   43/32  2.5e-01   4499/1656   -63.2%
```

The driver re-derives every one of these cells from the per-query records and
additionally verifies four validity invariants straight from the raw records (not
synthetic):

- **Conjunctive invariant** — across all 21,600 (record x config x attempt) tuples,
  the count of `ok == 1` that is *not* (`exec_status == success` AND tool in
  `acceptable_tools`) is **0**: the criterion is exactly the conjunction, never an
  oracle shortcut.
- **Zero false gains out of sandbox** — across the 240 out-of-sandbox control
  records x 6 configs, the count of `ok_final == 1` is **0**.
- **Real execution is strictly harder than the oracle** — the number of records the
  oracle would score but real execution fails is **2,867** (> 0), the direct evidence
  that the conjunction binds.
- **Empty-selection rate** — `kimi-k2.7-code` selects no tool on 171 of 300 pooled
  subset queries (57%), which is why its SR-vs-PF McNemar is diluted to
  non-significance (p = 0.25) and why the paper reports that backend's conclusion on
  the b/c counts and effect size rather than on the exact p.

## 7. The one manuscript number the recompute corrected (2382 → 2383)

Re-deriving every cell from the frozen records surfaced a single one-unit
discrepancy. The `deepseek-v4-flash` pooled final-token cell (`Tok PF`) read **2382**
in the manuscript, but the correctly rounded three-seed mean of the per-query token
counts is **2383** (raw mean 2382.55). Eleven of the twelve token cells in the table
use round-half-up; this one had been truncated. The manuscript now reports **2383**.

This is a **manuscript transcription fix, not a change to any shipped record**: the
frozen records always implied 2383, and no other cell, no effect size and the
`dTok` value (-63.5%) are affected. Synthetic negative control ⑤ in the driver pins
the exact-equality cell comparison (`2382 != recomputed`), so an off-by-one in this
cell cannot recur silently. The correction is disclosed here and in the revision log
rather than folded in quietly, precisely because it was found by the artifact's own
recomputation.

## 8. Language of the shipped artifacts

The E11 recompute driver and its report are **Chinese-annotated**, following the
already-public `exp3_table4_recompute.py` precedent (that driver carries 1,361 CJK
characters and its report 583) rather than the `exp12_variability_driver.py`
precedent (English-only, 0 CJK). The two recompute/verification families in this
archive differ in language; v6 follows the recompute family it belongs to. This was
measured against the published tree, not assumed. The verdict tokens
(`[ok]`/`[skip]`/`[FAIL]`), the numbers and the exit code are ASCII, so the result is
machine- and language-neutral.

The remaining files carry Chinese and are shipped **verbatim**:

| File(s) | Disposition |
| --- | --- |
| `exp11_real_execution.py`, `e11_query_set.py`, `test_sandbox_env.py`, `sandbox_env/families*.py` | Comments, docstrings and console strings in the original. Shipping a translated copy would mean shipping a script that is not the one that produced the results. |
| `e11_public_queries.py`, `raw_data/e11_query_set.json` | The benchmark queries are Chinese **by design** — they are the experiment content, not incidental prose. |
| 12 per-run `exp11_*.json` | Logged model input/output on Chinese queries — raw experimental records, never rewritten after the fact (editing them would falsify evidence). |
| `sandbox_env/fixtures/*.json` | Logged sandbox fixtures and cassette (system data). |

This follows the convention `ARTIFACT_v4_NOTES.md` Section 6 established: **logged
system data is never translated.** Three files are naturally CJK-free and were
checked to remain so (`sandbox_env/__init__.py` and the two aggregate JSONs).

No file in v6 contains an absolute author path, a user name, a real e-mail address, a
credential, a repository-owner identifier, or a machine-locating reference to the
authors' private directory layout. All twenty-eight were scanned for those patterns
before staging, with **bidirectional** positive controls: planted samples of each
pattern are caught, while benign look-alikes are explicitly *not* flagged. Two
look-alikes occur naturally and are dispositioned rather than redacted:

- `user@example.com` (one occurrence, inside model-generated tool description text in
  a `kimi` record) is an RFC 2606 reserved documentation domain, not a real address.
- `C:\Windows\system32\x.txt` (one occurrence, in `test_sandbox_env.py`) is a
  **negative-control assertion** proving the sandbox refuses to resolve an absolute
  path outside the workspace. It is a generic OS path, not an author path.

## 9. End-of-line convention and checksums

All twenty-eight digests below are over the **LF-normalized repository blobs** — the
byte stream this repository actually stores — so each equals
`git show paper-array-eval-v6:<path> | sha256sum` on any platform and under any
`core.autocrlf` value. v6 continues the v4/v5 convention and **adds no
`.gitattributes`**.

`core.autocrlf` is `true` here, set by the **system-level** Git configuration shipped
with the Windows installer — not an author preference, and not something a reader on
Linux or macOS will have. Consequently the digests below are **not** what
`sha256sum <file>` prints in a working tree: on Windows every file checks out as CRLF
and differs; on Linux/macOS every file checks out as LF and matches. Use the
`git show` form, which reads the blob and is identical under every setting.

The driver implements this convention itself — `_lf()` hashes
`b.replace(b"\r\n", b"\n")` — and states it in the report header. Synthetic negative
control ⑥ calls that same production function, so deleting the normalization turns
the report red; ⑥b additionally asserts the control is non-vacuous (the raw CRLF
digest differs from the LF digest).

```
30a7ebb8894354723f594103f418cdcadccf9037eba8e62528c5dc16b6256f2e  benchmarks/e11_public_queries.py
3d6a401226ca6a8d885c6aedd86b661d2a2118baf7c811337d6474be16a6ed54  benchmarks/e11_query_set.py
c73e55755aae9f8b8ce59b96c530ea83e047502345ab09efbc54cbd453dda485  benchmarks/exp11_real_execution.py
678b3ae307b766a3a5ae834c831a97d06dcad25d43107444bc69a574860cfae5  benchmarks/exp11_realexec_recompute.py
95c96bee13b01aa34ced8d4874169184616372fad616da060de0050e6ec2e504  benchmarks/raw_data/e11_query_set.json
60a2cdb56893605d270961443f3478f7d9d5655ef477948e74738e0c8898235c  benchmarks/raw_data/exp11_realexec_recompute_report.txt
874ef8107b7d70cbc32678d1e4aaf4e3be9eab7f3468a89f5b468e714265369f  benchmarks/raw_data/runs/exp11/exp11_aggregate_full.json
34e959ee0b06a46c136ea93e05f31c5118962d32073797315de1e3f959489871  benchmarks/raw_data/runs/exp11/exp11_aggregate_subset.json
db11243682fcd9fb77ddc381a3a9df058c0dbe1a0dbbbe75c61e8ae5e65e888a  benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed42.json
0aa8f848de3d36c4c480d052abe3e02242434ec23cc1734835c16723f560e351  benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed43.json
215370b33486889b9ee04c73fe6a5c9beaea977626190de043da59deb09a4ca2  benchmarks/raw_data/runs/exp11/exp11_deepseek_full_seed44.json
aa46f052a239b9a954aec3a4e89ccf396691ec47b9cc7923ba95db8ab00a9541  benchmarks/raw_data/runs/exp11/exp11_glm_full_seed42.json
1e1b34f8dbe45e6c4b4f19fb18cc7becedac44b05c781ebb319bb4308eb10b43  benchmarks/raw_data/runs/exp11/exp11_glm_full_seed43.json
c3c6118c0f29b76df0d953b01259d66247423c40b4a75845659d002ee7c51593  benchmarks/raw_data/runs/exp11/exp11_glm_full_seed44.json
48b3ed0a1d0d5b39d87e223d6211bc22a19f4935c16d0ea1f8f535d5080db939  benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed42.json
fde7db3eab3848f796b1e0d090e9d46c8453bc3ef867378211adbcb291fef2ec  benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed43.json
85a74caa560021a9b9d21baeef0c9889313444c2b7e4131b820919257cb729ce  benchmarks/raw_data/runs/exp11/exp11_kimi_subset_seed44.json
7d56e4cc57ed01ad8c03d113ff15f8f07df02233c3c62179ff1155deecae3b06  benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed42.json
e4a9afcab8dcc53d12eebcd5c44a03e06be1a7d95cfe4e4d6615907cb6f1e25f  benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed43.json
5112f835bca09b14475864834d85eeffb215f61e4164f01ab6db6d7a9401c83e  benchmarks/raw_data/runs/exp11/exp11_qwen_subset_seed44.json
fd81d3c1cae1247513725232154d8a1f7a41e1f72ad7d8f25575b0d78c1e86cc  benchmarks/sandbox_env/__init__.py
08f75c64eb8a0cb9e416967698ebd2bdb67c41eef717d2c6c09ab01d92a02ced  benchmarks/sandbox_env/executor.py
2c39e6d11647131693e049dccc14bf72cc26df1ac3613d9feaccec70b6443cd5  benchmarks/sandbox_env/families.py
ec98045d4d07ba61e9dedc7bb446c504b310702f0c849232bbb72e042d19d677  benchmarks/sandbox_env/families_tier1.py
72c8d6779a487c44540c692f64d290e3f1939002e0af18495d25c5adce9d287e  benchmarks/sandbox_env/families_tier2.py
f20a7287a89874b4b8f458a627c963872166911e0e4d80593eddd0fb6d945075  benchmarks/sandbox_env/fixtures/sandbox_fixtures.json
72d71027bf1ddd9932d961d713a204b85d644bc465df0ddbde49ab778d004f14  benchmarks/sandbox_env/fixtures/tier2_cassette.json
7698a250cc58fe69c5c78698b4ecd1c2439fecbde22075bdb668dae46ee1a851  benchmarks/test_sandbox_env.py
```

Platform-independent verification, reading each blob straight out of the tag:

```bash
for f in $(git ls-tree -r --name-only paper-array-eval-v6 | grep -E 'exp11|e11_|sandbox_env'); do
    git show "paper-array-eval-v6:$f" | sha256sum | sed "s|-|$f|"
done
```

The twelve per-run digests are pairwise distinct, the cheapest available proof that
they are twelve genuine (backend, arm, seed) runs rather than one record copied
twelve times.

## 10. What v6 deliberately does not contain

- **The manuscript sources.** No `.tex`, PDF, cover letter or response letter is
  published here, because the archive version carries the author block. This is the
  reason for the `bundle`-layout boundary in Section 5.2.
- **A runner for new real-execution runs.** Producing new records costs live paid LLM
  API across four backends and three seeds, and `kimi-k2.7-code` at `temperature=1`
  is not bit-reproducible in any case (Section 5.3).
- **The pilot run.** `raw_data/runs/exp11/exp11_kimi_pilot_seed42.json` (a 100-query
  smoke test used to size the subset arm) is deliberately excluded: it is superseded
  by the three seeded subset runs and is not referenced by any reported number.
- **The long-session RCR variability study (E14).** It remains a pending extension.
- **Any modification of a previously published file, or of any previously reported
  number.** The v4 Section 7 and v5 Section 9 checksums still verify unchanged.

**This supersedes the E11-pending bullet of `ARTIFACT_v5_NOTES.md` Section 10.**
E11 — the real-execution re-scoring behind `tab:realexec` — is no longer pending: v6
publishes the sandbox, the harness, the frozen query set, the twelve per-run records
and both aggregates, and a zero-API driver that re-derives every cell of the table
from those records. E14 is still pending, and nothing in v6 claims otherwise.
