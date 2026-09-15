# Artifact Notes for tag `paper-array-eval-v4`

## 1. Relationship between the published tags

| Tag | Commit | Content |
| --- | --- | --- |
| `paper-array-eval-v1`, `paper-array-eval-v2` | `1431ca31` (2026-06-15) | The **single frozen snapshot** from which every number reported in the submission was produced. The two tags are content-identical. Product version `5.30.1` (`src/__init__.py:3`). |
| `paper-array-eval-v3` | `3e9b4ce` | One commit on top of the frozen snapshot: a completeness and documentation supplement (four previously unpublished source files + a 4-line additive restoration of `src/__init__.py`). No benchmark script, no raw-data file changed. |
| `paper-array-eval-v4` | *(this tag)* | One commit on top of v3: the **recomputation evidence** for the revised statistical analysis and for two specific manuscript claims, plus the anonymized production snapshot named in Section 6.4. |

v4 is a **strictly additive evidence supplement**. It modifies no previously
published file, changes no experimental input, and re-runs no experiment. Every
result reported in v1/v2/v3 is therefore unchanged; v4 only makes three of them
independently recomputable by a reader.

The Zenodo archive `10.5281/zenodo.22014311` captures the **v2** tree (85 files,
559.7 kB). It does not contain the v3 or v4 additions; for those, the Git tags
are the authoritative source.

## 2. What v4 adds

| Path | Bytes | Role |
| --- | --- | --- |
| `benchmarks/analysis_stats_revision.py` | 40,697 | Recomputes every cluster-robust statistic, Holm-adjusted *p*-value and cluster-bootstrap confidence interval quoted in the revised manuscript (Tables 2, 7, 9 and the accompanying prose). |
| `benchmarks/raw_data/analysis_stats_revision_report.txt` | 30,045 | The deterministic report produced by the script above: `108/108 checks passed -- ALL OK`. |
| `benchmarks/exp3_table4_recompute.py` | 17,672 | Recomputes every cell of Table 4 (`tab:ablation_rcr`) from the frozen per-conversation lists in `raw_data/exp3_results.json`, including the `14.6 -> 0` API-error claim. |
| `benchmarks/raw_data/exp3_table4_recompute_report.txt` | 6,857 | The report produced **inside this bundle** (see the executability boundary in Section 5). |
| `benchmarks/planned_artifacts/weclaw_ebeac_log_replay_realdb_20260808_anon.json` | 192,103 | The anonymized production `experiences.db` snapshot named in Section 6.4 (196 records). |
| `ARTIFACT_v4_NOTES.md` | — | This file. |

## 3. No previously published file is modified

Both new scripts live beside the frozen data they consume and write their
reports next to themselves, so no existing path is touched. Verify after
checkout:

```bash
git diff --numstat paper-array-eval-v3 paper-array-eval-v4 -- benchmarks/raw_data/runs/
# -> empty: the frozen experimental inputs are byte-identical to v3
git diff --numstat paper-array-eval-v3 paper-array-eval-v4 -- src/ scripts/
# -> empty: no published source module changed
git diff --numstat paper-array-eval-v3 paper-array-eval-v4
# -> six added files, zero deleted lines
```

## 4. Dependencies

Both new scripts import **only the Python standard library**; neither requires a
network connection, an API key, or a third-party numerical package.

| File | Standard-library imports |
| --- | --- |
| `benchmarks/analysis_stats_revision.py` | `argparse`, `ast`, `json`, `math`, `random`, `sys`, `zlib`, `collections`, `pathlib` |
| `benchmarks/exp3_table4_recompute.py` | `json`, `re`, `sys`, `pathlib` |

The cluster bootstrap and the cluster-robust Wald statistic are implemented
directly on `math` primitives with a seeded `random.Random`, which is why the
reported intervals are reproducible to the last digit without NumPy or SciPy.
`requirements.txt` is unchanged.

All **input** data these scripts read were already published in v3 and are not
re-shipped:

| Input | Location in this repository |
| --- | --- |
| `exp10b_static_results.json` | `benchmarks/raw_data/runs/20260608_n500_static/` |
| `exp10b_pte_results.json` | `benchmarks/raw_data/runs/20260608_n500_pte/` |
| `exp10b_pte-fd_results.json` | `benchmarks/raw_data/runs/20260608_n500/` |
| `recanon_aggregate.json` | `benchmarks/raw_data/runs/recanon/` |
| `exp1_pte_ablation.py` (parsed for the intent-taxonomy cross-check) | `benchmarks/` |
| `exp3_results.json`, `exp3_summary.json` | `benchmarks/raw_data/` |

## 5. Reproducing, and the executability boundary

### 5.1 Statistical re-analysis (no boundary)

```bash
git checkout paper-array-eval-v4
cd benchmarks
python analysis_stats_revision.py            # -> 108/108 checks passed -- ALL OK, exit 0
```

The script rewrites `raw_data/analysis_stats_revision_report.txt` on every run.
That report is **byte-identical** to the one shipped here: it contains no
timestamp, no hostname and no absolute path. Verified by running it twice and
comparing SHA-256. The report produced inside this bundle is also byte-identical
to the one produced in the authors' working tree, because the script never reads
the manuscript.

The clustering is itself one of the 108 checks rather than an assumption: the
script asserts `G = 118 distinct query texts` against the pooled records (288
observations per back-end, i.e. 96 queries x 3 seeds) and then recomputes every
cluster-robust Wald statistic and every cluster-bootstrap interval from those
118 clusters. A reader who disagrees with the clustering therefore sees the
disagreement as a failed check, not as a silent difference in the intervals.

### 5.2 Table 4 recomputation (deliberate boundary)

```bash
cd benchmarks
python exp3_table4_recompute.py              # -> exit 0, report ends with "运行模式: bundle"
```

Sections S1, S2 and S5 of this script run in full everywhere: S1 recomputes all
24 aggregate cells (6 pipeline configurations x 4 metrics) from the frozen
per-conversation lists and cross-checks them against `exp3_summary.json`; S2
verifies orphan conservation (`14.6 = 439 injected orphans / 30 conversations`)
and the RCR repair semantics (`0` from stage S4 onward, `53.9` for LLMLingua);
S5 holds seven negative controls.

Sections S3 and S4 additionally cross-check the recomputed values against the
LaTeX source of both manuscripts, cell by cell. **The manuscript sources are not
part of this bundle**: `paper_array.tex` carries the author block, so shipping it
would break double-blind review. The script detects this and reports the two
sections as `SKIPPED` with an explicit reason rather than failing or silently
passing. It distinguishes three modes:

| Mode | Condition | Behaviour |
| --- | --- | --- |
| `paper` | `array_submission/` and both `.tex` files present | S3/S4 executed in full |
| `bundle` | `array_submission/` absent (this repository) | S3/S4 reported `SKIPPED`, not counted as failures |
| `broken` | `array_submission/` present but a `.tex` file missing | S3/S4 reported `FAIL` — a skip is never allowed to mask a damaged workspace |

The shipped report is the `bundle`-mode one, so a reader who runs the command
above reproduces it byte for byte: **64 `[ok]`, 2 `[skip]`, 0 `[FAIL]`, exit 0**.
For completeness, the same script run in `paper` mode in the authors' working
tree reports **119 `[ok]`, 0 `[skip]`, 0 `[FAIL]`**, the additional 55 checks
being the 24 Table 4 cells plus the 12 prose numbers verified against each of the
two manuscripts. That variant is intentionally not shipped, since it cannot be
reproduced from a double-blind bundle.

Detection power is not reduced by the skip: the seven S5 negative controls are
pure functions of the recomputed values and run in both modes. Two of them exist
specifically to prove that the S3 table parser is correctly scoped to
`tab:ablation_rcr` — the manuscript's related-work comparison table contains a
row whose label also begins with `LLMLingua`, and an unscoped parser silently
picks it up.

### 5.3 Production snapshot replay (no boundary, zero API)

The replay script itself was already published in v3; v4 adds only the dataset.

```bash
python benchmarks/exp9_ebeac_log_replay.py \
    --dataset benchmarks/planned_artifacts/weclaw_ebeac_log_replay_realdb_20260808_anon.json \
    --output /tmp/exp9_replay.json
```

`exp9_ebeac_log_replay.py` uses a `SqliteOnlyExperienceStore` adapter (source
comment: *"SQLite-only benchmark adapter to keep exp9 reproducible offline"*)
and contains no `openai` import, so the replay needs no API quota and no
network. Running it on the shipped snapshot reproduces, under
`summary.baselines.experience_store_memory`, exactly the figures quoted in
Section 6.4:

| Section 6.4 claim | Key in the replay output | Value |
| --- | --- | --- |
| 71.19% recall@1 | `recall_at_1` | 71.19 |
| 76.27% recall@3 | `recall_at_3` | 76.27 |
| 89.36% precision | `precision` | 89.36 |
| 79.66% coverage | `coverage_rate` | 79.66 |
| 76.27% cross-session prevention | `prevention_rate` | 76.27 |
| 3.39% false-positive rate | `false_positive_rate` | 3.39 |
| sub-millisecond latency | `avg_latency_ms` | 0.7744 / 0.7715 ms across two runs on the build host (both < 1 ms); **the only machine-dependent entry** |
| 137 train / 59 test | `summary.n_train`, `summary.n_test` | 137, 59 |
| 20 tools, 7 error types | `summary.tools`, `summary.error_types` | 20, 7 |

The manuscript states that anonymization leaves the reported metrics unchanged
and that *"only wall-clock latency differs"* between runs. Both halves were
machine-checked by running the replay twice on the anonymized artifact: all
non-latency keys are bit-identical, and the only four keys that differ are the
`avg_latency_ms` of the four memory baselines. The `no_memory` ablation baseline
in the same output is zero on all seven metrics with `miss_rate = 100.0`, which
is the internal negative control showing these figures are measured rather than
hard-coded.

`avg_latency_ms` is therefore the single entry in the table above that a reader
will *not* reproduce digit for digit -- it is wall-clock time on the reader's own
host. The claim it supports is an order-of-magnitude one ("sub-millisecond"), not
an exact value; every other row is deterministic and reproduces exactly.

## 6. The anonymized production snapshot

| Field | Value |
| --- | --- |
| `dataset_name` | `weclaw_ebeac_log_replay` |
| `version` | `0.3.0-real-db-export` |
| Source | 1 ExperienceStore SQLite snapshot, `exp9_snapshot_20260808.db` |
| **Export date** | 2026-08-08 |
| **Export window** (record timestamps) | 2026-05-30T08:40:06.093694 .. 2026-08-07T18:14:00.496846 |
| Rows before / after deduplication | 196 / 196 |
| Train / test split (`train_ratio = 0.7`, chronological) | 137 / 59 |
| Records | 196, all `status = error`, over 20 distinct tools and 7 error types |
| **SHA-256** | `dcd3d6f4bf472d4e566e6c0c727c33a88dd503703362a4d082c7db52f7bc6139` |

Error-type distribution: `generic_tool_retry` 155, `login_redirect_loop` 22,
`permission_error` 9, `unsupported_format` 4, `parse_error` 3,
`network_error` 2, `timeout_error` 1 (total 196).

Section 6.4 reports 197 raw rows of which one was dropped for carrying no
parsable `tool_names`. That drop happens while reading the database, i.e.
upstream of the counters above, so `rows_before_dedup` is already 196; the two
statements are consistent and refer to different stages.

**Filename convention.** The `_anon` suffix is not a deviation from the sibling
files in `planned_artifacts/`: `run_exp9_real_replay_pipeline.py:123` resolves
datasets as `weclaw_ebeac_log_replay_{tag}.json`, so this file is exactly the
member of that family with `tag = realdb_20260808_anon`. The suffix records that
this variant was produced by field-level anonymization rather than by a plain
export.

**Anonymization, mechanically verified on the shipped bytes.** Absolute paths
are mapped to placeholders and free-text e-mail addresses are replaced by a
fixed dummy address; every field used for retrieval scoring (`tool_name`,
`action`, `error_type`, `fix_pattern`, `timestamp`, `split`) is preserved
verbatim.

| Check | Result |
| --- | --- |
| `<HOME>` placeholders | 203 occurrences |
| `<PROJECT>` placeholders | 26 occurrences |
| All e-mail-shaped tokens in the file | exactly one distinct value, `anon@example.com`, 3 occurrences |
| Windows absolute paths (`X:\Users\`, `C:/Users/`) | 0 |
| Author name, affiliation, real repository name, real e-mail domain | 0 |

Free-text `diagnosis` and `fix_pattern` values are production output and remain
in their original Chinese, consistent with the repository's convention of never
translating logged system data.

## 7. Checksums

All five digests are over the **LF-normalized repository blobs** — the byte
stream this repository actually stores. Every tracked file in v1-v3 is `i/lf` and
there is no `.gitattributes`, so LF is the archive's convention, not an accident
of one workstation.

```
58b9bb738201ec00433d3e8c01cb619f04246c88da49660230b986a55cf06827  benchmarks/analysis_stats_revision.py
369e42776ed3cbdd6c8edf10772ed8dbc53fcdaa1d14901687c1fe14f6ee63be  benchmarks/exp3_table4_recompute.py
dcd3d6f4bf472d4e566e6c0c727c33a88dd503703362a4d082c7db52f7bc6139  benchmarks/planned_artifacts/weclaw_ebeac_log_replay_realdb_20260808_anon.json
581af5e765bdaa185194488f3254eed3711c4609468f7cb8987c8ce3aea3af6b  benchmarks/raw_data/analysis_stats_revision_report.txt
6b67b7d6f87549bf465ca9c380d20e9a6a2acb1f7c6d17ea223482b5d28c241e  benchmarks/raw_data/exp3_table4_recompute_report.txt
```

The platform-independent way to verify them reads each blob straight out of the
tag, so no local end-of-line setting can affect the result:

```bash
for f in benchmarks/analysis_stats_revision.py \
         benchmarks/exp3_table4_recompute.py \
         benchmarks/planned_artifacts/weclaw_ebeac_log_replay_realdb_20260808_anon.json \
         benchmarks/raw_data/analysis_stats_revision_report.txt \
         benchmarks/raw_data/exp3_table4_recompute_report.txt; do
    git show "paper-array-eval-v4:$f" | sha256sum | sed "s|-|$f|"
done
```

On a **Windows** checkout with `core.autocrlf=true`, Git rewrites every text file
to CRLF in the working tree, so a plain `sha256sum <file>` there yields a
different digest. That is a property of the local end-of-line setting, not of the
archive; the `git show` form above is authoritative on every platform. The two
report files are additionally self-verifying: both scripts write them with an
explicit `newline="\n"`, so a reader who **regenerates** a report obtains exactly
the bytes whose digest is listed here, on any platform and with any `autocrlf`
value.

## 8. What v4 deliberately does not contain

- **The manuscript sources.** No `.tex`, PDF, cover letter or response letter is
  published here, because the archive version carries the author block. This is
  the reason for the `bundle`-mode boundary in Section 5.2.
- **The real-execution benchmark (E11), the ERL/ExpeL same-trajectory comparison
  (E12) and the long-session RCR variability study (E14).** These are listed in
  the manuscript as pending extensions; nothing in v4 claims otherwise.
- **Any change to previously reported numbers.** v4 adds recomputation evidence
  for figures that were already published, plus the enlarged production snapshot
  that Section 6.4 describes as superseding the earlier 11-record snapshot.
