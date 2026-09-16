# ARTIFACT_v5_NOTES

## 1. Relationship between the published tags

| Tag | Commit | Content |
| --- | --- | --- |
| `paper-array-eval-v1`, `paper-array-eval-v2` | `1431ca31` (2026-06-15) | The **single frozen snapshot** from which every number reported in the submission was produced. The two tags are content-identical. Product version `5.30.1` (`src/__init__.py:3`). |
| `paper-array-eval-v3` | `3e9b4ce` (2026-09-14) | One commit on top of the frozen snapshot: a completeness and documentation supplement. No benchmark script, no raw-data file changed. |
| `paper-array-eval-v4` | `acd5c67` (2026-09-15) | One commit on top of v3: recomputation evidence for the revised statistical analysis and for two specific manuscript claims, plus the anonymized production snapshot named in Section 6.4. |
| `paper-array-eval-v5` | *(this tag)* | One commit on top of v4: the **five-repetition variability study (E12)** behind Table 3 (`tab:ablation_ebe`), the same-trajectory four-way comparison script that produced it, and a zero-API driver that verifies every cell of that table against the shipped records. |

v5 is a **strictly additive evidence supplement**. It modifies no previously
published file, changes no experimental input, and re-runs no experiment that was
already published. Every result reported in v1-v4 is therefore unchanged; v5 adds
new measurements and makes the revised Table 3 independently recomputable.

The Zenodo archive `10.5281/zenodo.22014311` captures the **v2** tree (85 files,
559.7 kB). It does not contain the v3, v4 or v5 additions; for those, the Git tags
are the authoritative source.

## 2. What v5 adds

Fifteen files are added and nothing else. "Blob bytes" is the size of the object
this repository stores, i.e. what `git cat-file -s` reports and what the checksums
in Section 9 are taken over; "tree bytes" is the size of the same file in the
working tree of the workstation that produced it. The two differ for exactly the
seven files that were written with CRLF line endings there (see Section 9).

| Path | Blob bytes | Tree bytes | Role |
| --- | --- | --- | --- |
| `benchmarks/exp12_ebeac_baselines_same_traces.py` | 48,521 | 49,448 | The experiment itself: scores four methods (Raw Error Log, ERL, ExpeL-style insight, EBEAC) over the **same** 59 test trajectories and the **same** denominator. |
| `benchmarks/exp12_variability_driver.py` | 45,123 | 46,034 | Zero-API verification driver. Re-derives every aggregate, every prose quantity and every Table 3 cell from the eleven shipped records. **This is the reviewer-facing entry point.** |
| `benchmarks/raw_data/exp12_variability_verification.txt` | 25,659 | 25,659 | The report the driver produced in the authors' tree: `RESULT: ALL OK -- 216 checks passed, 0 blocked`. |
| `benchmarks/raw_data/exp12_reps/rep1_results.json` | 3,513 | 3,640 | Machine-readable result of independent run 1. |
| `benchmarks/raw_data/exp12_reps/rep2_results.json` | 3,514 | 3,641 | Independent run 2. |
| `benchmarks/raw_data/exp12_reps/rep3_results.json` | 3,513 | 3,640 | Independent run 3. |
| `benchmarks/raw_data/exp12_reps/rep4_results.json` | 3,514 | 3,641 | Independent run 4. |
| `benchmarks/raw_data/exp12_reps/rep5_results.json` | 3,514 | 3,641 | Independent run 5. |
| `benchmarks/raw_data/exp12_reps/rep1_report.txt` | 8,973 | 8,973 | Original console transcript of run 1, shipped verbatim. |
| `benchmarks/raw_data/exp12_reps/rep2_report.txt` | 8,965 | 8,965 | Run 2 transcript. |
| `benchmarks/raw_data/exp12_reps/rep3_report.txt` | 8,965 | 8,965 | Run 3 transcript. |
| `benchmarks/raw_data/exp12_reps/rep4_report.txt` | 8,965 | 8,965 | Run 4 transcript. |
| `benchmarks/raw_data/exp12_reps/rep5_report.txt` | 8,965 | 8,965 | Run 5 transcript. |
| `benchmarks/raw_data/exp12_reps/variability_summary.json` | 11,920 | 11,920 | The five-run aggregate: mean / SD / range / all five raw values per metric, per-repetition counters, and the measured root cause of the spread. |
| `ARTIFACT_v5_NOTES.md` | — | — | This file. |

The seven non-zero deltas sum to 2,473 bytes, which is exactly the number of CRLF
sequences in those seven files (927 + 911 + 127 x 5). No other byte differs.

## 3. No previously published file is modified

Both new scripts live beside the frozen data they consume, and the eleven records
are new paths under a new directory, so no existing path is touched. Verify after
checkout:

```bash
git diff --numstat paper-array-eval-v4 paper-array-eval-v5 -- benchmarks/raw_data/runs/
# -> empty: the frozen experimental inputs are byte-identical to v4
git diff --numstat paper-array-eval-v4 paper-array-eval-v5 -- src/ scripts/
# -> empty: no published source module changed
git diff --numstat paper-array-eval-v4 paper-array-eval-v5
# -> fifteen added files, zero deleted lines
```

The five checksums published in `ARTIFACT_v4_NOTES.md` Section 7 are unchanged by
v5 and still verify.

## 4. Dependencies

**This section corrects an expectation set by v4.** v4's two scripts import only
the Python standard library. v5's chain does not: `exp12_ebeac_baselines_same_traces.py`
executes `from dotenv import load_dotenv` at module level (line 75), so importing
it — which the driver does, in section E — requires `python-dotenv` to be
installed. The following was measured inside this bundle by importing the chain in
a subprocess and inspecting `sys.modules`, not read off the source:

| Package | Needed for the zero-API verification path? | Why |
| --- | --- | --- |
| `python-dotenv` | **Yes** | Module-level import in `exp12_ebeac_baselines_same_traces.py`. |
| `openai` | No | Imported lazily *inside* the API-calling function (`from openai import OpenAI`, whose trailing source comment marks it as deferred). It never appears in `sys.modules` on the verification path. |
| `numpy` | No | Not imported anywhere in the v5 chain. |
| everything else | No | Standard library only: `argparse`, `asyncio`, `hashlib`, `json`, `os`, `re`, `statistics`, `sys`, `tempfile`, `time`, `collections`, `pathlib`, `typing`. |

`requirements.txt` already lists `openai>=1.0`, `python-dotenv>=1.0` and
`numpy>=1.24`, so `pip install -r requirements.txt` is sufficient. **v5 does not
modify `requirements.txt`.**

Importing the driver alone pulls in nothing: `exp9_ebeac_log_replay` and
`exp12_ebeac_baselines_same_traces` are both imported inside `measure_ebeac_kb()`,
and `exp9` deliberately first. `exp12` inserts its own directory at `sys.path[0]`
on import, so if `exp9` were not already in `sys.modules` the driver could bind to
a different copy of `exp9` than the one this repository ships. The driver asserts
the binding it got (`exp9 resolved inside this benchmarks directory`,
`exp12 resolved inside this benchmarks directory`), so a mis-resolution surfaces as
a blocked check rather than as a silently different measurement.

**Two quirks of `exp12` that a reader should know about, disclosed rather than
patched.** The script is shipped byte-identical to the copy that produced the
results, and two of its module-level constants were written for the authors'
directory depth:

1. `PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()`
   resolves to the repository root in the authors' tree but **overshoots it in
   this bundle** — measured inside the clone, it lands two levels above the
   repository root. Consequently `load_dotenv(PROJECT_ROOT / ".env")` finds no
   file and returns without effect (verified: no `.env` at that location, and the
   import chain exits 0). It reads only; it never writes. A reader who happens to
   keep an unrelated `.env` at that ancestor path would have it loaded into
   `os.environ`; nothing on the zero-API path consults those variables.
2. `DEFAULT_DATASET` names the **un-anonymized** export
   (`weclaw_ebeac_log_replay_realdb_20260808.json`), which this bundle does not
   ship because it carries un-anonymized free text. Running `exp12` directly
   therefore needs `--dataset` pointed at the anonymized snapshot that v4
   published. The driver does not use `DEFAULT_DATASET`: it resolves the
   anonymized variant unconditionally, which is why its section E prints the same
   `sha256(dataset)` on every machine.

**A documentation gap in v1-v4, closed here.** The copy of
`benchmarks/exp9_ebeac_log_replay.py` that this repository has shipped since v1 is
*not* byte-identical to the authors' working copy. Exactly one line differs:

```python
# as published in v1-v4 (and unchanged in v5)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# as it stands in the authors' tree, where the file sits five levels deeper
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
```

Both files are 404 lines; the published variant is the one adapted to this
repository's layout, and it is what makes the replay runnable from a checkout.
None of the notes for v1, v2, v3 or v4 mentioned the adaptation. That was an
omission in the documentation, not a difference in behaviour: `PROJECT_ROOT` is
used only for `sys.path.insert` and for locating `.env`, and no number in the
submission depends on it. It is recorded here so that a reader who diffs the
bundle against anything derived from the authors' tree is not misled.

## 5. Reproducing, and the executability boundary

### 5.1 The zero-API verification (runs anywhere)

```bash
git checkout paper-array-eval-v5
python benchmarks/exp12_variability_driver.py --report /tmp/e12_verify.txt
# -> RESULT: ALL OK -- 174 checks passed, 0 blocked ; exit 0
```

Exit codes: `0` = every check passed, `1` = a check BLOCKED, `2` = the driver
itself failed. `--nc-only` runs just the synthetic negative controls of section B
and needs no data files at all.

**Pass `--report`.** The default is `benchmarks/raw_data/exp12_variability_verification.txt`,
i.e. the shipped report, so a bare invocation overwrites it. Redirecting keeps the
working tree clean.

### 5.2 The shipped report is the 216-check variant, and that is deliberate

The report published here was produced in the authors' tree and ends with
`216 checks passed`. A reader who runs the driver inside this bundle gets **174**
checks and a report that is *not* byte-identical to the shipped one. This is the
opposite of the arrangement v4 chose for `exp3_table4_recompute.py`, and the
reason is that the two variants are not equally informative:

| Layout | Condition | Checks | Section F |
| --- | --- | --- | --- |
| `paper` | `array_submission/paper_array.tex` reachable from `benchmarks/` | **216** | executed in full — all 41 table-to-data comparisons run |
| `bundle` | no manuscript source in the tree (this repository) | **174** | loud self-skip, printed as `[skip ]` with the reason |

The 42-check difference was enumerated mechanically by diffing the two reports'
check labels, and it is fully accounted for:

- **41 checks are section F**, the cell-by-cell cross-check of Table 3 against the
  five-run aggregate. These compare the recomputed values with the LaTeX source of
  the manuscript. The manuscript is not part of this bundle — `paper_array.tex`
  carries the author block, so shipping it would break double-blind review.
- **1 check is in section E**: `the un-anonymized export gives the same library
  size`, which needs both the anonymized and the un-anonymized snapshot present.
  Only the anonymized one is published.
- **0 checks exist in the bundle run but not in the authors' run.** The bundle is
  not a different or weaker test suite; it is a strict subset.

Shipping the 216-check report rather than the 174-check one keeps the strongest
available evidence in the archive: section F is what ties the published Table 3 to
the shipped records, cell by cell, and discarding it would leave the bundle
verifying only its own internal consistency. The cost — that the shipped bytes are
not reproducible from the bundle — is stated here explicitly instead of being left
for a reader to discover.

Detection power is not reduced by the skip. Neither variant can pass silently:

- Section F prints `[skip ] no array_submission/paper_array.tex in this tree.`
  followed by `It is NOT a passed check`, and returns without emitting any `[ok ]`
  line, so the check count itself drops.
- Section E prints that only one of the two snapshots is present and that the
  equivalence therefore cannot be re-measured here.
- Every one of the 42 labels is a function of data the bundle *does* ship, so a
  reader who wants them can obtain the manuscript table values from the published
  article and compare by hand; sections C, D, D2, D3 and E already re-verify the
  same numbers against the published constants without needing the source.

### 5.3 The five repetitions themselves are not reproducible at zero cost

Sections C, D, D2, D3, E and F all read the eleven shipped records; none of them
calls an API. Producing *new* records does: one repetition costs 328 hosted LLM
calls, so five cost 1,640. `--reps N` is present in the argument parser but
deliberately refuses to run:

```
[BLOCK] --reps re-runs the experiment against a hosted LLM and is not
        implemented in the shipped driver; use the author-side runner.
```

The author-side runner is not published. It was a thin loop that invoked
`exp12_ebeac_baselines_same_traces.py` five times, each into its own output
directory, and then aggregated the five `results.json` files — the aggregation is
exactly what `variability_summary.json` records and what the driver re-derives
from scratch in section D, so a reader can audit the aggregation without being
able to reproduce the sampling. Section 6 states why reproducing the sampling
would not reproduce the numbers anyway.

To point the driver at your own re-runs instead of the shipped ones, set
`WECLAW_E12_REPS_DIR`. The report then tags itself `layout : override` rather than
`layout : in-repo`; it never prints the directory, so the report stays
machine-independent either way.

## 6. What the five-repetition study measures, and what it found

**Protocol** (identical in all five runs, recorded in each `rep*_results.json`):
dataset `weclaw_ebeac_log_replay` version `0.3.0-real-db-export` (the 196-record
anonymized production snapshot published in v4), chronological 137 train / 59 test
split, ground truth = the `fix_pattern` field, match rule = exact string equality,
`top_k = 3`, metric formulas identical to `exp9`'s `summarize_stats`, strict
denominator = 59 for every method. Backend `deepseek-chat` at `temperature = 0.0`,
`max_retries = 3`, SDK-level retries disabled. One event is worth 1.6949
percentage points (1/59).

**Table 3 as printed in the manuscript** (mean ± SD over the five runs; a cell
carries no `±` where the SD is exactly 0):

| Method | LLM calls | KB | Recall@1 | Recall@3 | Precision | FP rate |
| --- | --- | --- | --- | --- | --- | --- |
| Raw Error Log | 0 | 137 | 69.49 | 74.58 | 69.49 | 25.42 |
| ERL | 196 | 137 | 74.58 ± 1.20 | 78.65 ± 0.93 | 74.58 ± 1.20 | 21.35 ± 0.93 |
| ExpeL-style | 132 | 73 | 72.20 ± 0.93 | 78.65 ± 1.52 | 72.20 ± 0.93 | 21.35 ± 1.52 |
| **EBEAC (ours)** | 0 | 137\* | 71.19 | 76.27 | 89.36 | 3.39 |

\* EBEAC's library size is not reported by `exp12`; it is measured in section E of
the driver — see Section 7 below. EBEAC additionally reports coverage 79.66 and
cross-session prevention 76.27, both with SD 0.

All 41 cells of this table are recomputed from the shipped records and compared
with the manuscript in section F.

**The spread is real, and it is not in our favour alone.** The two rows whose
retrieval path contains no LLM call — Raw Error Log and EBEAC — have SD exactly
0.00 on every metric across all five runs, with all five raw values identical in
`variability_summary.json`. That is the internal negative control: the harness,
the dataset, the split and the scoring are deterministic. The two rows that do call
an LLM to build their knowledge base drift.

**Measured root cause.** `temperature = 0` did not make the backend deterministic.
The driver verifies the code path is deterministic on our side (list
comprehensions, `sorted`, `enumerate`; no set iteration, hence independent of
`PYTHONHASHSEED`) and then measures where the variation actually enters:

| Quantity | ERL | ExpeL-style |
| --- | --- | --- |
| extraction units compared across the five runs | 137 | 73 |
| units whose extracted text differs between runs | 115 | 67 |
| **drift** | **83.94 %** | **91.78 %** |
| distinct knowledge-base fingerprints across the five runs | 5 | 5 |

Input token counts per run were 577,034 / 575,854 / 576,798 / 576,326 / 579,571 —
a range of 3,717 tokens for byte-identical prompts, which is itself evidence that
the variation originates server-side. Each run made exactly 328 LLM calls with
0 cache hits, 0 extraction failures, 0 parse failures and an empty error-class
map; each recorded `control_failures = 0` and `data_failures = 0`. So the drift is
not caused by dropped or failed calls — every run completed cleanly and still
landed on a different knowledge base.

The practical consequence, and the reason this study exists: a single run of an
LLM-extraction baseline on 59 test items is not a reliable estimate. The observed
ranges are 3.39 pp (two events) for ERL Recall@1 and ExpeL Recall@3, and 1.69 pp
(one event) elsewhere. Table 3 therefore reports mean ± SD rather than a single
number, and the manuscript's Limitations section states the non-reproducibility
explicitly.

## 7. The `kb = 0` line in the shipped transcripts

The final summary table of each `rep*_report.txt` prints `0` in the KB column of
the EBEAC row, while the manuscript's Table 3 prints `137`. Both are correct and
they are not measuring the same thing:

- `exp12` summarises every method through `summarize()`, which reads
  `stats.get("kb_size", 0)`.
- The EBEAC path reuses `exp9_ebeac_log_replay.evaluate_experience_store_memory`,
  whose `summarize_stats()` **returns no `kb_size` key at all**.
- The `.get` default therefore surfaces as a `0` that looks like a measurement.
  **It is a reporting default, not a measurement.**

The other three rows print real values in the same transcripts (Raw Error Log 137,
ERL 137, ExpeL-style 73), which is why the anomaly is confined to the EBEAC row.

Section E of the driver measures the true value the honest way — it seeds an
actual `ExperienceStore` through the shipped `exp9` code path and counts the rows
written — obtaining **137**, and asserts that this equals the KB cell printed in
Table 3. Four negative controls prove the counter tracks its input rather than
returning a constant: seeding 5 records stores 5 rows; blanking every
`fix_pattern` stores 0; marking every record `status = ok` stores 0; dropping the
last record stores 136. Two further checks pin the root cause itself
(`exp9 summarize_stats returns no kb_size key`,
`exp12 falls back to stats.get('kb_size', 0)`), so a future refactor of either
file turns the section red instead of silently changing the answer.

Section E also records that the snapshot holds 196 records splitting 137 / 59, and
that those 137 stored rows carry only **20 distinct `abstract_pattern` values**.
The KB column is the library size (rows written); the recall path additionally
de-duplicates by `abstract_pattern`, which is production behaviour and is what
Section 6.4 of the article reports. The two counts differ by design.

## 8. Language of the shipped artifacts

Three of the fifteen files are English-only and machine-checked to contain zero
CJK characters: the driver, the verification report and
`variability_summary.json`. These are the reviewer-facing artifacts and the
English driver is the entry point.

The remaining files contain Chinese and are shipped **verbatim**:

| File | CJK characters | Disposition |
| --- | --- | --- |
| `exp12_ebeac_baselines_same_traces.py` | 2,948 | Comments and console strings, in the original. Shipping a translated or cleaned-up copy would mean shipping a script that is not the one that produced the results. |
| `rep1..rep5_report.txt` | 605 each | Original console transcripts. Run records are never rewritten after the fact; editing them would falsify evidence. |
| `rep1..rep5_results.json` | 20 each | `backend.selection_basis`, author-side run metadata describing how the model was chosen. |

This follows the convention `ARTIFACT_v4_NOTES.md` Section 6 already established
for the production snapshot: logged system data is never translated. No file in v5
contains an absolute path, a user name, an e-mail address, a credential, a
repository-owner identifier or a reference to the authors' private directory
layout; all fourteen were scanned for those patterns before staging, with positive
controls proving each pattern detects a planted sample.

## 9. End-of-line convention and checksums

All fourteen digests below are over the **LF-normalized repository blobs** — the
byte stream this repository actually stores — so each equals
`git show paper-array-eval-v5:<path> | sha256sum` on any platform and under any
`core.autocrlf` value. v5 continues v4's convention and **adds no
`.gitattributes`**.

That decision was measured, not assumed. Inside a clone of this repository:

| Observation | Result |
| --- | --- |
| `core.autocrlf` at local and global scope | unset (both `git config` calls exit 1) |
| `core.autocrlf` in effect | `true`, set by the **system-level** Git configuration shipped with the Windows installer — not an author preference, and not something a reader on Linux or macOS will have |
| `.gitattributes` files anywhere in the repository | 0 (so v4's published statement that LF is the archive's convention still holds at HEAD) |
| `git check-attr text eol` on a v5 record | both `unspecified` |
| A fresh `git clone` of v4, then reading the working tree | all five v4 text files arrive as **CRLF** although their blobs are LF |
| `git hash-object --path <p>` on a CRLF record vs. on its LF copy | **same object id** — the clean filter normalizes on the way in |
| the same with `--no-filters` | **different object ids** — proving the normalization is what makes them equal, not an identity mapping |

So a reader on Windows who runs `sha256sum <file>` in a working tree gets a
different digest for the seven CRLF-origin files listed in Section 2, and the same
digest for the other seven. This is a property of the local end-of-line setting,
not of the archive. Marking the v5 paths `-text` in a new `.gitattributes` would
have preserved the working-tree bytes, but it was rejected: it contradicts the
statement already published in v4 Section 7, and it would leave the repository with
two competing end-of-line conventions. The rule "original records are never
rewritten" governs the *content* of a record, not the transport-level line ending
Git applies to it, and every Windows-produced text file published since v1 is
already an LF blob.

The driver implements this convention itself — `_digest_lf()` hashes
`b.replace(b"\r\n", b"\n")` — and states it in the report header. Negative control
NC-G calls that same production function, so deleting the normalization turns the
report red; the control was verified to have detection power by mutation testing
(removing the `.replace`, and separately tampering with the control's expected
value, each produce a distinguishable failure, while the unmutated driver passes).

```
e6d153d819bcc2cf099dcccb6a2d95ea0b6e861d899588ba408f8034173fc749  benchmarks/exp12_ebeac_baselines_same_traces.py
dba072de4d05b4f5f12005dc2d95c0554f4844526e68241dead8ce13cbd6c449  benchmarks/exp12_variability_driver.py
b23dedd3f0c82bde9fca04513f79686840ef0c2e32bdd1467f13253191e17dd5  benchmarks/raw_data/exp12_variability_verification.txt
d6890532d8f16ece34b9636abf3c1724476772df968167cefefde922c6c3cbeb  benchmarks/raw_data/exp12_reps/rep1_report.txt
924c5ddadca5ab35aa918cf9d4248f2051da68b11f7bbdef77481fd06a28abf2  benchmarks/raw_data/exp12_reps/rep1_results.json
e94a28160fcb08f39f169550d96cd7a6618c04be489c6c4984c76f9295722fa3  benchmarks/raw_data/exp12_reps/rep2_report.txt
7e7fe309476f3d87f72d4e965603b1760cb9a7b61119844527376db805e9f561  benchmarks/raw_data/exp12_reps/rep2_results.json
20652c43f6b59963029a843d2a2cde71a821cbd87d2f07c45d9c22e7c73352af  benchmarks/raw_data/exp12_reps/rep3_report.txt
198e6175756ba19d387b0087c7b15bcbdda2862f421415aec7c13e7a23bae256  benchmarks/raw_data/exp12_reps/rep3_results.json
2f17e7980b7bf858b2fb82daa289f15aeeebf396bd5e1b03777283539e7d6911  benchmarks/raw_data/exp12_reps/rep4_report.txt
cd080e8127aa2d56b1c300271f4add54ef3fb7bf4d8cbf86e6d55f0cd4c3db15  benchmarks/raw_data/exp12_reps/rep4_results.json
185e9e5ab192133c68c4d18bd4d2265b5eb921d3bba642593f4d21e20a0c323b  benchmarks/raw_data/exp12_reps/rep5_report.txt
4c298ae58042e8ac3f7bbb2ca4a8bc70a7accf9ddf252ee33fa9dcc6e5204ce3  benchmarks/raw_data/exp12_reps/rep5_results.json
d6a7f285002512314de8a038f08913eb8e61fbcf5b29a8e9f84c0ba5852243e4  benchmarks/raw_data/exp12_reps/variability_summary.json
```

Platform-independent verification, reading each blob straight out of the tag:

```bash
for f in $(git ls-tree -r --name-only paper-array-eval-v5 | grep exp12); do
    git show "paper-array-eval-v5:$f" | sha256sum | sed "s|-|$f|"
done
```

The five `rep*_results.json` digests are pairwise distinct, which is the cheapest
available proof that they are five genuine runs rather than one record copied five
times; the driver additionally compares the five knowledge-base fingerprints and
requires five distinct values per method.

## 10. What v5 deliberately does not contain

- **The manuscript sources.** No `.tex`, PDF, cover letter or response letter is
  published here, because the archive version carries the author block. This is
  the reason for the `bundle`-layout boundary in Section 5.2.
- **A runner for new repetitions.** `--reps` refuses to execute; producing new
  records costs 1,640 hosted LLM calls (Section 5.3).
- **The un-anonymized production export.** Only the `_anon` snapshot published in
  v4 is shipped, which is why one section E check cannot run here.
- **The real-execution benchmark (E11) and the long-session RCR variability study
  (E14).** These remain pending extensions.
- **Any modification of a previously published file, or of any previously reported
  number.** The five checksums in v4 Section 7 still verify unchanged.

**This supersedes the second bullet of `ARTIFACT_v4_NOTES.md` Section 8**, which
listed E11, E12 and E14 together as pending extensions. E12 — the ERL/ExpeL
same-trajectory comparison — is no longer pending: v5 publishes the script, the
five independent runs and the verification driver, and the manuscript's Table 3
has been revised to report mean ± SD from those runs. E11 and E14 are still
pending, and nothing in v5 claims otherwise.
