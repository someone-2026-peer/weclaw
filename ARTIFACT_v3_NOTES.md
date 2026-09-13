# Artifact Notes for tag `paper-array-eval-v3`

## 1. Relationship between the published tags

`paper-array-eval-v1` and `paper-array-eval-v2` both point to the same commit
`1431ca31` (2026-06-15). That commit is the **single frozen snapshot** from
which every number reported in the submission was produced; there is no content
difference between the two tags.

`paper-array-eval-v3` adds exactly one commit on top of that frozen snapshot.
It is a **completeness and documentation supplement**. It changes no benchmark
script, no raw-data file, and no previously published source module other than
the additive version declaration described below. All reported results are
therefore unchanged: v1, v2 and v3 share the same experimental evidence.

## 2. What v3 adds

| Path | Role | Why it is needed |
| --- | --- | --- |
| `scripts/watchdog.py` | External process supervisor distinguishing four termination states | Named explicitly in the paper's *Production Reliability Layer* section, item (1) |
| `src/permissions/audit.py` | `AuditLogger`; EventBus subscriber at priority 50 | The paper describes subscriber priorities and names the audit logger |
| `src/permissions/__init__.py` | Package marker (empty file) | Required for `src.permissions.audit` to be importable |
| `src/core/session.py` | `SessionManager`; the RCR call site for prefix preservation | Supports the prefix-stability discussion and closes the import graph of the published modules |

One previously published file is modified, **additively only**:

| Path | Change |
| --- | --- |
| `src/__init__.py` | 0 bytes -> 4 lines, restoring the version declaration (`__version__ = "5.30.1"`) so that the product version of the frozen snapshot is directly inspectable. No `import` statement is added, so package resolution behaviour is unchanged. |

Verification, run between v2 and v3 over tracked files:

```
$ git diff --numstat paper-array-eval-v2 paper-array-eval-v3
4       0       src/__init__.py
```

Four lines added, zero lines deleted, no other tracked file touched. The 84
remaining published files are byte-identical to the frozen snapshot.

## 3. Verifiable code facts (line numbers as of v3, machine-checked)

### 3.1 The four watchdog termination states

All line numbers below were verified by reading the shipped file, not inferred.

| Paper statement | Location in `scripts/watchdog.py` |
| --- | --- |
| **(i)** clean exit (`exit_code == 0`) ⇒ no restart | `:172` (`else:`), `break` at `:178` |
| **(i)** counter reset once `uptime > 60s` | `:175` (`if proc_duration >= UPTIME_THRESHOLD:`), reset at `:176` |
| **(ii)** crash (`exit_code != 0`) ⇒ restart | `:161` (`elif exit_code != 0:`), increment at `:167`, `continue` at `:171` |
| **(ii)** bounded retry, at most 3 | `:37` (`MAX_RESTARTS = 3`), loop bound at `:124` (`while restart_count < MAX_RESTARTS:`), give-up at `:181` |
| **(ii)** counter reset once `uptime > 60s` | `:38` (`UPTIME_THRESHOLD = 60`), applied at `:164`, reset at `:165` |
| **(iii)** controlled restart via `.restart_flag` ⇒ immediate restart, counter **not** incremented | `:39` (flag path), branch at `:155` (`if FLAG_FILE.exists():`), counter reset at `:158`, `continue` at `:160` |
| **(iv)** debugger attached (`pydevd` / `debugpy`) ⇒ direct passthrough | `:108` (`if _is_debug_mode():`), detectors at `:48` and `:51`, delegation at `:111-112` |

### 3.2 Other facts referenced by the paper

| Paper statement | Location |
| --- | --- |
| Audit logger subscribes to EventBus at priority 50 | `src/permissions/audit.py:150-151` |
| `ToolFailureTracker` subscribes at priority 60 | `src/core/experience_recorder.py:91` |
| `SessionManager` | `src/core/session.py:92` |
| RCR prefix-preservation call site | `src/core/session.py:692`; also invoked directly by `benchmarks/exp3_rcr_pipeline.py:394` |
| Product version of the frozen snapshot | `src/__init__.py:3` (`5.30.1`) |

Because smaller priority numbers run first, the audit logger (50) records a
`TOOL_CALL` / `TOOL_RESULT` event before the experience recorder (60) consumes
the same event. This ordering is the one assumed in the paper's description of
the event-driven experience capture path.

## 4. Dependencies

The four added files import **only the Python standard library**. Verified
per-file inventory:

| File | Standard-library imports |
| --- | --- |
| `scripts/watchdog.py` | `subprocess`, `sys`, `time`, `datetime`, `pathlib` |
| `src/permissions/audit.py` | `collections`, `csv`, `dataclasses`, `datetime`, `json`, `logging`, `pathlib`, `sqlite3`, `typing` |
| `src/core/session.py` | `asyncio`, `concurrent.futures`, `dataclasses`, `datetime`, `hashlib`, `json`, `logging`, `typing`, `uuid` |
| `src/permissions/__init__.py` | (empty file) |

Their only intra-project imports are `src.core.token_utils`,
`src.core.event_bus` and `src.core.events`, all three of which were already
published in the frozen snapshot. `requirements.txt` is therefore unchanged,
and `openai`, `python-dotenv` and `numpy` remain sufficient to install and run
every benchmark script in this repository.

Note that `src/core/token_utils.py` optionally uses `litellm.token_counter` for
exact token counts (inside a `try` block, at function scope) and falls back to a
character-ratio estimator when that package is absent. The fallback path is what
the reported numbers use, so `litellm` is deliberately not listed as a
requirement.

## 5. Executability boundary of `scripts/watchdog.py`

`scripts/watchdog.py` is shipped for **inspection and audit**, not for execution
inside this repository. Consistent with the "curated artifact, not the full
product" scope stated in `README.md`, two of its runtime paths reach beyond the
artifact:

- In debug mode it delegates to the product entry point
  (`from src.__main__ import main`, line 111). `src/__main__.py` is not part of
  the artifact.
- Otherwise it spawns `[python, "-m", "src"]` as a child process (line 129),
  which likewise requires the full product package.

Both paths are inside function bodies, so importing the module has no side
effects and succeeds:

```bash
python -c "import scripts.watchdog as w; print(w.MAX_RESTARTS, w.UPTIME_THRESHOLD, w._PROJECT_ROOT)"
# -> 3 60 <repository root>
```

The flat artifact layout is handled by line 34,
`_PROJECT_ROOT = Path(__file__).resolve().parent.parent`, which resolves to the
repository root exactly as it does in the full product tree. The complete
four-state decision logic is contained in `main()` (lines 104-184) and can be
read end to end without the product entry point.

## 6. Provenance of `scripts/watchdog.py`

The supervisor had been in continuous operation before the snapshot was frozen,
but the directory containing it was not yet under version control on the freeze
date. The revision shipped here is the earliest tracked one (2026-06-21). It
contains exactly the four termination states described in the paper, and none
of the later extensions that were added to the product afterwards; those
extensions are outside the scope of the submitted claims and are therefore not
included.

## 7. Reproducing under v3

Setup and the experiment-to-paper mapping are unchanged; see `README.md`. To
inspect the frozen snapshot exactly as submitted:

```bash
git checkout paper-array-eval-v3
python -c "import src; print(src.__version__)"     # -> 5.30.1
python -c "import src.core.session, src.permissions.audit; print('import ok')"
python -c "import scripts.watchdog as w; print(w.MAX_RESTARTS, w.UPTIME_THRESHOLD)"
```

To confirm that v3 does not alter any reported result:

```bash
git diff --numstat paper-array-eval-v2 paper-array-eval-v3   # -> only src/__init__.py, 4 added / 0 deleted
git diff --stat paper-array-eval-v2 paper-array-eval-v3 -- benchmarks/   # -> empty
```
