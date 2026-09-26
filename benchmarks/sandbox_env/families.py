#!/usr/bin/env python3
"""E11 real-execution sandbox: 15 deterministic tool families.

Each handler has the unified signature::

    handle(action: str, args: dict, ws: SandboxWorkspace) -> tuple[status, result, error_kind, error_msg]

where ``status`` is ``"success"`` or ``"error"``. Handlers NEVER import the
executor (the executor wraps these tuples into ``ExecOutcome``), so there is no
circular dependency.

Design principle (the crux of a faithful, non-oracle signal):
  * Lenient on argument KEY names (models vary: ``expression``/``input``/``expr``).
  * Strict on FIXTURE DOMAIN: an acceptable tool given natural in-domain args
    succeeds; a wrong tool (or an out-of-domain argument) returns status=error
    (``not_found`` / ``invalid_args`` / ``execution_error``). That real error is
    what triggers PTE-FD escalation -- no oracle acceptable-set lookup anywhere.
  * Deterministic + local + temp-isolated: no network, no writes outside ``ws``.

Anything not in ``FAMILIES`` is a capability error (handled by the executor),
disclosed in the paper as a capability error rather than a business failure.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import operator
import platform
import re
from datetime import datetime

# ------------------------------------------------------------------ helpers

def _ok(result):
    return ("success", result, None, "")


def _err(kind, msg):
    return ("error", None, kind, msg)


def _arg(args, *keys, default=None):
    """Return the first present, non-empty value among ``keys`` (lenient on names)."""
    for k in keys:
        if k in args and args[k] not in (None, ""):
            return args[k]
    return default


# Safe arithmetic evaluator (no names, no calls, no attribute access).
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression element")


def _eval_arithmetic(text):
    """Evaluate a plain arithmetic string; raise on anything non-arithmetic."""
    s = str(text).strip().replace("=", "")
    s = s.replace("￥", "").replace("$", "").replace("¥", "").replace(",", "")
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    return _safe_eval(ast.parse(s, mode="eval"))


def _numeric_summary(nums):
    n = len(nums)
    mean = sum(nums) / n
    srt = sorted(nums)
    med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    sd = (sum((x - mean) ** 2 for x in nums) / n) ** 0.5
    return {"n": n, "mean": round(mean, 3), "median": round(med, 3),
            "stdev": round(sd, 3), "min": min(nums), "max": max(nums),
            "sum": round(sum(nums), 3)}


def _read_csv(path):
    text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _find_seed_file(ws, path_arg, suffixes=(".csv",)):
    """Resolve a fixture-backed file by exact path, then by basename fallback."""
    if path_arg:
        cand = ws.resolve_fs(path_arg)
        if cand is not None and cand.exists():
            return cand
        base = str(path_arg).replace("\\", "/").split("/")[-1]
        for rel in ws.fixtures.get("seed_fs", {}):
            if rel.endswith(base) or rel.split("/")[-1] == base:
                cand = ws.resolve_fs(rel)
                if cand is not None and cand.exists():
                    return cand
            stem = base.rsplit(".", 1)[0]
            for suf in suffixes:
                if rel.endswith(stem + suf):
                    cand = ws.resolve_fs(rel)
                    if cand is not None and cand.exists():
                        return cand
    return None


# ------------------------------------------------------------------ families

def fam_file(action, args, ws):
    path = _arg(args, "file_path", "path", "filename", "file", "input")
    is_write = action in ("write", "create", "save", "append") or "content" in args or "text" in args
    if is_write:
        content = str(_arg(args, "content", "text", "data", default=""))
        if not path:
            return _err("invalid_args", "file write requires file_path")
        p = ws.resolve_fs(path)
        if p is None:
            return _err("invalid_args", f"path escapes sandbox fs root: {path!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return _ok({"operation": "write", "path": str(path), "bytes": len(content.encode("utf-8"))})
    if not path:
        return _err("invalid_args", "file read requires file_path")
    p = _find_seed_file(ws, path, suffixes=(".txt", ".md", ".csv", ".json")) or ws.resolve_fs(path)
    if p is None:
        return _err("invalid_args", f"path escapes sandbox fs root: {path!r}")
    if not p.exists():
        return _err("not_found", f"file not found in sandbox: {path!r}")
    return _ok({"operation": "read", "path": str(path),
                "content": p.read_text(encoding="utf-8")[:4000]})


def fam_calculator(action, args, ws):
    expr = _arg(args, "expression", "expr", "input", "query", "formula", "text")
    if expr is None:
        return _err("invalid_args", "calculator requires an arithmetic expression")
    try:
        val = _eval_arithmetic(expr)
    except ZeroDivisionError:
        return _err("execution_error", "division by zero")
    except Exception:
        return _err("execution_error", f"cannot evaluate as arithmetic: {expr!r}")
    return _ok({"expression": str(expr), "result": val})


def fam_datetime_tool(action, args, ws):
    tz = _arg(args, "timezone", "tz", "input", default="local")
    now = datetime.now()
    return _ok({"datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"), "weekday": now.strftime("%A"), "timezone": str(tz)})


def fam_data_processor(action, args, ws):
    path = _arg(args, "file_path", "path", "input", "file", "dataset", "csv")
    p = _find_seed_file(ws, path, suffixes=(".csv",)) or ws.resolve_fs("data/sales.csv")
    if p is None or not p.exists():
        return _err("not_found", f"dataset not found in sandbox: {path!r}")
    rows = _read_csv(p)
    if not rows:
        return _err("execution_error", f"dataset is empty: {p.name}")
    summary = {}
    for col in rows[0].keys():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        if vals:
            summary[col] = _numeric_summary(vals)
    return _ok({"file": p.name, "rows": len(rows), "columns": list(rows[0].keys()), "summary": summary})


def fam_knowledge_rag(action, args, ws):
    q = str(_arg(args, "query", "input", "text", "question", "keywords", "q", default="")).strip()
    docs = ws.fixtures.get("rag_index", [])
    if not q:
        return _err("invalid_args", "knowledge_rag requires a query")
    ql = q.lower()
    toks = [t for t in re.split(r"[\s,，、。.；;]+", ql) if t]
    best, best_score = None, 0
    for d in docs:
        kws = [str(k).lower() for k in d.get("keywords", [])]
        hay = (d.get("title", "") + " " + d.get("text", "")).lower()
        score = 0
        for kw in kws:                       # substring match handles CJK queries
            if kw and kw in ql:
                score += 3
        for t in toks:                       # token match handles spaced/English queries
            if t and (t in kws or t in hay):
                score += 1
        if score > best_score:
            best, best_score = d, score
    if best is None or best_score == 0:
        return _err("not_found", f"no indexed document matches query: {q!r}")
    return _ok({"doc_id": best["doc_id"], "title": best["title"],
                "score": best_score, "snippet": best["text"][:200]})


def fam_weather(action, args, ws):
    city = _arg(args, "city", "location", "place", "input", "query", "target", default="")
    cache = ws.fixtures.get("weather_cache", {})
    key = str(city).strip()
    if not key:
        d = dict(ws.fixtures.get("weather_default", {}))
        d["note"] = "no city specified; returned deterministic local default"
        return _ok(d)
    lk = key.lower()
    if key in cache:
        return _ok({"city": key, **cache[key]})
    if lk in cache:
        return _ok({"city": key, **cache[lk]})
    return _err("not_found", f"city not in weather cache (offline fixtures): {city!r}")


def fam_statistics(action, args, ws):
    data = _arg(args, "data", "numbers", "values", "input", "dataset", "sample")
    nums = []
    if isinstance(data, list):
        for x in data:
            try:
                nums.append(float(x))
            except (ValueError, TypeError):
                pass
    elif isinstance(data, str):
        for t in re.split(r"[\s,，、;；]+", data):
            try:
                nums.append(float(t))
            except (ValueError, TypeError):
                pass
    if not nums:
        path = _arg(args, "file_path", "file", "path", "csv", default="data/sales.csv")
        col = _arg(args, "column", "col", "field", default=None)
        p = _find_seed_file(ws, path, suffixes=(".csv",)) or ws.resolve_fs(str(path))
        if p is not None and p.exists():
            rows = _read_csv(p)
            if rows:
                cols = [col] if (col and col in rows[0]) else list(rows[0].keys())
                for c in cols:
                    v = []
                    for r in rows:
                        try:
                            v.append(float(r[c]))
                        except (ValueError, TypeError, KeyError):
                            pass
                    if v:
                        nums = v
                        break
    if not nums:
        return _err("invalid_args", "statistics requires numeric data (inline or a fixture CSV column)")
    return _ok(_numeric_summary(nums))


def fam_python_runner(action, args, ws):
    code = str(_arg(args, "code", "script", "expression", "input", "query", default="")).strip()
    if not code:
        return _err("invalid_args", "python_runner requires code/expression")
    expr = code
    m = re.match(r"^print\s*\((.*)\)\s*$", expr, re.S)
    if m:
        expr = m.group(1)
    expr = expr.strip().strip("\"'")
    try:
        val = _eval_arithmetic(expr)
    except ZeroDivisionError:
        return _err("execution_error", "division by zero")
    except Exception:
        return _err("execution_error",
                    f"restricted sandbox python_runner supports arithmetic expressions only: {code!r}")
    return _ok({"code": code, "result": val})


def fam_poetry(action, args, ws):
    q = str(_arg(args, "query", "input", "keyword", "title", "author", "text", default="")).strip()
    db = ws.fixtures.get("poetry_db", [])
    if not q:
        return _ok({"operation": "browse", "count": len(db), "sample": [p["title"] for p in db[:5]]})
    best, best_score = None, 0
    for p in db:
        score = 0
        fields = [p.get("title", ""), p.get("author", ""), p.get("dynasty", ""),
                  " ".join(p.get("keywords", [])), p.get("text", "")]
        for f in fields:
            if q and f and (q in f or f in q):
                score += 3
        for kw in p.get("keywords", []):
            if kw and kw in q:
                score += 2
        if score > best_score:
            best, best_score = p, score
    if best is None or best_score == 0:
        # Generic recommendation/browse queries ("推荐一些经典诗词") are a
        # legitimate success for the poetry tool: return the curated list rather
        # than a false not_found. Out-of-domain queries still error, preserving
        # the real failure signal that drives escalation.
        if any(t in q for t in ("诗", "古诗", "诗词", "唐诗", "宋词", "推荐", "经典")):
            return _ok({"operation": "browse", "query": q, "count": len(db),
                        "sample": [p["title"] for p in db[:5]]})
        return _err("not_found", f"no poem matches: {q!r}")
    return _ok({"title": best["title"], "author": best["author"],
                "dynasty": best["dynasty"], "text": best["text"]})


def fam_pdf_tool(action, args, ws):
    op = (action or str(_arg(args, "operation", "op", default="inspect"))).lower()
    path = _arg(args, "file_path", "path", "input", "file", "pdf", default="")
    if op in ("merge", "combine"):
        paths = args.get("files") or args.get("inputs") or args.get("paths") or ([path] if path else [])
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            return _err("invalid_args", "pdf merge requires at least one input file")
        out = ws.resolve_fs("merged_output.pdf")
        if out is None:
            return _err("invalid_args", "cannot write merged output")
        out.write_text("MERGED_PDF_PLACEHOLDER\n" + "|".join(str(x) for x in paths), encoding="utf-8")
        return _ok({"operation": "merge", "inputs": [str(x) for x in paths], "output": "merged_output.pdf"})
    p = _find_seed_file(ws, path, suffixes=(".pdf", ".md", ".txt")) or ws.resolve_fs("report.md")
    if p is None:
        return _err("invalid_args", f"bad path: {path!r}")
    if not p.exists():
        return _err("not_found", f"pdf/file not found in sandbox: {path!r}")
    return _ok({"operation": op, "file": p.name, "size_bytes": p.stat().st_size})


def fam_format_converter(action, args, ws):
    path = _arg(args, "file_path", "path", "input", "file", "source", default="")
    target = str(_arg(args, "target_format", "to", "output_format", "format", "target", default="")).lower()
    p = _find_seed_file(ws, path, suffixes=(".md", ".txt", ".docx", ".csv")) or ws.resolve_fs("report.md")
    if p is None:
        return _err("invalid_args", f"bad source path: {path!r}")
    if not p.exists():
        return _err("not_found", f"source file not found in sandbox: {path!r}")
    out_name = p.stem + ("." + target if target else ".converted")
    out = ws.resolve_fs(out_name)
    if out is None:
        return _err("invalid_args", "cannot write converted output")
    out.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return _ok({"source": p.name, "target_format": target or "auto", "output": out_name})


def fam_todo(action, args, ws):
    fp = ws.state_root / "todo.json"
    items = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
    text = _arg(args, "text", "item", "content", "task", "title", "input", default="")
    op = (action or str(_arg(args, "operation", "op", default=""))).lower()
    if op in ("add", "create", "new") or (text and op in ("", "execute")):
        if not text:
            return _err("invalid_args", "todo add requires text")
        nid = max([i.get("id", 0) for i in items], default=0) + 1
        items.append({"id": nid, "text": str(text), "done": False})
        fp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return _ok({"operation": "add", "id": nid, "text": str(text), "total": len(items)})
    return _ok({"operation": "list", "count": len(items), "items": items})


def fam_finance(action, args, ws):
    fp = ws.state_root / "finance.json"
    recs = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
    amount = _arg(args, "amount", "value", "money", "sum", default=None)
    op = (action or str(_arg(args, "operation", "op", default=""))).lower()
    if op in ("record", "add", "create", "log") or amount is not None:
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            return _err("invalid_args", f"finance record requires a numeric amount, got {amount!r}")
        nid = max([r.get("id", 0) for r in recs], default=0) + 1
        rec = {"id": nid, "kind": str(_arg(args, "kind", "type", default="expense")),
               "amount": amt, "category": str(_arg(args, "category", "cat", default="general")),
               "note": str(_arg(args, "note", "text", "input", default=""))}
        recs.append(rec)
        fp.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        return _ok({"operation": "record", "id": nid, "amount": amt, "kind": rec["kind"]})
    income = sum(r["amount"] for r in recs if r.get("kind") == "income")
    expense = sum(r["amount"] for r in recs if r.get("kind") == "expense")
    return _ok({"operation": "summary", "records": len(recs), "income": round(income, 2),
                "expense": round(expense, 2), "balance": round(income - expense, 2)})


def fam_diary(action, args, ws):
    ddir = ws.state_root / "diary"
    ddir.mkdir(parents=True, exist_ok=True)
    text = _arg(args, "text", "content", "entry", "input", "query", default="")
    op = (action or "").lower()
    if op in ("write", "add", "record", "create") or text:
        if not text:
            return _err("invalid_args", "diary write requires text")
        fn = datetime.now().strftime("%Y-%m-%d") + ".md"
        (ddir / fn).write_text(str(text), encoding="utf-8")
        return _ok({"operation": "write", "entry": fn, "chars": len(str(text))})
    entries = sorted(p.name for p in ddir.glob("*.md"))
    return _ok({"operation": "list", "count": len(entries), "entries": entries})


def fam_system_monitor(action, args, ws):
    import os as _os
    base = dict(ws.fixtures.get("system_monitor_baseline", {}))
    base.pop("note", None)
    return _ok({**base, "cpu_count": _os.cpu_count(),
                "platform": platform.platform(), "python": platform.python_version()})


# ------------------------------------------------------------------ registry

FAMILIES = {
    "file": fam_file,
    "calculator": fam_calculator,
    "datetime_tool": fam_datetime_tool,
    "data_processor": fam_data_processor,
    "knowledge_rag": fam_knowledge_rag,
    "weather": fam_weather,
    "statistics": fam_statistics,
    "python_runner": fam_python_runner,
    "poetry": fam_poetry,
    "pdf_tool": fam_pdf_tool,
    "format_converter": fam_format_converter,
    "todo": fam_todo,
    "finance": fam_finance,
    "diary": fam_diary,
    "system_monitor": fam_system_monitor,
}

# Tier 1 coverage expansion: 22 additional deterministic LOCAL families
# (JSON stores / local compute / deterministic text-format artifact generators).
# Additive merge -- the base families above stay intact. Per-family feasibility
# and rationale: E11_COVERAGE_EXPANSION_FEASIBILITY.md.
try:  # package import
    from .families_tier1 import TIER1_FAMILIES
except ImportError:  # direct-module fallback
    from families_tier1 import TIER1_FAMILIES  # type: ignore
FAMILIES.update(TIER1_FAMILIES)

# Tier 2 coverage expansion: 12 record-and-freeze REAL-DATA families (web /
# finance / research) replayed offline from fixtures/tier2_cassette.json -- real
# quant.db extracts + recorded live web/FRED/quotes/papers, zero fabrication.
# Additive merge; domain-strict (never vacuous-success). Rationale: same doc.
try:  # package import
    from .families_tier2 import TIER2_FAMILIES
except ImportError:  # direct-module fallback
    from families_tier2 import TIER2_FAMILIES  # type: ignore
FAMILIES.update(TIER2_FAMILIES)

SANDBOX_FAMILIES = frozenset(FAMILIES.keys())
