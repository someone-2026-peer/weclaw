#!/usr/bin/env python3
"""E11 sandbox Tier 1: 22 additional deterministic LOCAL tool families.

Extends the 15 base families (``families.py``) toward full benchmark coverage so
every query's task-valid success path can actually RUN (see
``E11_COVERAGE_EXPANSION_FEASIBILITY.md``). Tier 1 = pure-local real
implementations / harmless local no-ops: JSON stores, local compute, and
deterministic text-format artifact generation (SVG / HTML / Markdown / minimal
PDF). No network, no hardware, no irreversible side effect, fully reproducible.

Same handler contract as ``families.py``::

    handle(action, args, ws) -> (status, result, error_kind, error_msg)

Same design discipline (memory 8c25ca0b -- never vacuous-success):
  * Lenient on arg KEY names; STRICT on fixture DOMAIN.
  * In-domain args on the right family -> success; malformed / unknown-entity /
    out-of-domain -> status="error" (invalid_args / not_found / execution_error),
    which is the real signal that drives PTE-FD escalation.
  * Generators emit deterministic TEXT artifacts (not byte-identical binary
    Office/matplotlib output); the SCIENCE is the success/error STATUS, which is
    constant for a given (family, action, args).

Dangerous / hardware / network families (shell, email, browser, stock, camera...)
are NOT here -- they are Tier 2 (recorded-frozen fixtures) / Tier 3 (safe mock).
"""
from __future__ import annotations

import json
import re
from datetime import datetime

try:  # package import
    from .families import _ok, _err, _arg, _find_seed_file, _numeric_summary, _read_csv
except ImportError:  # direct-module fallback
    from families import _ok, _err, _arg, _find_seed_file, _numeric_summary, _read_csv  # type: ignore


# ------------------------------------------------------------------ store helpers

def _op(action, args) -> str:
    return (action or str(_arg(args, "operation", "op", "action", default=""))).lower()


def _load_store(ws, name, default=None):
    fp = ws.state_root / f"{name}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return default if default is not None else []


def _save_store(ws, name, data) -> None:
    (ws.state_root / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id(items) -> int:
    return max([i.get("id", 0) for i in items if isinstance(i, dict)], default=0) + 1


def _match(items, subj):
    """Entities whose any string value contains ``subj`` (case-insensitive)."""
    s = str(subj).lower()
    return [it for it in items if isinstance(it, dict)
            and any(s in str(v).lower() for v in it.values())]


def _crud(op, args, ws, store, label, name_keys, add_fields=()):
    """Generic add/query/update/delete over a JSON list store with domain checks.

    Family handlers intercept their special read-ops first, then delegate here.
    STRICT: add needs a subject; query/update/delete by an unknown subject is a
    real ``not_found`` (never a vacuous success)."""
    items = _load_store(ws, store, [])
    subj = _arg(args, *name_keys)
    if op in ("add", "create", "new", "record", "insert", "log") or (subj and op in ("", "execute", "run", "do")):
        if not subj:
            return _err("invalid_args", f"{label} add requires one of {list(name_keys)}")
        ent = {"id": _next_id(items), name_keys[0]: str(subj)}
        for f in add_fields:
            v = _arg(args, f, f + "_value")
            if v is not None:
                ent[f] = v
        items.append(ent)
        _save_store(ws, store, items)
        return _ok({"operation": "add", "store": store, "id": ent["id"],
                    label: str(subj), "total": len(items)})
    if op in ("update", "edit", "modify", "set"):
        found = _match(items, subj) if subj else []
        if not found:
            return _err("not_found", f"no {label} to update matches {subj!r} in sandbox store")
        tgt = found[0]
        for f in add_fields:
            v = _arg(args, f, f + "_value")
            if v is not None:
                tgt[f] = v
        _save_store(ws, store, items)
        return _ok({"operation": "update", "store": store, "id": tgt.get("id"), label: tgt})
    if op in ("delete", "remove", "drop"):
        found = _match(items, subj) if subj else []
        if not found:
            return _err("not_found", f"no {label} to delete matches {subj!r} in sandbox store")
        keep = [it for it in items if it not in found]
        _save_store(ws, store, keep)
        return _ok({"operation": "delete", "store": store, "removed": len(found), "total": len(keep)})
    # default: list / query
    if subj:
        found = _match(items, subj)
        if not found:
            return _err("not_found", f"no {label} matches {subj!r} in sandbox store")
        return _ok({"operation": "query", "store": store, "query": str(subj),
                    "count": len(found), "items": found})
    return _ok({"operation": "list", "store": store, "count": len(items), "items": items})


# ------------------------------------------------------------------ artifact helpers
# Deterministic TEXT artifacts (no heavy deps, no timestamp bytes).

def _slug(text, default="artifact") -> str:
    """Filesystem-safe slug that KEEPS CJK (Chinese titles must not collapse to '_')."""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(text).strip().lower()).strip("_")
    return s[:40] or default


def _write_fs(ws, rel, content) -> str | None:
    p = ws.resolve_fs(rel)
    if p is None:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(str(content), encoding="utf-8")
    return rel


def _parse_numbers(args, ws):
    """Collect (labels, values) from inline data or a fixture CSV column."""
    data = _arg(args, "data", "values", "numbers", "series", "input")
    vals, labels = [], []
    if isinstance(data, list):
        for i, x in enumerate(data):
            if isinstance(x, dict):
                labels.append(str(x.get("label", x.get("name", i))))
                try:
                    vals.append(float(x.get("value", x.get("y", 0))))
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    vals.append(float(x))
                    labels.append(str(i))
                except (ValueError, TypeError):
                    pass
    elif isinstance(data, str):
        for t in re.split(r"[\s,，、;；]+", data):
            try:
                vals.append(float(t))
                labels.append(str(len(vals) - 1))
            except ValueError:
                pass
    if not vals:
        path = _arg(args, "file_path", "file", "csv", "dataset", "source")
        col = _arg(args, "column", "col", "field", "y")
        p = _find_seed_file(ws, path, suffixes=(".csv",)) or ws.resolve_fs("data/sales.csv")
        if p is not None and p.exists():
            rows = _read_csv(p)
            if rows:
                c = col if (col and col in rows[0]) else list(rows[0].keys())[-1]
                lc = list(rows[0].keys())[0]
                for r in rows:
                    try:
                        vals.append(float(r[c]))
                        labels.append(str(r.get(lc, "")))
                    except (ValueError, TypeError, KeyError):
                        pass
    return labels, vals


def _svg_chart(kind, labels, values, title) -> str:
    W, H, pad = 480, 320, 40
    n = max(len(values), 1)
    mx = max(values) if values and max(values) > 0 else 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W//2}" y="24" font-size="16" text-anchor="middle">{title}</text>']
    if kind == "pie":
        import math
        total = sum(values) or 1
        cx, cy, r, ang = W // 2, H // 2 + 10, 100, -math.pi / 2
        for i, v in enumerate(values):
            a2 = ang + 2 * math.pi * v / total
            x1, y1 = cx + r * math.cos(ang), cy + r * math.sin(ang)
            x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
            large = 1 if (a2 - ang) > math.pi else 0
            parts.append(f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r},{r} 0 {large},1 {x2:.1f},{y2:.1f} Z" '
                         f'fill="hsl({i*47%360},60%,60%)" stroke="white"/>')
            ang = a2
    elif kind in ("line", "scatter"):
        pts = []
        for i, v in enumerate(values):
            x = pad + i * (W - 2 * pad) / max(n - 1, 1)
            y = H - pad - (v / mx) * (H - 2 * pad)
            pts.append(f"{x:.1f},{y:.1f}")
        if kind == "line":
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2b6cb0" stroke-width="2"/>')
        else:
            for p in pts:
                x, y = p.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#2b6cb0"/>')
    else:  # bar / heatmap default
        bw = (W - 2 * pad) / n
        for i, v in enumerate(values):
            h = (v / mx) * (H - 2 * pad)
            x = pad + i * bw
            y = H - pad - h
            parts.append(f'<rect x="{x+2:.1f}" y="{y:.1f}" width="{bw-4:.1f}" height="{h:.1f}" fill="#4299e1"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _minimal_pdf(lines) -> bytes:
    """A valid single-page PDF (computed xref), deterministic byte-for-byte."""
    text = "\n".join(lines[:40]) or "WeClaw E11 sandbox document"
    esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = "BT /F1 11 Tf 56 736 Td 15 TL\n" + "".join(f"({ln}) Tj T*\n" for ln in esc.split("\n")) + "ET"
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>",
        f"<</Length {len(stream.encode('latin-1','replace'))}>>\nstream\n{stream}\nendstream".encode("latin-1", "replace"),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<</Size {len(objs)+1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def _html_deck(title, slides) -> str:
    body = "\n".join(f'<section class="slide"><h2>{s}</h2></section>' for s in slides) or \
        '<section class="slide"><h2>Untitled deck</h2></section>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>.slide{{page-break-after:always;padding:48px;font-family:sans-serif}}</style>'
            f'</head><body><h1>{title}</h1>\n{body}\n</body></html>')


def _split_bullets(text):
    parts = [p.strip(" -*\t") for p in re.split(r"[\n;；]+", str(text)) if p.strip(" -*\t")]
    return parts or [str(text)]


# ------------------------------------------------------------------ STORE families

def fam_cron(action, args, ws):
    op = _op(action, args)
    if op in ("list_jobs", "query_schedules", "list", "query", "get"):
        return _crud("list", args, ws, "cron", "job", ("name", "job", "task", "title"))
    if op in ("remove_job", "delete_schedule", "delete", "remove"):
        return _crud("delete", args, ws, "cron", "job", ("name", "job", "task", "id"))
    if op in ("pause_job", "resume_job", "pause", "resume", "update_schedule", "complete_schedule"):
        return _crud("update", args, ws, "cron", "job", ("name", "job", "task", "id"), ("status",))
    return _crud("add", args, ws, "cron", "job",
                 ("name", "job", "task", "title", "command", "prompt"),
                 ("schedule", "cron", "interval", "time", "kind", "payload"))


_HEALTH_METRICS = frozenset({"weight", "blood_pressure", "heart_rate", "blood_sugar",
                             "sleep", "steps", "temperature", "体重", "血压", "心率", "血糖", "睡眠", "步数"})


def fam_health(action, args, ws):
    op = _op(action, args)
    if op in ("get_health_trends", "trends", "trend", "analyze"):
        items = _load_store(ws, "health", [])
        metric = _arg(args, "metric", "type", "kind", "name")
        if metric and str(metric).lower() not in _HEALTH_METRICS and not _match(items, metric):
            return _err("not_found", f"unknown health metric: {metric!r}")
        by = {}
        for it in items:
            by.setdefault(str(it.get("metric", "other")), []).append(it.get("value"))
        trends = {k: {"n": len(v), "first": v[0], "last": v[-1]} for k, v in by.items() if v}
        return _ok({"operation": "trends", "records": len(items), "trends": trends})
    if op in ("record_health_data", "update_health_data", "record", "add", "create", "log"):
        metric = _arg(args, "metric", "type", "kind", "name")
        value = _arg(args, "value", "reading", "amount", "data")
        if not metric:
            return _err("invalid_args", "health record requires a metric (weight/blood_pressure/...)")
        if str(metric).lower() not in _HEALTH_METRICS:
            return _err("not_found", f"health metric outside tracked domain: {metric!r}")
        try:
            vnum = float(value)
        except (TypeError, ValueError):
            vnum = None
        items = _load_store(ws, "health", [])
        ent = {"id": _next_id(items), "metric": str(metric), "value": vnum if vnum is not None else str(value or ""),
               "unit": str(_arg(args, "unit", default="")), "date": datetime.now().strftime("%Y-%m-%d")}
        items.append(ent)
        _save_store(ws, "health", items)
        return _ok({"operation": "record", "store": "health", "id": ent["id"], "metric": ent["metric"], "value": ent["value"]})
    return _crud(op or "query", args, ws, "health", "record", ("metric", "type", "name", "query"))


def fam_medication(action, args, ws):
    op = _op(action, args)
    if op in ("mark_medication_taken", "mark_taken", "take", "taken"):
        items = _load_store(ws, "medication", [])
        subj = _arg(args, "name", "medication", "drug", "med")
        found = _match(items, subj) if subj else []
        if not found:
            return _err("not_found", f"no medication matches {subj!r} in sandbox store")
        found[0].setdefault("taken", []).append(datetime.now().strftime("%Y-%m-%d %H:%M"))
        _save_store(ws, "medication", items)
        return _ok({"operation": "mark_taken", "medication": found[0].get("name"), "times": len(found[0]["taken"])})
    return _crud(op, args, ws, "medication", "medication",
                 ("name", "medication", "drug", "med", "title"), ("dosage", "frequency", "freq", "dose"))


def fam_course_schedule(action, args, ws):
    op = _op(action, args)
    if op in ("list_schedules", "list", "get", "query", "search_courses", "search"):
        return _crud("query" if _arg(args, "member", "course", "name", "query") else "list",
                     args, ws, "course_schedule", "course", ("member", "course", "name", "query"))
    return _crud(op or "add", args, ws, "course_schedule", "course",
                 ("course", "name", "subject", "member"), ("member", "weekday", "time", "day"))


def fam_daily_task(action, args, ws):
    op = _op(action, args)
    if op in ("get_today_summary", "summary", "today"):
        items = _load_store(ws, "daily_task", [])
        done = sum(1 for i in items if str(i.get("status", "")).lower() in ("done", "complete", "completed"))
        return _ok({"operation": "summary", "total": len(items), "done": done, "pending": len(items) - done})
    if op in ("generate_recommendations", "accept_recommendations", "recommend"):
        items = _load_store(ws, "daily_task", [])
        return _ok({"operation": "recommendations", "based_on": len(items),
                    "suggestions": ["review pending tasks", "prioritize high-effort items"]})
    if op in ("complete_task", "start_task", "cancel_task", "complete", "start", "cancel", "update_daily_task", "update"):
        items = _load_store(ws, "daily_task", [])
        subj = _arg(args, "text", "task", "name", "title", "id")
        found = _match(items, subj) if subj else []
        if not found:
            return _err("not_found", f"no daily task matches {subj!r}")
        found[0]["status"] = {"complete_task": "done", "complete": "done", "start_task": "active",
                              "start": "active", "cancel_task": "cancelled", "cancel": "cancelled"}.get(op, "updated")
        _save_store(ws, "daily_task", items)
        return _ok({"operation": "update", "task": found[0]})
    return _crud(op or "add", args, ws, "daily_task", "task",
                 ("text", "task", "name", "title", "content"), ("status", "priority", "due"))


def fam_meal_menu(action, args, ws):
    op = _op(action, args)
    if op in ("parse_image",):
        img = _arg(args, "image", "file_path", "path", "image_path")
        if not img:
            return _err("invalid_args", "meal_menu parse_image requires an image path")
        return _ok({"operation": "parse_image", "image": str(img),
                    "dishes": ["西红柿炒蛋", "米饭"], "note": "deterministic sandbox parse (no live vision model)"})
    if op in ("add_dish", "edit_dish", "delete_dish"):
        return _crud({"add_dish": "add", "edit_dish": "update", "delete_dish": "delete"}[op],
                     args, ws, "meal_menu", "dish", ("dish", "name", "food", "item"), ("menu", "type", "nutrition"))
    return _crud(op, args, ws, "meal_menu", "menu", ("name", "menu", "title", "type"), ("type", "dishes", "day"))


def fam_family_member(action, args, ws):
    op = _op(action, args)
    if op in ("get_family_tree", "tree", "family_tree"):
        items = _load_store(ws, "family_member", [])
        return _ok({"operation": "family_tree", "members": len(items),
                    "tree": [{"name": i.get("name"), "relation": i.get("relation", "member")} for i in items]})
    return _crud(op, args, ws, "family_member", "member",
                 ("name", "member", "person", "title"), ("relation", "birthday", "age", "gender"))


def fam_family_album(action, args, ws):
    op = _op(action, args)
    if op in ("analyze_photo",):
        photo = _arg(args, "photo", "image", "file_path", "path")
        if not photo:
            return _err("invalid_args", "family_album analyze_photo requires a photo path")
        return _ok({"operation": "analyze_photo", "photo": str(photo),
                    "tags": ["family", "outdoor"], "note": "deterministic sandbox analysis (no live vision model)"})
    if op in ("add_photo", "import_photos"):
        items = _load_store(ws, "family_album", [])
        album = _arg(args, "album", "name", "album_name")
        found = _match(items, album) if album else items[:1]
        if not found:
            return _err("not_found", f"no album matches {album!r} in sandbox store")
        found[0].setdefault("photos", []).append(str(_arg(args, "photo", "path", "file", default="photo.jpg")))
        _save_store(ws, "family_album", items)
        return _ok({"operation": "add_photo", "album": found[0].get("name"), "photos": len(found[0]["photos"])})
    if op in ("get_gallery", "gallery", "search_photos"):
        return _crud("query" if _arg(args, "album", "name", "query", "tag") else "list",
                     args, ws, "family_album", "album", ("album", "name", "query", "tag"))
    return _crud(op, args, ws, "family_album", "album", ("name", "album", "title"), ("date", "cover", "photos"))


def fam_family_milestone(action, args, ws):
    op = _op(action, args)
    if op in ("get_timeline", "export_timeline", "timeline"):
        items = _load_store(ws, "family_milestone", [])
        return _ok({"operation": "timeline", "count": len(items),
                    "timeline": sorted(items, key=lambda i: str(i.get("date", "")))})
    if op in ("get_upcoming", "upcoming"):
        items = _load_store(ws, "family_milestone", [])
        return _ok({"operation": "upcoming", "count": len(items), "items": items})
    if op in ("get_statistics", "statistics", "stats"):
        items = _load_store(ws, "family_milestone", [])
        by = {}
        for i in items:
            by[str(i.get("type", "other"))] = by.get(str(i.get("type", "other")), 0) + 1
        return _ok({"operation": "statistics", "total": len(items), "by_type": by})
    return _crud(op, args, ws, "family_milestone", "milestone",
                 ("title", "name", "milestone", "event"), ("date", "type", "description"))


def fam_user_profile(action, args, ws):
    op = _op(action, args)
    st = _load_store(ws, "user_profile", {"profile": {}, "members": [], "contacts": []})
    if not isinstance(st, dict):
        st = {"profile": {}, "members": [], "contacts": []}
    if op in ("query_profile", "get_profile", "query", "get", "profile"):
        return _ok({"operation": "query_profile", "profile": st.get("profile", {}),
                    "members": len(st.get("members", [])), "contacts": len(st.get("contacts", []))})
    if op in ("update_profile", "update", "set"):
        for k in ("name", "age", "gender", "email", "phone", "birthday", "occupation"):
            v = _arg(args, k)
            if v is not None:
                st.setdefault("profile", {})[k] = v
        _save_store(ws, "user_profile", st)
        return _ok({"operation": "update_profile", "profile": st["profile"]})
    if op in ("get_upcoming_birthdays", "birthdays"):
        bdays = [m for m in st.get("members", []) if m.get("birthday")]
        return _ok({"operation": "birthdays", "count": len(bdays), "items": bdays})
    if op in ("add_family_member", "add_social_contact", "add_member", "add_contact", "add"):
        name = _arg(args, "name", "member", "contact", "person")
        if not name:
            return _err("invalid_args", "user_profile add requires a name")
        bucket = "contacts" if "contact" in op else "members"
        ent = {"id": _next_id(st.get(bucket, [])), "name": str(name),
               "relation": str(_arg(args, "relation", "birthday", default=""))}
        st.setdefault(bucket, []).append(ent)
        _save_store(ws, "user_profile", st)
        return _ok({"operation": "add", "bucket": bucket, "id": ent["id"], "name": str(name)})
    if op in ("record_child_growth", "growth"):
        return _ok({"operation": "record_child_growth", "recorded": True,
                    "child": str(_arg(args, "name", "child", default="")), "note": "stored locally"})
    return _ok({"operation": op or "query_profile", "profile": st.get("profile", {})})


def fam_music_player(action, args, ws):
    op = _op(action, args)
    st = _load_store(ws, "music_player", {"songs": [], "playlists": [], "now_playing": None})
    if not isinstance(st, dict):
        st = {"songs": [], "playlists": [], "now_playing": None}
    songs = st.get("songs", [])
    if op in ("play_song", "play", "resume_song", "resume"):
        subj = _arg(args, "song", "title", "name", "query")
        found = _match(songs, subj) if subj else songs[:1]
        if not found:
            return _err("not_found", f"no song matches {subj!r} in local library")
        st["now_playing"] = found[0].get("title", found[0].get("name"))
        _save_store(ws, "music_player", st)
        return _ok({"operation": "play", "now_playing": st["now_playing"],
                    "note": "sandbox no-op playback (no audio device)"})
    if op in ("pause_song", "pause", "stop_song", "stop", "next_song", "prev_song", "next", "prev",
              "seek_to", "set_volume", "set_loop_mode", "shuffle_play", "set_player_settings", "get_player_settings"):
        return _ok({"operation": op, "now_playing": st.get("now_playing"),
                    "note": "sandbox no-op transport control (no audio device)"})
    if op in ("get_now_playing", "now_playing"):
        return _ok({"operation": "now_playing", "title": st.get("now_playing"),
                    "note": "" if st.get("now_playing") else "nothing playing"})
    if op in ("search_songs", "search", "get_playlist", "playlist", "list"):
        subj = _arg(args, "song", "title", "query", "keyword", "name", "playlist")
        if subj:
            found = _match(songs, subj)
            if not found:
                return _err("not_found", f"no song matches {subj!r} in local library")
            return _ok({"operation": "search", "count": len(found), "songs": found})
        return _ok({"operation": "list", "count": len(songs), "songs": songs, "playlists": st.get("playlists", [])})
    if op in ("add_songs", "add_song", "download_song", "add", "scan_local_music", "scan"):
        title = _arg(args, "title", "song", "name")
        if not title and op not in ("scan_local_music", "scan"):
            return _err("invalid_args", "music_player add requires a song title")
        if op in ("scan_local_music", "scan"):
            return _ok({"operation": "scan", "scanned": len(songs), "library": "sandbox-local"})
        ent = {"id": _next_id(songs), "title": str(title), "artist": str(_arg(args, "artist", default="unknown"))}
        songs.append(ent)
        st["songs"] = songs
        _save_store(ws, "music_player", st)
        return _ok({"operation": "add_song", "id": ent["id"], "title": str(title), "total": len(songs)})
    if op in ("create_playlist", "create_tag"):
        name = _arg(args, "name", "playlist", "tag", "title")
        if not name:
            return _err("invalid_args", "create_playlist requires a name")
        st.setdefault("playlists", []).append({"id": _next_id(st.get("playlists", [])), "name": str(name), "songs": []})
        _save_store(ws, "music_player", st)
        return _ok({"operation": "create_playlist", "name": str(name)})
    if op in ("delete_song", "remove_song", "delete", "remove", "remove_from_playlist"):
        subj = _arg(args, "song", "title", "name")
        found = _match(songs, subj) if subj else []
        if not found:
            return _err("not_found", f"no song to delete matches {subj!r}")
        st["songs"] = [s for s in songs if s not in found]
        _save_store(ws, "music_player", st)
        return _ok({"operation": "delete_song", "removed": len(found)})
    return _ok({"operation": op or "list", "count": len(songs), "songs": songs})


def fam_fitness_nutrition(action, args, ws):
    op = _op(action, args)
    if op in ("calculate", "compute", "bmi", "tdee", "calorie", "analyze", "assess") or \
            _arg(args, "height", "weight") is not None:
        h = _arg(args, "height", "height_cm", "tall")
        w = _arg(args, "weight", "weight_kg", "mass")
        try:
            hcm = float(h) if h is not None else None
            wkg = float(w) if w is not None else None
        except (ValueError, TypeError):
            return _err("invalid_args", f"fitness requires numeric height/weight, got {h!r}/{w!r}")
        if hcm is None or wkg is None:
            return _err("invalid_args", "fitness_nutrition compute requires height and weight")
        if not (50 <= hcm <= 250) or not (10 <= wkg <= 400):
            return _err("invalid_args", f"height/weight outside plausible range: {hcm}/{wkg}")
        bmi = round(wkg / ((hcm / 100) ** 2), 2)
        cat = "underweight" if bmi < 18.5 else "normal" if bmi < 24 else "overweight" if bmi < 28 else "obese"
        return _ok({"operation": "bmi", "height_cm": hcm, "weight_kg": wkg, "bmi": bmi, "category": cat})
    return _crud(op, args, ws, "fitness_nutrition", "log",
                 ("food", "meal", "name", "item", "activity"), ("calories", "protein", "carbs", "fat", "date"))


# ------------------------------------------------------------------ GENERATOR families

def fam_data_visualization(action, args, ws):
    kind = {"plot_bar": "bar", "plot_line": "line", "plot_pie": "pie", "plot_scatter": "scatter",
            "plot_heatmap": "heatmap", "generate_dashboard": "bar"}.get(
        (action or "").lower(), str(_arg(args, "chart_type", "type", "kind", default="bar")).lower())
    kind = kind.replace("plot_", "")
    labels, values = _parse_numbers(args, ws)
    if not values:
        return _err("invalid_args", "data_visualization requires data (inline numbers or a fixture CSV)")
    title = str(_arg(args, "title", "name", default=f"{kind} chart"))
    svg = _svg_chart(kind, labels, values, title)
    out = _write_fs(ws, f"{_slug(title, 'chart')}.svg", svg)
    if out is None:
        return _err("execution_error", "cannot write chart into sandbox fs")
    return _ok({"operation": "plot", "chart_type": kind, "points": len(values), "output": out,
                "summary": _numeric_summary(values)})


def fam_ppt_generator(action, args, ws):
    topic = _arg(args, "topic", "title", "subject", "theme", "name")
    content = _arg(args, "content", "slides", "outline", "text", "markdown")
    if not topic and not content:
        return _err("invalid_args", "ppt_generator requires a topic or slide content")
    title = str(topic or "Presentation")
    slides = _split_bullets(content) if content else [f"Overview of {title}", "Key points", "Summary"]
    html = _html_deck(title, slides)
    out = _write_fs(ws, f"{_slug(title, 'deck')}.html", html)
    if out is None:
        return _err("execution_error", "cannot write deck into sandbox fs")
    return _ok({"operation": "generate_ppt", "title": title, "slides": len(slides),
                "format": "html", "output": out})


def fam_pdf_generator(action, args, ws):
    src = _arg(args, "source", "content", "text", "markdown", "html", "md", "file_path", "path")
    title = str(_arg(args, "title", "output", "name", default="document"))
    looks_like_path = bool(re.search(r"\.(md|txt|html?|pdf|docx?|csv)$", str(src or ""), re.I)) \
        or ("/" in str(src or "")) or ("\\" in str(src or ""))
    lines = None
    if src and looks_like_path:
        p = _find_seed_file(ws, src, suffixes=(".md", ".txt", ".html"))
        if p is None or not p.exists():
            return _err("not_found", f"pdf source file not found in sandbox: {src!r}")
        lines = p.read_text(encoding="utf-8").splitlines()
    elif src:
        lines = _split_bullets(src)
    else:
        p = ws.resolve_fs("report.md")
        if p is not None and p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        return _err("not_found", f"pdf_generator has no source content/file: {src!r}")
    pdf = _minimal_pdf(lines)
    out = _write_fs(ws, f"{_slug(title, 'doc')}.pdf", pdf)
    if out is None:
        return _err("execution_error", "cannot write pdf into sandbox fs")
    return _ok({"operation": "generate_pdf", "title": title, "pages": 1, "bytes": len(pdf), "output": out})


def fam_doc_generator(action, args, ws):
    op = (action or "").lower()
    if op in ("list_templates",):
        return _ok({"operation": "list_templates",
                    "templates": ["report", "resume", "letter", "proposal", "meeting_minutes"]})
    if op in ("get_brand_profile", "get_user_preferences"):
        return _ok({"operation": op, "profile": _load_store(ws, "doc_brand", {"tone": "professional", "lang": "zh"})})
    if op in ("validate_document", "check_document_quality", "run_advanced_quality_check"):
        doc = _arg(args, "document", "content", "file_path", "path", "text")
        if not doc:
            return _err("invalid_args", "document validation requires a document/content")
        return _ok({"operation": "quality_check", "passed": True, "score": 0.9, "issues": []})
    content = _arg(args, "content", "text", "markdown", "body", "topic", "title")
    if not content:
        return _err("invalid_args", "doc_generator requires content/topic")
    fmt = str(_arg(args, "format", "output_format", "type", default="md")).lower().lstrip(".")
    title = str(_arg(args, "title", "name", default="document"))
    body = f"# {title}\n\n" + "\n".join(f"- {b}" for b in _split_bullets(content))
    out = _write_fs(ws, f"{_slug(title, 'doc')}.{fmt if fmt in ('md','html','txt') else 'md'}",
                    body if fmt != "html" else f"<html><body><h1>{title}</h1>{''.join(f'<p>{b}</p>' for b in _split_bullets(content))}</body></html>")
    if out is None:
        return _err("execution_error", "cannot write document into sandbox fs")
    return _ok({"operation": "generate_document", "title": title, "format": fmt, "output": out})


def fam_concept_diagrams(action, args, ws):
    nodes = _arg(args, "nodes", "concepts", "items", "text", "content", "topic")
    if not nodes:
        return _err("invalid_args", "concept_diagrams requires nodes/concepts/text")
    labels = nodes if isinstance(nodes, list) else _split_bullets(nodes)
    title = str(_arg(args, "title", "name", default="concept diagram"))
    W, H = 480, 60 + 40 * len(labels)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W//2}" y="24" font-size="15" text-anchor="middle">{title}</text>']
    for i, lb in enumerate(labels[:12]):
        y = 50 + i * 40
        parts.append(f'<rect x="60" y="{y}" width="360" height="30" rx="6" fill="#ebf8ff" stroke="#4299e1"/>')
        parts.append(f'<text x="240" y="{y+20}" font-size="13" text-anchor="middle">{lb}</text>')
    parts.append("</svg>")
    out = _write_fs(ws, f"{_slug(title, 'diagram')}.svg", "\n".join(parts))
    if out is None:
        return _err("execution_error", "cannot write diagram into sandbox fs")
    return _ok({"operation": "concept_diagram", "nodes": len(labels), "output": out})


def fam_meme_generation(action, args, ws):
    text = _arg(args, "text", "caption", "top_text", "content", "title")
    template = _arg(args, "template", "style", "image")
    if not text:
        return _err("invalid_args", "meme_generation requires caption text")
    title = str(_arg(args, "title", default="meme"))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">'
           f'<rect width="400" height="300" fill="#f7fafc" stroke="#a0aec0"/>'
           f'<text x="200" y="40" font-size="20" text-anchor="middle" font-family="Impact">{text}</text>'
           f'<text x="200" y="170" font-size="13" text-anchor="middle">[{template or "classic"} template]</text>'
           f'</svg>')
    out = _write_fs(ws, f"{_slug(title, 'meme')}.svg", svg)
    if out is None:
        return _err("execution_error", "cannot write meme into sandbox fs")
    return _ok({"operation": "meme", "template": str(template or "classic"), "output": out})


def fam_mind_map(action, args, ws):
    text = _arg(args, "text", "content", "topic", "title", "markdown")
    if not text:
        return _err("invalid_args", "mind_map requires text/topic")
    topic = str(_arg(args, "topic", "title", default="MindMap"))
    branches = text if isinstance(text, list) else _split_bullets(text)
    fmt = str(_arg(args, "format", "output_format", default="md")).lower().lstrip(".")
    if fmt == "opml":
        body = ('<?xml version="1.0"?><opml version="2.0"><head>'
                f'<title>{topic}</title></head><body><outline text="{topic}">'
                + "".join(f'<outline text="{b}"/>' for b in branches[:20])
                + "</outline></body></opml>")
        ext = "opml"
    else:
        body = f"# {topic}\n" + "".join(f"- {b}\n" for b in branches[:20])
        ext = "md"
    out = _write_fs(ws, f"{_slug(topic, 'mindmap')}.{ext}", body)
    if out is None:
        return _err("execution_error", "cannot write mindmap into sandbox fs")
    return _ok({"operation": "mind_map", "topic": topic, "branches": len(branches), "format": ext, "output": out})


def fam_ai_writer(action, args, ws):
    topic = _arg(args, "topic", "title", "subject", "prompt", "theme")
    if not topic:
        return _err("invalid_args", "ai_writer requires a topic/title")
    kind = (action or str(_arg(args, "type", default="article"))).lower().replace("write_", "")
    topic = str(topic)
    if kind == "paper":
        body = (f"# {topic}\n\n## Abstract\nDeterministic outline (sandbox, no LLM call).\n\n"
                "## 1. Introduction\n## 2. Related Work\n## 3. Method\n## 4. Experiments\n## 5. Conclusion\n")
    elif kind == "novel":
        body = f"# {topic}\n\n## Chapter 1 — Setup\n## Chapter 2 — Conflict\n## Chapter 3 — Resolution\n"
    else:
        body = f"# {topic}\n\n## Opening\n## Key Point 1\n## Key Point 2\n## Closing\n"
    out = _write_fs(ws, f"{_slug(topic, 'draft')}.md", body)
    if out is None:
        return _err("execution_error", "cannot write draft into sandbox fs")
    return _ok({"operation": f"write_{kind}", "topic": topic, "words": len(body.split()),
                "note": "deterministic outline (sandbox, no LLM call)", "output": out})


# ------------------------------------------------------------------ NO-OP families

def fam_notify(action, args, ws):
    msg = _arg(args, "message", "text", "title", "content", "body", "notification")
    if not msg:
        return _err("invalid_args", "notify requires a message/title")
    log = _load_store(ws, "notify_log", [])
    log.append({"id": _next_id(log), "message": str(msg), "at": datetime.now().strftime("%H:%M:%S")})
    _save_store(ws, "notify_log", log)
    return _ok({"operation": "send", "delivered": True, "message": str(msg),
                "note": "sandbox no-op notification (no real toast, no durable side effect)"})


def fam_clipboard(action, args, ws):
    op = _op(action, args)
    st = _load_store(ws, "clipboard", {"text": "", "image": None})
    if not isinstance(st, dict):
        st = {"text": "", "image": None}
    if op in ("write", "set", "copy"):
        content = _arg(args, "text", "content", "data", "value")
        if content is None:
            return _err("invalid_args", "clipboard write requires text/content")
        st["text"] = str(content)
        _save_store(ws, "clipboard", st)
        return _ok({"operation": "write", "chars": len(st["text"]), "note": "sandbox-local clipboard"})
    if op in ("clear", "empty"):
        st = {"text": "", "image": None}
        _save_store(ws, "clipboard", st)
        return _ok({"operation": "clear"})
    if op in ("read_image",):
        if not st.get("image"):
            return _err("not_found", "no image on the sandbox clipboard")
        return _ok({"operation": "read_image", "image": st["image"]})
    # read
    if not st.get("text"):
        return _err("not_found", "sandbox clipboard is empty")
    return _ok({"operation": "read", "text": st["text"], "chars": len(st["text"])})


# ------------------------------------------------------------------ registry

TIER1_FAMILIES = {
    "cron": fam_cron,
    "music_player": fam_music_player,
    "data_visualization": fam_data_visualization,
    "ppt_generator": fam_ppt_generator,
    "fitness_nutrition": fam_fitness_nutrition,
    "health": fam_health,
    "course_schedule": fam_course_schedule,
    "daily_task": fam_daily_task,
    "medication": fam_medication,
    "pdf_generator": fam_pdf_generator,
    "family_album": fam_family_album,
    "family_member": fam_family_member,
    "family_milestone": fam_family_milestone,
    "meal_menu": fam_meal_menu,
    "user_profile": fam_user_profile,
    "doc_generator": fam_doc_generator,
    "notify": fam_notify,
    "concept_diagrams": fam_concept_diagrams,
    "clipboard": fam_clipboard,
    "meme_generation": fam_meme_generation,
    "mind_map": fam_mind_map,
    "ai_writer": fam_ai_writer,
}
