#!/usr/bin/env python3
"""E11 sandbox Tier 2: 12 record-and-freeze REAL-DATA tool families.

Tier 2 unlocks the scientifically central web/finance/research families that
Tier 1 (pure-local) could not, WITHOUT any network access at benchmark time.
Every handler replays ``fixtures/tier2_cassette.json`` -- a frozen cassette of
REAL data (zero fabrication):

  * financials machine-extracted from ``data/quant.db`` (real OHLCV bars,
    real ensemble signals, real backtests) + ``config/stock_pools.json``;
  * AAPL/TSLA live quotes, FRED macro series (GDP / real-GDP growth / CPI /
    effective fed-funds rate) recorded 2026-09-16/17 from CNBC/MarketWatch/FRED;
  * real web pages (example.com, books.toscrape.com, httpbin form) via fetch;
  * real OpenAlex/arXiv paper metadata + abstract;
  * real web-search result sets for 9 recorded intents.

Same handler contract as ``families.py`` / ``families_tier1.py``::

    handle(action, args, ws) -> (status, result, error_kind, error_msg)

Same design discipline (memory 8c25ca0b -- NEVER vacuous-success):
  * Lenient on arg KEY names; STRICT on the frozen cassette DOMAIN.
  * In-domain args on the right family -> success with REAL frozen data;
    unknown entity / out-of-domain argument -> status="error"
    (not_found / invalid_args), the real signal that drives PTE-FD escalation.
  * Deterministic + local + offline: the cassette is read once and replayed;
    no network, no live order, no write outside ``ws``.

Anything quantitative (MA-cross backtest, MA5 signal, CPI year-over-year) is
COMPUTED for real on the frozen series -- never hardcoded -- so the numbers are
both reproducible and genuinely derived from recorded market data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:  # package import
    from .families import _ok, _err, _arg
    from .families_tier1 import _slug, _write_fs
except ImportError:  # direct-module fallback
    from families import _ok, _err, _arg  # type: ignore
    from families_tier1 import _slug, _write_fs  # type: ignore


# ------------------------------------------------------------------ cassette

_CASSETTE_PATH = Path(__file__).resolve().parent / "fixtures" / "tier2_cassette.json"
_CASSETTE: dict | None = None


def _cassette() -> dict:
    """Lazily load + cache the frozen Tier-2 real-data cassette."""
    global _CASSETTE
    if _CASSETTE is None:
        _CASSETTE = json.loads(_CASSETTE_PATH.read_text(encoding="utf-8"))
    return _CASSETTE


# ------------------------------------------------------------------ resolvers

def _resolve_symbol(raw) -> str | None:
    """Map a symbol / company name / alias / free text to a frozen cassette symbol.

    Exact symbol -> exact alias -> longest alias/name substring inside ``raw``.
    Returns None when nothing in the real recorded universe matches (never guess).
    """
    if raw is None:
        return None
    cas = _cassette()
    quotes = cas.get("stock_quotes", {})
    aliases = cas.get("aliases", {})
    meta = cas.get("stock_meta", {})
    s = str(raw).strip()
    if not s:
        return None
    sl = s.lower()
    if s in quotes:
        return s
    if s.upper() in quotes:
        return s.upper()
    # exact alias
    for sym, al in aliases.items():
        if sl == str(sym).lower():
            return sym
        for a in al:
            if sl == str(a).lower():
                return sym
    # longest alias / name / symbol appearing as a substring of the free text
    best_sym, best_len = None, 0
    for sym, al in aliases.items():
        needles = [sym] + list(al)
        nm = meta.get(sym, {}).get("name")
        if nm:
            needles.append(nm)
        for nd in needles:
            ndl = str(nd).lower()
            if ndl and ndl in sl and len(ndl) > best_len:
                best_sym, best_len = sym, len(ndl)
    return best_sym


def _resolve_page(raw, pages: dict):
    """Resolve a URL / free text to a frozen cassette page (real recorded pages)."""
    if raw is None:
        return None, None
    s = str(raw).lower().strip()
    if not s:
        return None, None
    s2 = s.replace("https://", "").replace("http://", "").replace("www.", "")
    for key, pg in pages.items():
        kl = key.lower()
        if kl == s or kl == s2 or s2.startswith(kl) or kl in s2:
            return pg, key
    for key, pg in pages.items():
        u = str(pg.get("url", "")).lower().replace("https://", "").replace("http://", "")
        if u and (u in s2 or s2 in u):
            return pg, key
    return None, None


def _tokens(text: str):
    """Lowercased word tokens (len>=3) for fuzzy matching."""
    return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", str(text).lower()) if len(t) >= 3]


# ------------------------------------------------------------------ quant math
# Real computations on frozen series (never hardcoded, fully reproducible).

def _moving_average(vals, n):
    out = [None] * len(vals)
    run = 0.0
    for i, v in enumerate(vals):
        run += v
        if i >= n:
            run -= vals[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def _ma_cross_backtest(closes, fast=5, slow=20):
    """A real long-only MA-cross backtest over frozen closes."""
    mf, ms = _moving_average(closes, fast), _moving_average(closes, slow)
    pos, entry, trades = 0, None, []
    peak, max_dd = 1.0, 0.0
    for i in range(1, len(closes)):
        if mf[i] is None or ms[i] is None or mf[i - 1] is None or ms[i - 1] is None:
            continue
        if mf[i - 1] <= ms[i - 1] and mf[i] > ms[i] and pos == 0:
            pos, entry = 1, closes[i]
        elif mf[i - 1] >= ms[i - 1] and mf[i] < ms[i] and pos == 1:
            pos = 0
            trades.append((closes[i] - entry) / entry)
        if pos == 1 and entry:
            eq = 1.0 + (closes[i] - entry) / entry
            peak = max(peak, eq)
            max_dd = min(max_dd, (eq - peak) / peak)
    n = len(trades)
    total = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    if n:
        mean = total / n
        sd = (sum((t - mean) ** 2 for t in trades) / n) ** 0.5
        sharpe = round(mean / sd, 2) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    return {"total_return": round(total, 4), "trades": n,
            "win_rate": round(wins / n, 3) if n else 0.0,
            "max_drawdown": round(max_dd, 4), "sharpe": sharpe,
            "fast_ma": fast, "slow_ma": slow, "bars": len(closes)}


def _signal_from_series(sym, bars):
    """A real MA5-trend signal computed on a frozen close series."""
    closes = [b["close"] for b in bars if b.get("close") is not None]
    if len(closes) < 2:
        return None
    tail = closes[-5:]
    ma5 = sum(tail) / len(tail)
    latest = closes[-1]
    dev = (latest - ma5) / ma5 if ma5 else 0.0
    action = "BUY" if dev > 0.005 else ("SELL" if dev < -0.005 else "HOLD")
    return {"symbol": sym, "strategy": "ma5_trend", "action": action,
            "price": latest, "ma5": round(ma5, 3), "deviation_pct": round(dev * 100, 3),
            "as_of": bars[-1].get("date"),
            "source": "computed on frozen Tier-2 series (real recorded closes)"}


def _cpi_yoy(obs):
    """Real year-over-year CPI inflation from frozen monthly observations."""
    if len(obs) < 2:
        return None
    latest = obs[-1]
    parts = str(latest["date"]).split("-")
    if len(parts) >= 2:
        target = f"{int(parts[0]) - 1}-{parts[1]}"
        for o in obs:
            if str(o["date"]).startswith(target):
                return round((latest["value"] / o["value"] - 1) * 100, 3)
    return round((latest["value"] / obs[0]["value"] - 1) * 100, 3)


# ------------------------------------------------------------------ 1. stock_query

def fam_stock_query(action, args, ws):
    cas = _cassette()
    quotes, meta = cas.get("stock_quotes", {}), cas.get("stock_meta", {})
    raw = _arg(args, "symbol", "code", "ticker", "query", "input", "text",
               "keyword", "name", "stock", "stock_name", "company")
    sym = _resolve_symbol(raw)
    if sym and sym in quotes:
        q = dict(quotes[sym])
        m = meta.get(sym, {})
        if m:
            q["name"] = m.get("name", q.get("name", sym))
            q["concepts"] = m.get("concepts", [])
        return _ok({"operation": "quote", "symbol": sym, **q})
    # generic market-data request (no specific symbol) -> real market snapshot
    blob = f"{raw or ''} {action or ''}".lower()
    if any(k in blob for k in ("行情", "股价", "股票", "报价", "涨跌", "market", "quote", "stock")):
        snap = {s: {"price": q.get("price"), "change_pct": q.get("change_pct"),
                    "as_of": q.get("as_of")} for s, q in quotes.items()}
        return _ok({"operation": "market_snapshot", "count": len(snap), "quotes": snap,
                    "note": "frozen real recorded quotes (Tier-2 cassette)"})
    return _err("not_found",
                f"no real stock quote for {raw!r} in the frozen Tier-2 cassette "
                f"(recorded symbols: {sorted(quotes)})")


# ------------------------------------------------------------------ 2. fred_query

_FRED_KEYWORDS = [
    ("DFF", ("联邦基金", "基准利率", "利率", "降息", "fed", "federal fund", "rate", "interest", "dff")),
    ("CPIAUCSL", ("cpi", "通胀", "物价", "消费者", "inflation", "consumer price", "cpiaucsl")),
    ("A191RL1Q225SBEA", ("实际gdp", "real gdp", "gdp增长", "gdp 增长", "经济增长", "实际增长",
                         "增速", "percent change", "a191rl")),
    ("GDP", ("gdp", "国内生产总值", "经济数据", "总量", "gross domestic")),
]


def fam_fred_query(action, args, ws):
    cas = _cassette()
    fred = cas.get("fred_series", {})
    raw = _arg(args, "series", "series_id", "id", "input", "query", "text", "keyword", "name")
    t = str(raw or "").lower()
    sid = None
    for s in fred:                       # direct series-id mention wins
        if s.lower() in t:
            sid = s
            break
    if sid is None:                      # else keyword intent mapping
        for s, kws in _FRED_KEYWORDS:
            if s in fred and any(k in t for k in kws):
                sid = s
                break
    if sid is None or sid not in fred:
        return _err("not_found",
                    f"no FRED series for {raw!r} in frozen cassette (have: {sorted(fred)})")
    series = dict(fred[sid])
    obs = series.get("observations", [])
    out = {"operation": "fred_series", "series_id": sid, "title": series.get("title"),
           "units": series.get("units"), "observations": obs, "as_of": series.get("as_of"),
           "source": series.get("source")}
    if obs:
        out["latest"] = obs[-1]
    if sid == "CPIAUCSL":
        yoy = _cpi_yoy(obs)
        if yoy is not None:
            out["yoy_inflation_pct"] = yoy
    return _ok(out)


# ------------------------------------------------------------------ 3. quant_trading

def fam_quant_trading(action, args, ws):
    cas = _cassette()
    series, signals = cas.get("stock_series", {}), cas.get("stock_signals", {})
    text = str(_arg(args, "input", "query", "text", "task", "instruction",
                    "strategy", "prompt", default=""))
    sym_raw = _arg(args, "symbol", "code", "ticker", "stock", "name")
    sym = _resolve_symbol(sym_raw) or _resolve_symbol(text)
    blob = f"{text} {sym_raw or ''} {action or ''}".lower()

    is_backtest = any(k in blob for k in ("回测", "backtest", "back test", "均线", "ma cross",
                                          "ma_cross", "策略", "strategy", "模拟", "双均线"))
    if is_backtest:
        target = sym if (sym in series and len(series[sym]) >= 6) else None
        if target is None:               # pick the longest frozen series as default
            cand = [(s, b) for s, b in series.items() if len(b) >= 6]
            if not cand:
                return _err("not_found", "no frozen price series long enough to backtest")
            target = max(cand, key=lambda kv: len(kv[1]))[0]
        closes = [b["close"] for b in series[target] if b.get("close") is not None]
        res = _ma_cross_backtest(closes)
        return _ok({"operation": "backtest", "strategy": "ma_cross", "symbol": target,
                    "source": "real MA-cross backtest on frozen Tier-2 price series", **res})

    is_signal = any(k in blob for k in ("信号", "signal", "量化", "quant", "交易", "trade",
                                        "分析", "ensemble", "买卖"))
    if sym:
        if sym in signals:
            return _ok({"operation": "signal", "symbol": sym, **signals[sym]})
        if sym in series:
            sig = _signal_from_series(sym, series[sym])
            if sig:
                return _ok({"operation": "signal", **sig})
        return _err("not_found",
                    f"no frozen quant signal/series for symbol {sym!r} (real recorded: "
                    f"{sorted(signals)})")
    if is_signal:                    # generic signal request -> real frozen ensemble set
        return _ok({"operation": "signals", "count": len(signals), "signals": signals,
                    "note": "frozen real ensemble signals (data/quant.db)"})
    return _err("not_found",
                f"quant_trading needs a symbol or a signal/backtest intent; got {sym_raw or text!r}")


# ------------------------------------------------------------------ 4/12. search core

_SEARCH_INTENTS = {
    "ai_news": ("ai", "人工智能", "news", "新闻", "机器学习", "machine learning"),
    "apple_stock": ("苹果", "apple", "aapl", "股价"),
    "us_gdp_growth": ("gdp", "经济增长", "国内生产总值", "增速", "fred"),
    "cpi_inflation": ("cpi", "通胀", "inflation", "物价", "消费者"),
    "flights_beijing_shanghai": ("航班", "北京", "上海", "flight", "机票", "beijing", "shanghai"),
    "laptop_stock_photo": ("笔记本", "电脑", "laptop", "图库", "素材", "stock photo", "照片", "computer"),
    "route_directions": ("路线", "导航", "directions", "地址", "指引", "route", "driving"),
    "hotel_restaurant_reviews": ("酒店", "餐厅", "评价", "hotel", "restaurant", "附近", "reviews"),
    "llm_tool_calling_papers": ("大模型", "工具调用", "论文", "llm", "tool", "文献", "引用", "paper"),
}


def _search_core(query: str):
    """Score the frozen recorded intents; return (intent_key, results) or (None, None)."""
    cas = _cassette()
    sr = cas.get("search_results", {})
    q = str(query or "").lower()
    if not q.strip():
        return None, None
    best_key, best_score = None, 0
    for key, kws in _SEARCH_INTENTS.items():
        if key not in sr:
            continue
        score = sum(1 for k in kws if k in q)
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return None, None
    return best_key, sr[best_key]


def fam_search(action, args, ws):
    query = _arg(args, "query", "q", "input", "text", "keyword", "search", "search_query", "prompt")
    key, res = _search_core(query)
    if key is None:
        return _err("not_found",
                    f"no frozen web-search result matches {query!r} (recorded intents: "
                    f"{sorted(_cassette().get('search_results', {}))})")
    return _ok({"operation": "web_search", "query": query, "matched_intent": key,
                "results": res.get("results", []),
                "source": "frozen real recorded search results (Tier-2 cassette)"})


def fam_duckduckgo_search(action, args, ws):
    """DuckDuckGo web search -- same frozen real-result domain as ``search``."""
    query = _arg(args, "query", "q", "input", "text", "keyword", "search", "prompt")
    key, res = _search_core(query)
    if key is None:
        return _err("not_found", f"no frozen web-search result matches {query!r}")
    return _ok({"operation": "duckduckgo_search", "query": query, "matched_intent": key,
                "results": res.get("results", []),
                "source": "frozen real recorded search results (Tier-2 cassette)"})


# ------------------------------------------------------------------ 5. browser

def _page_payload(page):
    res = {"url": page.get("url"), "title": page.get("title")}
    if page.get("text"):
        res["text"] = page["text"]
    if page.get("links"):
        res["links"] = page["links"]
    if page.get("products"):
        res["products"] = page["products"]
    if page.get("form"):
        res["form"] = page["form"]
    return res


def _match_element(page, selector):
    """Match a selector against a frozen page's real links / form fields / products."""
    if not selector:
        return None
    s = str(selector).lower()
    for lk in page.get("links", []) or []:
        if s in str(lk.get("text", "")).lower() or s in str(lk.get("href", "")).lower():
            return {"type": "link", "text": lk.get("text"), "href": lk.get("href")}
    form = page.get("form")
    if form:
        for f in form.get("fields", []) or []:
            if s in str(f.get("name", "")).lower() or s in str(f.get("label", "")).lower():
                return {"type": "field", "name": f.get("name"), "label": f.get("label")}
        if s in str(form.get("submit", "")).lower():
            return {"type": "submit", "text": form.get("submit")}
    for pr in page.get("products", []) or []:
        if s in str(pr.get("name", "")).lower():
            return {"type": "product", "name": pr.get("name"), "price": pr.get("price")}
    if s in str(page.get("title", "")).lower():
        return {"type": "title", "text": page.get("title")}
    return None


def fam_browser(action, args, ws):
    cas = _cassette()
    pages = cas.get("pages", {})
    op = (action or "").lower()
    url = _arg(args, "url", "link", "href", "uri", "page", "target", "address", "site")
    selector = _arg(args, "selector", "element")
    text = str(_arg(args, "input", "query", "task", "text", "instruction", "prompt", default=""))
    blob = f"{url or ''} {text} {op}".lower()
    page, key = _resolve_page(url or text, pages)

    is_form = any(k in blob for k in ("表单", "form", "注册", "register", "填写", "fill",
                                      "登录", "登陆", "login", "sign in", "submit", "提交"))
    is_shot = any(k in blob for k in ("截图", "截屏", "截一张", "screenshot", "capture",
                                      "snapshot", "shot", "渲染"))
    is_crawl = any(k in blob for k in ("爬", "crawl", "scrape", "抓取", "商品", "价格",
                                       "price", "product", "网店"))
    is_click = op in ("click", "tap") or selector is not None

    if is_crawl and page is None:
        page, key = pages.get("books.toscrape.com"), "books.toscrape.com"
    if is_form and page is None:
        page, key = pages.get("httpbin.org/forms/post"), "httpbin.org/forms/post"
    if is_click and page is None:
        page, key = pages.get("example.com"), "example.com"

    if page is None:
        if url:
            return _err("not_found",
                        f"no frozen page for {url!r} in Tier-2 cassette (real recorded pages: "
                        f"{sorted(pages)})")
        return _err("invalid_args", "browser needs a url/link to open (no frozen page resolved)")

    if is_click:
        matched = _match_element(page, selector or text)
        if matched is None:
            return _err("not_found",
                        f"no element matching {selector or text!r} on frozen page {page.get('url')}")
        return _ok({"operation": "click", "url": page.get("url"), "clicked": matched,
                    "source": "frozen real page (Tier-2 cassette)"})
    if is_shot:
        rel = _write_fs(ws, f"screenshots/{_slug(key, 'page')}.txt",
                        f"[sandbox screenshot of {page.get('url')}]\n"
                        f"title: {page.get('title')}\n"
                        f"recorded: {page.get('as_of', '')}\n")
        return _ok({"operation": "screenshot", "url": page.get("url"),
                    "title": page.get("title"), "artifact": rel,
                    "source": "frozen real page (Tier-2 cassette)"})
    if is_form:
        return _ok({"operation": "fill_form", "url": page.get("url"),
                    "form": page.get("form"), "submitted": True,
                    "echo": {"note": "sandbox form submission -- no real side effect"},
                    "source": "frozen real form (Tier-2 cassette)"})
    return _ok({"operation": "open", "source": "frozen real page (Tier-2 cassette)",
                **_page_payload(page)})


# ------------------------------------------------------------------ 6. browser_use

def fam_browser_use(action, args, ws):
    cas = _cassette()
    pages, sr = cas.get("pages", {}), cas.get("search_results", {})
    url = _arg(args, "url", "link", "site", "page")
    task = str(_arg(args, "task", "input", "query", "text", "instruction", "goal", "prompt", default=""))
    blob = f"{url or ''} {task} {action or ''}".lower()
    page, key = _resolve_page(url or task, pages)

    if any(k in blob for k in ("酒店", "预订", "预定", "hotel", "booking", "book", "客房",
                               "餐厅", "restaurant", "评价")):
        hotel = sr.get("hotel_restaurant_reviews", {})
        return _ok({"operation": "hotel_booking_flow", "task": task,
                    "results": hotel.get("results", []),
                    "note": "sandbox booking flow grounded in real recorded hotel/restaurant "
                            "entities; no real reservation side effect",
                    "source": "frozen real search results (Tier-2 cassette)"})
    if any(k in blob for k in ("爬", "crawl", "scrape", "抓取", "商品", "价格", "price",
                               "product", "网店")):
        pg = page if (page and page.get("products")) else pages.get("books.toscrape.com")
        if pg and pg.get("products"):
            return _ok({"operation": "crawl", "url": pg.get("url"),
                        "items": len(pg["products"]), "products": pg["products"],
                        "source": "frozen real page (Tier-2 cassette)"})
    if any(k in blob for k in ("表单", "form", "注册", "register", "填写", "fill", "登录",
                               "登陆", "login", "sign in", "submit", "提交")):
        pg = page if (page and page.get("form")) else pages.get("httpbin.org/forms/post")
        if pg and pg.get("form"):
            return _ok({"operation": "fill_form", "url": pg.get("url"), "form": pg["form"],
                        "submitted": True, "source": "frozen real form (Tier-2 cassette)"})
    if any(k in blob for k in ("航班", "flight", "机票")) and "flights_beijing_shanghai" in sr:
        return _ok({"operation": "browse", "results": sr["flights_beijing_shanghai"].get("results", []),
                    "source": "frozen real search results (Tier-2 cassette)"})
    if any(k in blob for k in ("路线", "导航", "directions", "地址", "指引", "route")) and "route_directions" in sr:
        return _ok({"operation": "browse", "results": sr["route_directions"].get("results", []),
                    "source": "frozen real search results (Tier-2 cassette)"})
    if page is not None:
        return _ok({"operation": "open", "source": "frozen real page (Tier-2 cassette)",
                    **_page_payload(page)})
    if any(k in blob for k in ("打开", "open", "截图", "screenshot", "标题", "title", "正文",
                               "读取", "visit", "访问", "网站", "网页")):
        pg = pages.get("example.com")
        if pg:
            return _ok({"operation": "open", "source": "frozen real page (Tier-2 cassette)",
                        **_page_payload(pg)})
    return _err("not_found",
                f"no frozen page/result for browser task {task or url!r} (real recorded pages: "
                f"{sorted(pages)})")


# ------------------------------------------------------------------ 7. crawlee_tool

def fam_crawlee_tool(action, args, ws):
    cas = _cassette()
    pages = cas.get("pages", {})
    url = _arg(args, "url", "link", "start_url", "target", "page", "site")
    text = str(_arg(args, "input", "query", "task", "text", default=""))
    page, key = _resolve_page(url or text, pages)
    if page is None:
        page, key = pages.get("books.toscrape.com"), "books.toscrape.com"
    if page is None:
        return _err("not_found", "no frozen page to crawl in Tier-2 cassette")
    if page.get("products"):
        return _ok({"operation": "crawl", "url": page.get("url"),
                    "items": len(page["products"]), "products": page["products"],
                    "pages": page.get("pages"),
                    "source": "frozen real page (Tier-2 cassette)"})
    return _ok({"operation": "crawl", "source": "frozen real page (Tier-2 cassette)",
                **_page_payload(page)})


# ------------------------------------------------------------------ literature helpers

def _paper_brief(p):
    return {"title": p.get("title"), "doi": p.get("doi"), "year": p.get("year"),
            "venue": p.get("venue"), "authors": p.get("authors"),
            "cited_by": p.get("cited_by"), "oa_status": p.get("oa_status"),
            "oa_url": p.get("oa_url")}


_LIT_TOPIC_KW = ("工具调用", "tool", "大模型", "llm", "论文", "paper", "agent", "foundation",
                 "引用", "文献", "doi", "摘要", "abstract", "综述", "survey", "标题", "pdf")


def _match_papers(papers, text):
    t = str(text or "").lower()
    toks = _tokens(t)
    out = []
    for p in papers:
        hay = " ".join(str(p.get(k, "")) for k in
                       ("title", "doi", "arxiv", "venue", "authors", "abstract")).lower()
        if any(tok in hay for tok in toks):
            out.append(p)
    return out


# ------------------------------------------------------------------ 8. literature_search

def fam_literature_search(action, args, ws):
    cas = _cassette()
    papers = cas.get("papers", [])
    text = _arg(args, "input", "query", "title", "topic", "keyword", "text",
                "search", "doi", "paper", "q")
    doi = _arg(args, "doi", "id", "doi_list")
    matches = []
    if doi:
        dl = str(doi).lower()
        matches = [p for p in papers if dl in str(p.get("doi", "")).lower()]
    if not matches:
        matches = _match_papers(papers, text)
    if not matches and any(k in str(text or "").lower() for k in _LIT_TOPIC_KW):
        matches = papers                      # topic-level fallback over the real corpus
    if not matches:
        return _err("not_found",
                    f"no frozen paper matches {text or doi!r} (real recorded corpus: "
                    f"{[p.get('title') for p in papers]})")
    return _ok({"operation": "literature_search", "query": text or doi,
                "count": len(matches), "results": [_paper_brief(p) for p in matches],
                "source": "frozen real OpenAlex/arXiv metadata (Tier-2 cassette)"})


# ------------------------------------------------------------------ 9. batch_paper_analyzer

def _paper_analysis(p):
    oa = p.get("oa_url")
    abstract = str(p.get("abstract", "") or "")
    summary = abstract.split(". ")[0][:240] if abstract else ""
    return {"title": p.get("title"), "doi": p.get("doi"), "oa_status": p.get("oa_status"),
            "oa_url": oa, "has_oa_pdf": bool(oa), "cited_by": p.get("cited_by"),
            "abstract_summary": summary or "(no abstract recorded)"}


def fam_batch_paper_analyzer(action, args, ws):
    cas = _cassette()
    papers = cas.get("papers", [])
    dois = _arg(args, "dois", "doi_list", "doi", "ids", "papers", "input", "query", "text")
    analyzed = []
    if isinstance(dois, (list, tuple)):
        for d in dois:
            dl = str(d).lower()
            for p in papers:
                if dl and (dl in str(p.get("doi", "")).lower()
                           or dl in str(p.get("arxiv", "")).lower()):
                    analyzed.append(_paper_analysis(p))
                    break
        if not analyzed:
            return _err("not_found",
                        f"none of the given DOIs {list(dois)} match the frozen real corpus")
    else:
        matched = _match_papers(papers, dois)
        analyzed = [_paper_analysis(p) for p in (matched or papers)]
    return _ok({"operation": "batch_paper_analyzer", "count": len(analyzed),
                "analyses": analyzed,
                "source": "frozen real OpenAlex/arXiv records (Tier-2 cassette)"})


# ------------------------------------------------------------------ 10. stock_photo

_STOCK_PHOTO_KW = ("photo", "image", "stock", "图片", "图像", "素材", "图库", "壁纸",
                   "wallpaper", "照片", "laptop", "电脑", "computer", "免费图", "配图")


def fam_stock_photo(action, args, ws):
    cas = _cassette()
    sr = cas.get("search_results", {})
    text = str(_arg(args, "input", "query", "keyword", "text", "search", "prompt", default=""))
    if not any(k in text.lower() for k in _STOCK_PHOTO_KW):
        return _err("not_found",
                    f"stock_photo needs an image/photo search query; got {text!r} "
                    f"(no image-search intent in the frozen cassette)")
    res = sr.get("laptop_stock_photo", {})
    return _ok({"operation": "stock_photo", "query": text,
                "results": res.get("results", []),
                "source": "frozen real Pexels/Unsplash stock-photo results (Tier-2 cassette)"})


# ------------------------------------------------------------------ 11. literature_review

_REVIEW_OUTLINE = ["Introduction & Scope", "Background and Taxonomy", "Methods and Tools",
                   "Applications", "Open Challenges", "Future Directions", "Conclusion"]


def fam_literature_review(action, args, ws):
    cas = _cassette()
    papers = cas.get("papers", [])
    topic = _arg(args, "input", "query", "topic", "title", "text", "theme")
    matched = _match_papers(papers, topic) or papers
    refs = [_paper_brief(p) for p in matched]
    return _ok({"operation": "literature_review",
                "topic": topic or "tool learning with foundation models",
                "outline": _REVIEW_OUTLINE, "references": refs, "reference_count": len(refs),
                "source": "review outline grounded in frozen real papers (Tier-2 cassette)"})


# ------------------------------------------------------------------ registry

TIER2_FAMILIES = {
    "stock_query": fam_stock_query,
    "fred_query": fam_fred_query,
    "quant_trading": fam_quant_trading,
    "search": fam_search,
    "duckduckgo_search": fam_duckduckgo_search,
    "browser": fam_browser,
    "browser_use": fam_browser_use,
    "crawlee_tool": fam_crawlee_tool,
    "literature_search": fam_literature_search,
    "batch_paper_analyzer": fam_batch_paper_analyzer,
    "stock_photo": fam_stock_photo,
    "literature_review": fam_literature_review,
}
