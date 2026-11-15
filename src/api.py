"""Trading — FastAPI backend.

Serves the scan tables (built by `pixi run update-data`) as JSON,
plus the Trading.html frontend at the site root.

Run with:  pixi run ui   ->   http://localhost:8500

Endpoints:
    GET  /                              -> Trading.html (the UI)
    GET  /api/health                    -> {"status": "ok"}
    GET  /api/markets/{m}/scan          -> full scan table for one market
    GET  /api/last-updated              -> {"last_updated": "<iso>" | null}
    POST /api/refresh-index             -> run `pixi run refresh_market_index_list`
    POST /api/update-data               -> run `pixi run update-data`
    GET  /logo.png                      -> the app logo
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_data import MARKETS, ROOT, read_last_updated  # noqa: E402
from indicators import catalogue as indicator_catalogue  # noqa: E402
from indicators import compute as indicator_compute  # noqa: E402

app = FastAPI(title="Trading")


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Answer every unhandled error with JSON instead of plain text.

    Starlette's default 500 body is the literal string "Internal Server
    Error". The UI calls `r.json()` on every response, so that text surfaces
    as "Unexpected token 'I', \"Internal S\"... is not valid JSON" and hides
    the real cause. Returning JSON keeps the failure readable in the UI.
    """
    return JSONResponse(
        {"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


# (column key, label, kind, decimals) — mirrors the old Streamlit column_config.
# kind: "text" | "num" | "int" | "bool" | "rsi"  (rsi renders as a 0-100 bar).
COL_SPECS = [
    ("symbol", "Ticker", "text", None),
    ("name", "Name", "text", None),
    ("sector", "Sector", "text", None),
    ("price", None, "num", 2),            # label gets the currency: "Price (USD)"
    ("Discount Zone", "Discount Zone", "num", 1),
    ("Premium Zone", "Premium Zone", "num", 1),
    ("ADX", "ADX", "num", 1),
    ("RVOL", "RVOL", "num", 2),
    ("% off 52W high", "% off 52W high", "num", 1),
    ("1D %", "1D %", "num", 2),
    ("5D %", "5D %", "num", 2),
    ("21D %", "21D %", "num", 2),
    ("YTD %", "YTD %", "num", 2),
    ("RSI(14)", "RSI(14)", "rsi", 1),
    ("MACD hist", "MACD hist", "num", 4),
    ("ATR(14)", "ATR(14)", "num", 4),
    ("OBV slope", "OBV slope", "num", 1),
    ("SMA20", "SMA20", "num", 2),
    ("SMA50", "SMA50", "num", 2),
    ("SMA200", "SMA200", "num", 2),
    ("52W high", "52W high", "num", 2),
    ("52W low", "52W low", "num", 2),
    ("21D support", "21D support", "num", 2),
    ("21D resistance", "21D resistance", "num", 2),
    ("last bar", "As of", "text", None),
    ("bars", "Bars", "int", None),
]


def _py(x):
    """Make one cell JSON-serialisable: NaN/NaT -> None, numpy -> plain python."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if pd.isna(x) else float(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, str):
        return x
    return str(x)


def payload(market: str) -> dict | None:
    """The scan table for one market as a compact JSON dict (None = no data yet)."""
    p = MARKETS[market]["table"]
    if not p.exists():
        return None
    df = pd.read_parquet(p)  # symbol stays a regular column (it leads each row)
    cfg = MARKETS[market]

    cols = []
    for key, label, kind, dec in COL_SPECS:
        if key not in df.columns:
            continue
        c = {
            "key": key,
            "label": f"Price ({cfg['currency']})" if key == "price" else label,
            "kind": kind,
        }
        if dec is not None:
            c["dec"] = dec
        # filter bounds for numeric columns (skipped when constant or all-null)
        if kind in ("num", "int", "rsi"):
            vals = df[key].dropna()
            if not vals.empty:
                vmin, vmax = float(vals.min()), float(vals.max())
                if vmin != vmax:
                    step = float(round(abs(vmax - vmin) / 500 or 0.01, 6))
                    c.update({"min": vmin, "max": vmax, "step": step})
        cols.append(c)

    keys = [c["key"] for c in cols]
    sub = df[keys].astype(object)
    rows = [[_py(x) for x in row] for row in sub.values.tolist()]

    last_bar = _py(df["last bar"].iloc[0]) if "last bar" in df.columns else ""
    return {
        "market": market,
        "label": cfg["label"],
        "currency": cfg["currency"],
        "last_bar": last_bar or "",
        "n": int(len(df)),
        "sectors": sorted({str(s) for s in df["sector"].dropna()}),
        "columns": cols,
        "rows": rows,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/last-updated")
def last_updated() -> dict:
    """Timestamp of the last successful `update-data` run (null if never)."""
    return {"last_updated": read_last_updated()}


def _run_task(task: str, extra: list[str] | None = None,
              timeout: int = 300) -> JSONResponse:
    """Run a pixi task in the project root and return its log as JSON.

    Falls back to invoking the script directly with the current interpreter
    when `pixi` is not on PATH (e.g. the server was started outside pixi).
    """
    extra = extra or []
    cmd = ["pixi", "run", task, *extra]
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        # Map the pixi task to its direct entry point.
        direct = {
            "refresh_market_index_list": ["--refresh-index"],
            "update-data": [],
        }.get(task)
        if direct is None:
            return JSONResponse(
                {"ok": False, "error": f"pixi not found and no fallback for '{task}'"},
                status_code=500,
            )
        fallback = [sys.executable, str(ROOT / "src" / "get_data.py"), *direct, *extra]
        try:
            proc = subprocess.run(
                fallback, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return JSONResponse(
                {"ok": False, "error": f"{task} timed out after {timeout}s"},
                status_code=504,
            )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            {"ok": False, "error": f"{task} timed out after {timeout}s"},
            status_code=504,
        )

    out = (proc.stdout or "") + (proc.stderr or "")
    # Drop pixi's environment noise (SSL_CERT_DIR warning, task banner) so the
    # UI shows just the pipeline log.
    lines = [
        ln for ln in out.splitlines()
        if ln.strip()
        and "SSL_CERT_DIR" not in ln
        and not ln.startswith("Pixi task (")
    ]
    return JSONResponse({
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "log": "\n".join(lines).strip(),
    }, status_code=200 if proc.returncode == 0 else 500)


@app.post("/api/refresh-index")
def refresh_index(market: str | None = None):
    """Run `pixi run refresh_market_index_list` (Step 1_Refresh_Index_list).

    Deletes the cached index files and rebuilds them from upstream. Blocks
    until the command finishes and returns its log so the UI can show it.
    `market` may be "us" or "india" to refresh just one side.
    """
    extra = []
    if market in MARKETS:
        extra.append("--" + ("usa" if market == "us" else "india"))
    return _run_task("refresh_market_index_list", extra)


@app.post("/api/update-data")
def update_data(market: str | None = None):
    """Run `pixi run update-data` (Step 2 prices + Step 3 indicators).

    Step 1_Refresh_Index_list is bypassed — the cached index list is reused.
    `market` may be "us" or "india" to update just one side.
    """
    extra = []
    if market in MARKETS:
        extra.append("--" + ("usa" if market == "us" else "india"))
    return _run_task("update-data", extra, timeout=1800)


@app.get("/api/markets/{market}/scan")
def scan(market: str):
    if market not in MARKETS:
        return JSONResponse({"error": f"unknown market '{market}'"}, status_code=404)
    p = payload(market)
    if p is None:
        task = "update-data-us" if market == "us" else "update-data-india"
        return JSONResponse(
            {"error": "no data", "task": task}, status_code=404
        )
    return p


# ---- chart data -----------------------------------------------------------
# Resample rule per interval. Daily bars are the only thing cached, so
# anything coarser is aggregated from them; intraday is not available.
_RESAMPLE = {"1D": None, "1W": "W-FRI", "1M": "ME"}


def _series(index, values, nd=2):
    """[(date, value)] -> [{time, value}] with NaN dropped (Lightweight Charts
    rejects nulls, so gaps are simply omitted)."""
    out = []
    for t, v in zip(index, values):
        if v is None or pd.isna(v):
            continue
        out.append({"time": t.strftime("%Y-%m-%d"), "value": round(float(v), nd)})
    return out


@app.get("/api/prices/{market}/{symbol}")
def prices(market: str, symbol: str, interval: str = "1D"):
    """OHLCV + indicator series for one ticker, for the chart modal.

    `interval` is 1D (raw daily bars), 1W or 1M (aggregated from daily).
    Indicators are computed on the *displayed* series so the overlays line up
    with the candles.
    """
    if market not in MARKETS:
        return JSONResponse({"error": f"unknown market '{market}'"}, status_code=404)
    cfg = MARKETS[market]
    p = cfg["prices"] / f"{symbol}.parquet"
    if not p.exists():
        return JSONResponse(
            {"error": f"no price data for {symbol}"}, status_code=404)

    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()

    # Yahoo writes a placeholder row for the session that is still trading:
    # Volume is filled in but Open/High/Low/Close are NaN. Those NaNs cannot be
    # JSON-encoded (Starlette's JSONResponse uses allow_nan=False), so one such
    # row used to turn the whole chart request into a plain-text 500. Drop any
    # bar without a complete OHLC set and de-duplicate the index.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    if df.empty:
        return JSONResponse(
            {"error": f"no usable price data for {symbol}"}, status_code=404)

    rule = _RESAMPLE.get(interval.upper())
    if rule:
        df = df.resample(rule).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        })
        # A week/month with no trading days aggregates to all-NaN; drop it.
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        return JSONResponse(
            {"error": f"no usable price data for {symbol}"}, status_code=404)

    close = df["Close"]
    vol = df["Volume"].fillna(0) if "Volume" in df.columns else None
    candles = [{
        "time": t.strftime("%Y-%m-%d"),
        "open": round(float(o), 2), "high": round(float(h), 2),
        "low": round(float(l), 2), "close": round(float(c), 2),
    } for t, o, h, l, c in zip(df.index, df["Open"], df["High"],
                               df["Low"], df["Close"])]
    volume = [] if vol is None else [
        {"time": t.strftime("%Y-%m-%d"), "value": int(v),
         "color": "#26a69a80" if c >= o else "#ef535080"}
        for t, o, c, v in zip(df.index, df["Open"], df["Close"], vol)]

    # Indicators on the displayed series (same length as the candles).
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    return {
        "market": market,
        "symbol": symbol,
        "currency": cfg["currency"],
        "interval": interval.upper(),
        "bars": len(candles),
        "candles": candles,
        "volume": volume,
        "sma20": _series(df.index, sma20),
        "sma50": _series(df.index, sma50),
        "sma200": _series(df.index, sma200),
        "rsi14": _series(df.index, rsi14, 1),
        "macd": _series(df.index, macd_line, 4),
        "macdSignal": _series(df.index, macd_sig, 4),
        "macdHist": _series(df.index, macd_hist, 4),
    }


# ---- strategies / indicators ----------------------------------------------
# Every module under src/indicators that exposes INDICATOR + compute() is
# auto-registered (see src/indicators/__init__.py). The UI lists them under the
# "Custom" group of the indicator menu and fetches one's overlay from
# /api/strategies/{id}/apply.

def _load_bars(market: str, symbol: str, interval: str) -> pd.DataFrame | None:
    """The same OHLC frame the chart shows, so overlays line up with candles."""
    if market not in MARKETS:
        return None
    p = MARKETS[market]["prices"] / f"{symbol}.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    rule = _RESAMPLE.get(interval.upper())
    if rule:
        df = df.resample(rule).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        })
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df if not df.empty else None


@app.get("/api/strategies")
def strategies() -> dict:
    """Catalogue of every indicator in src/indicators, for the Strategy menu."""
    return {"strategies": indicator_catalogue()}


@app.get("/api/strategies/{indicator_id}/apply")
def apply_strategy(indicator_id: str, market: str, symbol: str,
                   interval: str = "1D", params: str | None = None):
    """Run one indicator over one ticker and return its overlay payload.

    `params` is an optional JSON object overriding the indicator's defaults,
    e.g. `?params={"swingLength":20}`.
    """
    if market not in MARKETS:
        return JSONResponse({"error": f"unknown market '{market}'"}, status_code=404)

    df = _load_bars(market, symbol, interval)
    if df is None:
        return JSONResponse(
            {"error": f"no price data for {symbol}"}, status_code=404)

    overrides: dict = {}
    if params:
        try:
            overrides = json.loads(params)
            if not isinstance(overrides, dict):
                raise ValueError("params must be a JSON object")
        except Exception as exc:
            return JSONResponse(
                {"error": f"bad params: {exc}"}, status_code=400)

    try:
        result = indicator_compute(indicator_id, df, overrides)
    except Exception as exc:
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    if result is None:
        return JSONResponse(
            {"error": f"unknown strategy '{indicator_id}'"}, status_code=404)

    return {
        "market": market,
        "symbol": symbol,
        "interval": interval.upper(),
        "strategy": indicator_id,
        "bars": int(len(df)),
        "overlay": result,
    }


# catch-all so unknown /api/... paths get a JSON 404 (not the frontend index.html)
@app.get("/api/{path:path}")
def api_not_found(path: str):
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/")
def index() -> HTMLResponse:
    """Serve the Trading UI (React + Tailwind single-page app)."""
    html_path = ROOT / "Trading.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Trading.html not found</h1>", status_code=404)


@app.get("/logo.png")
def logo() -> FileResponse:
    """App logo + splash animation source (lives with the other static assets)."""
    return FileResponse(ROOT / "cdn" / "logo.png")


@app.get("/cdn/{name:path}")
def cdn(name: str) -> FileResponse:
    """Serve the locally-cached frontend libraries (see /cdn).

    Everything is pinned and cached so the app never needs the internet:
    react 19.3.1 + react-dom 19.3.1 (React 19 ships no UMD build, so these
    are the official CommonJS production builds wrapped into browser globals
    by src/update_cdn.py), tailwind 4.3.5 (official browser build), Tabulator
    6.7.1 (MIT, UMD) which renders the data table, Lightweight Charts
    5.2.1 (standalone) for the chart modal, and talib-web 0.1.3
    (UMD + talib.wasm) for indicators.
    """
    f = (ROOT / "cdn" / name).resolve()
    if not f.is_relative_to((ROOT / "cdn").resolve()) or not f.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
