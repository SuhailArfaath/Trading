"""Trading data pipeline - three steps, one module.

  STEP 1_Refresh_Index_list : Getting stocks list (USA S&P 500 + India Nifty 500)
  STEP 2 Incremental Load data from year 2000 (Yahoo Finance, parquet cache)
  STEP 3 : Adding indicators to each parquet file (writes the UI scan table)

Step 1_Refresh_Index_list runs ONLY via `pixi run refresh_market_index_list`.
It deletes the four cached index files and rebuilds them from upstream.
`pixi run update-data` always bypasses Step 1 and reuses the cached files.

Outputs (under data/):
  tickers_us_detail.json     index membership (Wikipedia)
  tickers_in_detail.json     index membership (NSE CSV)
  prices_us/<SYMBOL>.parquet, prices_in/<SYMBOL>.parquet  daily OHLCV
  scan_table_us.parquet      flattened table the UI displays (USA)
  scan_table_in.parquet      (India)
  last_updated.json          {"last_updated": "<iso timestamp>"} for the run

Incremental by default: a normal run reads last_updated.json, asks Yahoo only
for bars from that date onwards, and APPENDS them to the existing parquet.
`--force` is the only mode that re-downloads the full 2000-onward history and
overwrites every file.

Run with:
    pixi run refresh_market_index_list      # Step 1 only: rebuild index files
    pixi run update-data                    # both markets, incremental append
    pixi run update-data --usa              # USA only
    pixi run update-data --india            # India only
    pixi run update-data --force            # full reload from 2000, overwrite
    pixi run update-data --usa --start 2023-01-01 --interval 1d
                                            # custom Yahoo download window
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import yfinance as yf

from indicators import smart_money

log = logging.getLogger("trading")


# ======================================================================
# Shared config - paths + per-market settings
# ======================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each market: its own cache files (detail/, prices/, table) so the USA and
# India tabs never touch each other's data. The detail file is the single
# source of truth for index membership — the symbol list is derived from it.
MARKETS = {
    "us": {
        "label": "USA — S&P 500",
        "suffix": "",            # Yahoo symbol = display symbol
        "currency": "USD",
        "detail": DATA_DIR / "tickers_us_detail.json",
        "prices": DATA_DIR / "prices_us",
        "table": DATA_DIR / "scan_table_us.parquet",
    },
    "india": {
        "label": "India — Nifty 500",
        "suffix": ".NS",         # Yahoo symbol = display symbol + ".NS"
        "currency": "INR",
        "detail": DATA_DIR / "tickers_in_detail.json",
        "prices": DATA_DIR / "prices_in",
        "table": DATA_DIR / "scan_table_in.parquet",
    },
}
for _m in MARKETS.values():
    _m["prices"].mkdir(parents=True, exist_ok=True)
# ======================================================================
# STEP 1_Refresh_Index_list : Getting stocks list (USA S&P 500 + India Nifty 500)
#   Which tickers are in each index? Fetched once, then cached to
#   data/tickers*.json so the app works offline.
#   USA source:  Wikipedia "List of S&P 500 companies"
#   India source: NSE's official index-constituents CSV
#
#   This step runs ONLY via `pixi run refresh_market_index_list`. It deletes
#   the cached detail file and rebuilds it from upstream.
#   `pixi run update-data` never touches it — it just reads the cache.
# ======================================================================
HEADERS = {"User-Agent": "Mozilla/5.0 (trading-scraper; research use)"}
US_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IN_NSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


def _us_records() -> list[dict]:
    # pandas 3.0: fetch HTML ourselves, then parse (headers are not forwarded).
    html = requests.get(US_WIKI_URL, headers=HEADERS, timeout=30).text
    tables = pd.read_html(io.BytesIO(html.encode("utf-8")))
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Symbol" in cols:
            t = t.rename(columns={c: str(c).strip() for c in t.columns})
            t["Symbol"] = t["Symbol"].astype(str).str.strip().str.upper()
            t["GICS Sector"] = t["GICS Sector"].astype(str).str.strip()
            # Wikipedia writes BRK.B; Yahoo wants BRK-B
            return [
                {"symbol": str(r["Symbol"]).replace(".", "-"),
                 "name": str(r["Security"]).strip(),
                 "sector": r["GICS Sector"],
                 "sub-sector":r["GICS Sub-Industry"]}
                for r in t.to_dict("records")
            ]
    raise RuntimeError("S&P 500 table not found on the Wikipedia page — layout changed?")


def _india_records() -> list[dict]:
    r = requests.get(IN_NSE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Equity series only (drops rare non-EQ rows); dedupe by symbol.
    df = df[df["Series"] == "EQ"].drop_duplicates(subset="Symbol")
    return [
        {"symbol": str(r["Symbol"]).strip().upper(),
         "name": str(r["Company Name"]).strip(),
         "sector": str(r["Industry"]).strip()}
        for r in df.to_dict("records")
    ]


def refresh_index_list(market: str) -> list[str]:
    """Step 1_Refresh_Index_list for one market.

    Deletes the cached detail file (tickers_*_detail.json) and rebuilds it
    from the upstream source. Only called by
    `pixi run refresh_market_index_list`.
    """
    cfg = MARKETS[market]
    if cfg["detail"].exists():
        cfg["detail"].unlink()
        log.info("[%s] deleted %s", cfg["label"], cfg["detail"].name)

    records = _us_records() if market == "us" else _india_records()
    cfg["detail"].write_text(json.dumps(records, indent=2))
    syms = sorted({r["symbol"] for r in records})
    log.info("[%s] wrote %s (%d records, %d tickers)",
             cfg["label"], cfg["detail"].name, len(records), len(syms))
    return syms


def load_tickers(market: str) -> list[str] | None:
    """Cached index membership for a market (None when the file is missing).

    The symbol list is derived from the detail file, which is the single
    source of truth — there is no separate tickers_*.json any more.
    """
    p = MARKETS[market]["detail"]
    if not p.exists():
        return None
    return sorted({r["symbol"] for r in json.loads(p.read_text())})


def get_details(market: str) -> dict[str, dict]:
    """display symbol -> {name, sector}, from the cached detail file."""
    p = MARKETS[market]["detail"]
    if not p.exists():
        return {}
    return {r["symbol"]: r for r in json.loads(p.read_text())}

def _load_universe(market: str) -> list[str] | None:
    """Read the cached ticker list for one market.

    Step 1_Refresh_Index_list is deliberately NOT run here: `update-data`
    always bypasses it. A missing cache is reported with the command to run.
    """
    cfg = MARKETS[market]
    try:
        tickers = load_tickers(market)
    except Exception as e:  # noqa: BLE001 - report, skip this market, keep going
        log.error("[%s] index list unreadable (%s): %s: %s",
                  cfg["label"], cfg["detail"].name, type(e).__name__, e)
        return None
    if tickers is None:
        log.error("[%s] index list missing (%s) - run `pixi run refresh_market_index_list` first",
                  cfg["label"], cfg["detail"].name)
        return None
    log.info("[%s] universe: %d tickers (cached)", cfg["label"], len(tickers))
    return tickers
# ======================================================================
# STEP 2 Incremental Load data from year 2000
#   Daily OHLCV per ticker symbol from Yahoo Finance, cached as parquet.
#
#   data/last_updated.json holds ONE timestamp for the whole run. A normal
#   run reads it, asks Yahoo only for bars from that date onwards, and
#   appends them to the existing parquet (deduped on date). `--force` is the
#   only mode that re-downloads the full 2000-onward history and overwrites.
# ======================================================================
DEFAULT_START = "2000-01-01"  # full history from 2000 unless --start/--period given
STALE_AFTER_HOURS = 20        # a run younger than this is skipped entirely
OVERLAP_DAYS = 7              # re-fetch this many days before last_updated

LAST_UPDATED_FILE = DATA_DIR / "last_updated.json"


def read_last_updated() -> str | None:
    """Timestamp of the last successful run, or None if it never ran."""
    try:
        raw = json.loads(LAST_UPDATED_FILE.read_text(encoding="utf-8"))
        return raw["last_updated"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def write_last_updated(ts: str | None = None) -> str:
    """Stamp the run as done. One timestamp covers every market."""
    ts = ts or datetime.now().isoformat(timespec="seconds")
    LAST_UPDATED_FILE.write_text(
        json.dumps({"last_updated": ts}, indent=2), encoding="utf-8")
    return ts


def _incremental_start(last_updated: str | None) -> str:
    """First date to ask Yahoo for on an incremental run.

    `last_updated` minus OVERLAP_DAYS, so weekend/holiday gaps, late Yahoo
    corrections and tickers that failed last time all get picked up again.
    Falls back to DEFAULT_START when there is no timestamp yet.
    """
    if not last_updated:
        return DEFAULT_START
    try:
        d = datetime.fromisoformat(last_updated)
    except ValueError:
        return DEFAULT_START
    return (d - timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")


OHLC = ["Open", "High", "Low", "Close"]


def _drop_incomplete_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without a complete OHLC set.

    Yahoo returns a placeholder row for the session that is still trading:
    Volume is populated but Open/High/Low/Close are NaN. Keeping that row
    poisons everything downstream — the scan table's last close becomes NaN
    and the chart endpoint cannot JSON-encode the bar. Dropping it means the
    cache always ends on the last *completed* session.
    """
    cols = [c for c in OHLC if c in df.columns]
    if not cols:
        return df
    out = df.dropna(subset=cols)
    if "Volume" in out.columns:
        out = out.assign(Volume=out["Volume"].fillna(0))
    return out


def _merge_bars(out_path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Append `new` bars to the cached parquet, deduped on date (newest wins)."""
    old = pd.read_parquet(out_path)
    old.index = pd.to_datetime(old.index).tz_localize(None).normalize()
    old.index.name = "date"
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    # A previously cached in-progress bar may still be sitting in `old`.
    return _drop_incomplete_bars(combined)


def _market_age_hours(market: str) -> float | None:
    """Age of the newest cached parquet for one market (None = no cache yet).

    Per-market on purpose: `last_updated.json` is a single global timestamp,
    but running `--usa` then `--india` must not make the second market look
    fresh just because the first one just ran.
    """
    files = list(MARKETS[market]["prices"].glob("*.parquet"))
    if not files:
        return None
    newest = max(f.stat().st_mtime for f in files)
    return (time.time() - newest) / 3600


def _clean_fetch_error(e: Exception) -> str:
    """One-line human-readable error (yfinance 404s arrive as raw JSON)."""
    try:
        desc = (json.loads(str(e)).get("quoteSummary", {})
                .get("error", {}).get("description"))
    except (ValueError, AttributeError):
        desc = None
    return f"HTTP 404: {desc}" if desc else f"{type(e).__name__}: {e}"


def fetch_one(yahoo_symbol: str, out_path: Path,
              start: str | None = None, period: str | None = None,
              interval: str = "1d", merge: bool = True) -> tuple[str, str]:
    """Download one ticker; returns (symbol, status).

    `start` / `period` / `interval` are passed straight to Yahoo. When `start`
    is given it wins over `period` (yfinance ignores period in that case).
    With neither, the full history from DEFAULT_START (2000-01-01) is fetched.

    `merge=True` appends the downloaded bars to the existing parquet (deduped
    on date, newest wins) instead of overwriting it — that is what makes an
    incremental run cheap. `merge=False` (used by --force) overwrites.
    """
    try:
        t = yf.Ticker(yahoo_symbol)
        kwargs = {"interval": interval, "auto_adjust": True}
        if start:
            kwargs["start"] = start
        elif period:
            kwargs["period"] = period
        else:
            kwargs["start"] = DEFAULT_START
        df = t.history(**kwargs)
        if df is None or df.empty:
            # Nothing new since the last run: keep the cache as it is.
            if merge and out_path.exists():
                return yahoo_symbol, "ok"
            return yahoo_symbol, "no data returned (delisted or unknown symbol)"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        # Drop the in-progress session's NaN placeholder row before it is
        # cached (see _drop_incomplete_bars).
        df = _drop_incomplete_bars(df)
        if df.empty:
            if merge and out_path.exists():
                return yahoo_symbol, "ok"
            return yahoo_symbol, "no complete bars returned"
        if merge and out_path.exists():
            df = _merge_bars(out_path, df)
        df.to_parquet(out_path)
        return yahoo_symbol, "ok"
    except Exception as e:  # noqa: BLE001 - log and move on, one bad ticker != whole run dies
        return yahoo_symbol, _clean_fetch_error(e)


def update_prices(market: str, symbols: list[str], force: bool = False,
                  workers: int = 16, start: str | None = None,
                  period: str | None = None, interval: str = "1d",
                  since: str | None = None):
    """Step 2 for one market.

    force=True  -> re-download the full window and OVERWRITE every parquet.
    force=False -> fetch only bars from `since` onwards and MERGE them into
                   the existing parquet. A ticker with no cached file yet
                   still gets a full download from DEFAULT_START.
    """
    cfg = MARKETS[market]
    suffix = cfg["suffix"]

    # Nothing to do when this market's cache is younger than the threshold.
    if not force:
        age_h = _market_age_hours(market)
        if age_h is not None and age_h <= STALE_AFTER_HOURS:
            log.info("[%s] cache is fresh (%.1fh old, threshold %dh) — "
                     "skipping downloads. Use --force to reload anyway.",
                     cfg["label"], age_h, STALE_AFTER_HOURS)
            return {}

    # (symbol, start, period, merge)
    jobs: list[tuple[str, str | None, str | None, bool]] = []
    for s in symbols:
        p = cfg["prices"] / f"{s}.parquet"
        if force or not p.exists():
            jobs.append((s, start, period, False))       # full window, overwrite
        elif start:
            jobs.append((s, start, None, True))          # explicit window, merge
        elif period:
            jobs.append((s, None, period, True))         # explicit period, merge
        else:
            jobs.append((s, since, None, True))          # incremental, merge

    n_full = sum(1 for j in jobs if not j[3])
    # What the incremental jobs actually ask Yahoo for.
    inc_window = start or (f"period={period}" if period else f"start={since}")
    log.info("[%s] %d tickers: %d full, %d incremental (%s)",
             cfg["label"], len(jobs), n_full, len(jobs) - n_full, inc_window)
    if not jobs:
        return {}

    results: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, s + suffix, cfg["prices"] / f"{s}.parquet",
                          st, per, interval, mg): s
                for s, st, per, mg in jobs}
        for fut in as_completed(futs):
            sym = futs[fut]
            _yahoo, status = fut.result()
            done += 1
            if status != "ok":
                results[sym] = status
            if done % 25 == 0 or done == len(jobs):
                log.info("  %d/%d done", done, len(jobs))
    if results:
        bad = cfg["prices"].parent / f"fetch_errors_{market}.json"
        bad.write_text(json.dumps(results, indent=2))
        log.warning("[%s] Step 2: %d ticker(s) failed price download — see %s",
                    cfg["label"], len(results), bad)
    return results
# ======================================================================
# STEP 3 : Adding indicators to each parquet file
#   Reads the cached price parquets, computes the indicators per ticker,
#   and writes the flat scan table the UI displays.
# ======================================================================
def sma(s: pd.Series, n: int) -> float:
    if len(s) < n:
        return np.nan
    return float(s.tail(n).mean())


def ema_last(prices: pd.Series, n: int) -> float:
    return float(prices.ewm(span=n, adjust=False).mean().iloc[-1])


def rsi(prices: pd.Series, n: int = 14) -> float:
    if len(prices) < n + 1:
        return np.nan
    d = prices.diff().dropna()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    # Wilder smoothing (standard RSI)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    al = loss.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    if len(prices) < slow + sig:
        return np.nan, np.nan, np.nan
    ema_f = prices.ewm(span=fast, adjust=False).mean()
    ema_s = prices.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=sig, adjust=False).mean()
    return float(line.iloc[-1]), float(signal.iloc[-1]), float((line - signal).iloc[-1])


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> float:
    if len(close) < n + 1:
        return np.nan
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> float:
    """Wilder's Average Directional Index (0..100) — trend strength, no direction.

    Standard construction: directional movement and true range are smoothed
    with Wilder's alpha (1/n), DI+/DI- give DX, and ADX is the Wilder average
    of DX. Needs ~2n bars before the first value exists.
    """
    if len(close) < 2 * n + 1:
        return np.nan

    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    a = 1 / n
    # Wilder smoothing of TR and DM is an EMA with alpha = 1/n.
    tr_s = tr.ewm(alpha=a, adjust=False).mean()
    plus_s = pd.Series(plus_dm, index=high.index).ewm(alpha=a, adjust=False).mean()
    minus_s = pd.Series(minus_dm, index=high.index).ewm(alpha=a, adjust=False).mean()

    di_plus = 100 * plus_s / tr_s
    di_minus = 100 * minus_s / tr_s
    denom = (di_plus + di_minus).replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / denom

    adx_s = dx.ewm(alpha=a, adjust=False).mean()
    val = adx_s.iloc[-1]
    return float(val) if val == val else np.nan


def rvol(volume: pd.Series, n: int = 20) -> float:
    """Relative volume: today's volume divided by its ``n``-bar simple average.

    1.0 means volume is exactly in line with the recent average, 2.0 means
    twice the average. The average window includes the current bar, matching
    how the chart's ``sma(volume, 20)`` reads.
    """
    if len(volume) < n:
        return np.nan
    avg = float(volume.tail(n).mean())
    if avg <= 0:
        return np.nan
    return float(volume.iloc[-1]) / avg


def obv_last(close: pd.Series, volume: pd.Series, n: int = 14) -> float:
    """OBV slope over the last n bars (normalized per bar) — weak proxy for flow."""
    if len(close) < n + 1:
        return np.nan
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return float(obv.tail(n).diff().mean())


def discount_zone(df: pd.DataFrame) -> float | None:
    """-100..+100: how far the last close sits from the Smart Money discount zone.

    Pulled from ``src.indicators.smart_money`` so the scanner and the chart
    indicator can never disagree about where the zone is.
    """
    try:
        return smart_money.discount_zone(df)
    except Exception as exc:  # an indicator bug must not fail the ticker
        log.debug("discount zone failed: %s", exc)
        return None


def premium_zone(df: pd.DataFrame) -> float | None:
    """-100..+100: how far the last close sits from the Smart Money premium zone.

    Mirror of :func:`discount_zone`: 0 inside the premium zone (the sell area),
    -100 at the swing low, clamped to +100 above the swing high.
    """
    try:
        return smart_money.premium_zone(df)
    except Exception as exc:  # an indicator bug must not fail the ticker
        log.debug("premium zone failed: %s", exc)
        return None

def process_ticker(symbol: str, detail: dict, prices_dir: Path) -> dict | None:
    p = prices_dir / f"{symbol}.parquet"
    if not p.exists():
        return {"symbol": symbol, "error": "no data (fetch failed?)"}
    df = pd.read_parquet(p)
    # Older caches may still hold the in-progress session's NaN row; ignore it
    # so the indicators are computed on completed bars only.
    df = _drop_incomplete_bars(df)
    if len(df) < 60:
        return {"symbol": symbol, "error": f"only {len(df)} bars"}

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    d1 = (last / prev - 1) * 100
    d5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else np.nan
    d21 = (last / float(close.iloc[-22]) - 1) * 100 if len(close) >= 22 else np.nan
    ytd = (last / float(df.iloc[0]["Close"]) - 1) * 100

    ma20, ma50, ma200 = sma(close, 20), sma(close, 50), sma(close, 200)
    r = rsi(close)
    macd_line, macd_sig, macd_hist = macd(close)
    a = atr(high, low, close)
    ob = obv_last(close, vol)
    dz = discount_zone(df)
    pz = premium_zone(df)
    adx14 = adx(high, low, close)
    rv = rvol(vol)

    # 52-week range
    year = df.tail(252)
    hi52, lo52 = float(year["High"].max()), float(year["Low"].min())
    off_52w_high = (last / hi52 - 1) * 100

    support, resistance = float(low.tail(21).min()), float(high.tail(21).max())

    return {
        "symbol": symbol,
        "name": detail.get("name", ""),
        "sector": detail.get("sector", ""),
        "price": round(last, 2),
        "Discount Zone": (round(dz, 1) if dz is not None and dz == dz else None),
        "Premium Zone": (round(pz, 1) if pz is not None and pz == pz else None),
        "ADX": (round(adx14, 1) if adx14 == adx14 else None),
        "RVOL": (round(rv, 2) if rv == rv else None),
        "1D %": round(d1, 2),
        "5D %": round(d5, 2),
        "21D %": round(d21, 2),
        "YTD %": round(ytd, 2),
        "RSI(14)": round(r, 1),
        "MACD hist": round(macd_hist, 4),
        "ATR(14)": round(a, 4),
        "OBV slope": round(ob, 2),
        "SMA20": round(ma20, 2) if ma20 == ma20 else None,
        "SMA50": round(ma50, 2) if ma50 == ma50 else None,
        "SMA200": round(ma200, 2) if ma200 == ma200 else None,
        "52W high": round(hi52, 2),
        "52W low": round(lo52, 2),
        "% off 52W high": round(off_52w_high, 1),
        "21D support": round(support, 2),
        "21D resistance": round(resistance, 2),
        "bars": len(df),
        "last bar": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }

def _fail_reason(symbol: str, row: dict | None, fetch_errors: dict[str, str]) -> str:
    """Label a ticker's failure with the pipeline step where the root cause is."""
    if symbol in fetch_errors:
        return f"Step 2 - Price not found - {symbol}: {fetch_errors[symbol]}"
    err = (row or {}).get("error", "skipped")
    if err.startswith("only ") and "bars" in err:
        return f"Step 3 - Not many bars - {symbol}: {err} (need at least 60)"
    if err == "no data (fetch failed?)":
        return (f"Step 2 - Price not found - {symbol}: "
                f"no price file (earlier download failed)")
    return f"Step 3 - Process failed - {symbol}: {err}"

def build_scan_table(market: str, tickers: list[str],
                     fetch_errors: dict[str, str]) -> None:
    """Step 3 for one market: compute indicators, write the scan table parquet."""
    cfg = MARKETS[market]
    details = get_details(market)
    rows, failed = [], []
    for sym in tickers:
        row = process_ticker(sym, details.get(sym, {}), cfg["prices"])
        if row is None or "error" in row:
            failed.append((sym, _fail_reason(sym, row, fetch_errors)))
        else:
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_parquet(cfg["table"])
    log.info("[%s] wrote %s: %d rows (%d failed/skipped)",
             cfg["label"], cfg["table"].name, len(table), len(failed))
    if failed:
        for _s, why in failed[:15]:
            log.warning("  %s", why)
# ======================================================================
# Orchestration
#   main()          -> Step 2 + Step 3 (update-data). Step 1 is bypassed.
#   refresh_main()  -> Step 1_Refresh_Index_list only.
# ======================================================================

def _resolve_markets(usa: bool, india: bool, market: str | None) -> list[str]:
    """Explicit --usa/--india win, then the legacy --market, else both."""
    if usa or india:
        return [m for m, on in (("us", usa), ("india", india)) if on]
    if market and market != "all":
        return [market]
    return sorted(MARKETS)


def _add_market_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--usa", action="store_true", help="USA market only")
    ap.add_argument("--india", action="store_true", help="India market only")
    # Back-compat: the old --market us|india|all still works.
    ap.add_argument("--market", choices=sorted(MARKETS) + ["all"], default=None,
                    help=argparse.SUPPRESS)


def refresh_main():
    """Step 1_Refresh_Index_list: delete + rebuild the cached index files."""
    ap = argparse.ArgumentParser(
        description="Step 1_Refresh_Index_list - rebuild the index membership "
                    "file (tickers_*_detail.json).",
        epilog="Examples:\n"
               "  pixi run refresh_market_index_list\n"
               "  pixi run refresh_market_index_list --usa\n"
               "  pixi run refresh_market_index_list --india\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_market_args(ap)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    markets = _resolve_markets(args.usa, args.india, args.market)
    scope = ", ".join(MARKETS[m]["label"] for m in markets)

    log.info("")
    log.info("Step 1_Refresh_Index_list : Getting stocks list (%s) %s",
             scope, "-" * 30)
    ok = 0
    for m in markets:
        try:
            refresh_index_list(m)
            ok += 1
        except Exception as e:  # noqa: BLE001 - report, keep going
            log.error("[%s] Step 1_Refresh_Index_list failed: %s: %s",
                      MARKETS[m]["label"], type(e).__name__, e)
    if not ok:
        log.error("Step 1_Refresh_Index_list failed for every market.")
        return
    log.info("")
    log.info("Done - %d/%d market index list(s) refreshed.", ok, len(markets))


def main():
    ap = argparse.ArgumentParser(
        description="Trading data pipeline (prices -> indicators). "
                    "Step 1_Refresh_Index_list is bypassed; run "
                    "`pixi run refresh_market_index_list` to rebuild the index.",
        epilog="Examples:\n"
               "  pixi run update-data --usa\n"
               "  pixi run update-data --india --start 2023-01-01 --period 1d\n"
               "  pixi run update-data --usa --force --interval 1d\n"
               "\n"
               "Default is incremental: bars since data/last_updated.json are\n"
               "appended to the existing parquet files. --force is the only\n"
               "mode that re-downloads the full 2000-onward history.\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--force", action="store_true",
                    help="full reload: re-download from 2000 and OVERWRITE "
                         "every parquet (default is incremental append)")
    _add_market_args(ap)
    # Yahoo download window, passed straight through to yfinance.
    ap.add_argument("--start", default=None,
                    help=f"history start date (default: {DEFAULT_START}); "
                         "overrides --period")
    ap.add_argument("--period", default=None,
                    help="Yahoo period, e.g. 1d, 5d, 1mo, 6mo, 1y, 5y, max "
                         f"(default: full history from {DEFAULT_START})")
    ap.add_argument("--interval", default="1d",
                    help="Yahoo bar interval, e.g. 1d, 1wk, 1mo (default: 1d)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    markets = _resolve_markets(args.usa, args.india, args.market)

    # ---- one timestamp for the whole run -----------------------------------
    last_updated = read_last_updated()
    if args.force:
        since = None
        log.info("Mode: FULL reload (--force) — every ticker re-downloaded")
    else:
        since = _incremental_start(last_updated)
        log.info("Mode: incremental — last updated %s, fetching from %s",
                 last_updated or "never", since)

    # Describe the Yahoo window once so the log shows what was requested.
    if args.force:
        window = (f"start={args.start}" if args.start
                  else f"period={args.period}" if args.period
                  else f"start={DEFAULT_START}")
    else:
        window = f"start={args.start or since} (incremental)"
    log.info("Yahoo window: %s, interval=%s", window, args.interval)

    # ---------------- STEP 1_Refresh_Index_list: BYPASSED ---------------------
    # update-data never rebuilds the index; it only reads the cached files.
    log.info("")
    log.info("Step 1_Refresh_Index_list : bypassed (using cached index list) %s",
             "-" * 10)
    tickers: dict[str, list[str]] = {}
    for m in markets:
        got = _load_universe(m)
        if got is not None:
            tickers[m] = got
    if not tickers:
        log.error("No cached index list for any market - aborting. "
                  "Run `pixi run refresh_market_index_list` first.")
        return

    # ---------------- STEP 2 Incremental Load data from year 2000 -------------
    log.info("")
    log.info("Step 2 Incremental Load data from year 2000 %s", "-" * 30)

    # update_prices() skips a market whose own cache is younger than
    # STALE_AFTER_HOURS, so a quick re-run costs nothing.
    fetch_errors: dict[str, dict[str, str]] = {}
    for m in tickers:
        fetch_errors[m] = update_prices(
            m, tickers[m], force=args.force,
            start=args.start, period=args.period, interval=args.interval,
            since=since,
        )

    # ---------------- STEP 3: indicators + scan table -------------------------
    log.info("")
    log.info("Step 3 : Adding indicators to each parquet file %s", "-" * 30)
    for m in tickers:
        build_scan_table(m, tickers[m], fetch_errors[m])

    # Stamp the run so the next incremental load knows where to resume.
    log.info("")
    log.info("Last updated: %s", write_last_updated())


if __name__ == "__main__":
    # `python src/get_data.py --refresh-index` is the same as the pixi task
    # `refresh_market_index_list`; everything else runs the update pipeline.
    if "--refresh-index" in sys.argv:
        sys.argv.remove("--refresh-index")
        refresh_main()
    else:
        main()
