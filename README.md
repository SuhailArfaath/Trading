# Trading — stock scanner (USA + India)

Scans the **USA (S&P 500)** and **India (Nifty 500)** universes and shows
each ticker's key technical indicators in a sortable, searchable table with
two market tabs.

## What the table shows

Per ticker: price, % change over 1/5/21 days and YTD, RSI(14), MACD
histogram, ATR(14), OBV slope, SMA20/50/200, 52-week high/low, % off the
52-week high, and 21-day support/resistance.

## Setup

```bash
pixi install                              # one time
pixi run refresh_market_index_list        # Step 1: rebuild the index files
pixi run update-data                      # both markets, incremental
pixi run update-data --usa                # USA only
pixi run update-data --india              # India only
pixi run update-data --force              # full reload from 2000, overwrite all
pixi run update-data --usa --force        # full reload, USA only
pixi run ui                               # open http://localhost:8500
```

### Step 1_Refresh_Index_list

Index membership is cached in one file per market under `data/`:

```
tickers_us_detail.json     # symbol, name, sector, sub-sector
tickers_in_detail.json
```

Each record carries the symbol plus its metadata, so the symbol list used by
Step 2 is derived from this file — there is no separate `tickers_*.json`.

Only `pixi run refresh_market_index_list` rebuilds them. It **deletes** those
files and re-fetches the membership from upstream (Wikipedia for the S&P 500,
NSE's CSV for the Nifty 500).

```bash
pixi run refresh_market_index_list          # both markets
pixi run refresh_market_index_list --usa    # USA only
pixi run refresh_market_index_list --india  # India only
```

`pixi run update-data` (with any of its arguments) **always bypasses** this
step — it reads the cached files and fails with a hint if they are missing.

### Incremental vs. full reload

`data/last_updated.json` holds **one** timestamp for the whole run:

```json
{ "last_updated": "2026-11-01T17:58:51" }
```

A normal `pixi run update-data` is **incremental**:

1. Read `last_updated` from that file.
2. Ask Yahoo only for bars from `last_updated - 7 days` onwards (the 7-day
   overlap re-covers weekends, holidays, late corrections and tickers that
   failed last time).
3. **Append** the new bars to the existing parquet, deduped on date with the
   newest value winning. The 2000-onward history is never truncated.
4. Rewrite `last_updated.json` with the new timestamp.

A ticker with no cached parquet yet still gets a full download from
`2000-01-01`. If the timestamp is younger than 20 hours the run skips the
downloads entirely and only rebuilds the scan table.

`--force` is the **only** mode that re-downloads the full 2000-onward history
and overwrites every parquet file.

### Download window (passed to Yahoo)

`--start`, `--period` and `--interval` are forwarded to Yahoo Finance. `--start`
wins over `--period` when both are given. With neither, an incremental run
resumes from `last_updated - 7 days` and a `--force` run starts at
**2000-01-01**.

```bash
pixi run update-data --usa --start 2023-01-01        # from a fixed date
pixi run update-data --india --period 5y             # last 5 years
pixi run update-data --usa --period 2y --interval 1wk  # weekly bars
```

| Flag | Meaning | Default |
|---|---|---|
| `--usa` | update the USA market only | — |
| `--india` | update the India market only | — |
| `--start YYYY-MM-DD` | history start date (overrides `--period`) | `last_updated - 7d` |
| `--period` | Yahoo period: `1d`, `5d`, `1mo`, `6mo`, `1y`, `2y`, `5y`, `max` | incremental window |
| `--interval` | bar size: `1d`, `1wk`, `1mo` | `1d` |
| `--force` | full reload from 2000, overwrite every parquet | off |

Passing neither `--usa` nor `--india` updates both markets.

> **Note:** `--start` / `--period` on an incremental run *merge* into the
> existing parquet, so they narrow what is fetched without destroying history.
> Only `--force` overwrites.

## The UI

The frontend is a single-page app (`Trading.html`) built with:

- **React 19** — the entire app: splash screen + data table (no JSX/Babel —
  written with `React.createElement`)
- **Tailwind CSS 4** — styling (official browser build; generates utilities
  in-page, theme tokens + splash keyframes in a `text/tailwindcss` block).
  Splash animations only animate `transform`/`opacity` and only ever scale
  layers *down* (never up), so they stay on the compositor instead of
  repainting each frame. The data grid mounts after the splash hands over —
  building 500 Tabulator rows *during* the zoom is what made it stutter.
- **Tabulator 6** — the data table (UMD global, no bundler)
- **Lightweight Charts 5** — the ticker chart modal
- **talib-web** — TA-Lib compiled to WebAssembly, for the indicators

All of them are cached in `cdn/` and served from `/cdn/...` by the FastAPI
server, so the app never needs the internet. To pull the latest published
versions into `cdn/`, run `pixi run update-cdn` (or
`python src/update_cdn.py --check` to only report versions).

React 19 no longer publishes a UMD bundle, so `update_cdn.py` downloads the
official CommonJS production builds of `react`, `react-dom`, `react-dom/client`
and `scheduler` and wraps them into the `window.React` / `window.ReactDOM`
globals this build-free page expects.

The FastAPI server (`src/api.py`) serves both the HTML page at `/` and the JSON
scan endpoints at `/api/markets/{us,india}/scan`.

The checkboxes beside the USA and India tabs choose which markets an
incremental load updates. At least one market must remain selected; selecting
only one also keeps the UI on that market. The selection and active market are
stored in browser local storage, so they persist across page reloads and server
restarts.

Applied chart indicators, including their parameters and duplicate instances,
are also stored in browser local storage. They remain applied when the chart is
closed, another stock is opened, or the page is reloaded.

### Custom indicators (Python, `src/indicators`)

The chart's **Indicators** dropdown lists every indicator found in
`src/indicators/` under a **Custom** group that always sits at the top of the
list. Picking one adds it to the applied indicators, and the backend recomputes
it over the ticker and interval currently on screen, then shades the chart from
its output.

```
src/indicators/
  __init__.py       registry — auto-discovers the modules below
  smart_money.py    Smart Money Concepts (LuxAlgo port)
  smartmoney.txt    the original Pine Script v5 source
```

Adding an indicator is one file: drop a module in `src/indicators/` that
exposes an `INDICATOR` dict and a `compute(df, params)` function. The registry
picks it up on the next server start — no changes to the API or the frontend.
The `group` key is ignored for the UI grouping: every module found here is
shown under **Custom**.

```python
INDICATOR = {
    "id": "my_indicator", "label": "My Indicator", "group": "Custom",
    "description": "...", "color": "#2157f3",
    "params": {"length": 20},          # defaults used by the UI
}

def compute(df, params):               # df: OHLC, indexed by date
    ...                                # -> JSON-ready overlay payload
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/strategies` | catalogue of every indicator in `src/indicators` |
| `GET /api/strategies/{id}/apply?market=&symbol=&interval=` | run one indicator over one ticker |

`params` may be passed as a JSON object to override the defaults, e.g.
`?params={"swingLength":20}`.

`smart_money.py` is a full port of LuxAlgo's *Smart Money Concepts* Pine
script: premium/equilibrium/discount bands, swing + internal BOS/CHoCH,
order blocks with mitigation, EQH/EQL, and strong/weak high & low levels. It
is a pure function of the OHLC frame, so the same code can be unit-tested,
back-tested, or driven bar-by-bar by a strategy.

## Data source & limits

- Ticker lists: USA from Wikipedia "List of S&P 500 companies"; India from
  NSE's official Nifty 500 constituents CSV (both cached under `data/`).
- Prices: Yahoo Finance via `yfinance`, daily bars from 2000-01-01, cached to
  `data/prices_us/<TICKER>.parquet` (USA) and `data/prices_in/<TICKER>.parquet`
  (India; Yahoo symbols carry a `.NS` suffix). Later runs append only the new
  bars — see [Incremental vs. full reload](#incremental-vs-full-reload).
- Failures per ticker are recorded in `data/fetch_errors_<market>.json`.
- `data/last_updated.json` records when the pipeline last ran.
- India prices are in INR, USA in USD (each tab labels its currency).
- This is an educational scanner, not financial advice. Yahoo data can lag or
  have gaps; verify anything you actually trade.

## Files

```
pixi.toml            project + tasks (pixi is the only package manager needed)
Trading.html         UI — React 18 + Tailwind 4 single-page app (served at /)
cdn/                 static assets served at /cdn: cached frontend libs (react,
                     react-dom, tailwind, tabulator, lightweight-charts,
                     talib-web) plus the app logo (cdn/logo.png, served at /logo.png)
src/get_data.py      data pipeline: market config, ticker lists (S&P 500 + Nifty 500), price download + indicators
src/api.py           FastAPI backend — serves JSON API + Trading.html
src/run_ui.py        launcher: kills anything on port 8500, then starts src/api.py
src/indicators/      Python indicators (auto-discovered, listed under "Custom" in the indicator menu)
src/update_cdn.py    refreshes cdn/ from unpkg (latest published versions)
data/                cache: prices_us/, prices_in/, scan_table*.parquet,
                     tickers*_detail.json, last_updated.json
```
