"""Smart Money Concepts — Python port of LuxAlgo's Pine Script v5 indicator.

Ported from ``src/indicators/smartmoney.txt``
("Smart Money Concepts [LuxAlgo]", CC BY-NC-SA 4.0, (c) LuxAlgo).

The module is a pure function of an OHLC DataFrame: it returns a JSON-ready
overlay payload that the FastAPI backend serves and the Lightweight Charts
frontend paints. Nothing here touches the network or the filesystem, so the
same code can be unit-tested, back-tested, or driven bar-by-bar by a
NautilusTrader strategy.

What it produces
----------------
* premium / equilibrium / discount bands (trailing swing range)
* swing + internal BOS / CHoCH structure breaks
* swing + internal order blocks (with high/low mitigation)
* equal highs / equal lows (EQH / EQL)
* strong / weak high & low trailing levels

Pivot rule (LuxAlgo's ``leg()``): the bar ``i - size`` is confirmed as a pivot
high at bar ``i`` when ``high[i-size] > highest(high, size)`` over the window
``(i-size, i]``; a pivot low when ``low[i-size] < lowest(low, size)``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (mirrors the Pine script's BULLISH / BEARISH)
# ---------------------------------------------------------------------------
BULLISH = 1
BEARISH = -1

#: Default parameters, matching the Pine script's input defaults.
DEFAULTS: dict[str, Any] = {
    "swingLength": 50,      # swing structure pivot length
    "internalLength": 5,    # internal structure pivot length
    "equalLength": 3,       # EQH/EQL confirmation bars
    "equalThreshold": 0.1,  # EQH/EQL sensitivity (fraction of ATR)
    "atrLength": 200,       # volatility measure (OB filter + EQH/EQL)
    "maxInternalOB": 5,     # internal order blocks kept on screen
    "maxSwingOB": 0,        # swing order blocks kept on screen (0 = off, Pine default)
}

#: Metadata consumed by the indicator registry (shown under "Custom" in the UI).
INDICATOR: dict[str, Any] = {
    "id": "smart_money",
    "label": "Smart Money Concepts",
    "group": "Smart Money",
    "description": (
        "LuxAlgo Smart Money Concepts: BOS/CHoCH structure, order blocks, "
        "EQH/EQL, premium/discount zones and strong/weak levels."
    ),
    "color": "#2157f3",
    "palette": {
        "premium": "#5b2d8e",
        "equilibrium": "#878b94",
        "discount": "#2157f3",
    },
    "params": dict(DEFAULTS),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fmt_time(ts: Any) -> str:
    """Bar timestamp -> the 'YYYY-MM-DD' string Lightweight Charts expects."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         length: float) -> np.ndarray:
    """Wilder ATR (Pine's ``ta.atr``).

    Pine returns ``na`` during the warm-up, which would suppress EQH/EQL on
    short windows (a 1Y chart is ~260 bars, so ATR(200) leaves only 60). The
    expanding mean of true range is used until the full window is available so
    the zones still appear on shorter timeframes.
    """
    n = len(high)
    tr = np.zeros(n, dtype=float)
    for i in range(n):
        h, l = high[i], low[i]
        if not (np.isfinite(h) and np.isfinite(l)):
            tr[i] = 0.0
            continue
        if i == 0:
            tr[i] = h - l
            continue
        pc = close[i - 1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))

    atr = np.full(n, np.nan, dtype=float)
    length = max(1, int(round(length)))
    acc = 0.0
    rma: float | None = None
    for i in range(n):
        if i < length:
            acc += tr[i]
            atr[i] = acc / (i + 1)
            if i == length - 1:
                rma = acc / length
        else:
            assert rma is not None
            rma = (rma * (length - 1) + tr[i]) / length
            atr[i] = rma
    return atr


def _parsed_hl(high: np.ndarray, low: np.ndarray,
               atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volatility filter -> parsed high/low (LuxAlgo's order block filter).

    A bar whose range is at least twice the volatility measure is "high
    volatility"; its high and low are swapped so order blocks anchor on the
    meaningful edge of the candle.
    """
    n = len(high)
    parsed_high = np.empty(n, dtype=float)
    parsed_low = np.empty(n, dtype=float)
    for i in range(n):
        hv = bool(np.isfinite(atr[i])) and (high[i] - low[i]) >= 2 * atr[i]
        parsed_high[i] = low[i] if hv else high[i]
        parsed_low[i] = high[i] if hv else low[i]
    return parsed_high, parsed_low


def _find_pivots(high: np.ndarray, low: np.ndarray, size: int,
                 cache: dict[int, tuple[np.ndarray, np.ndarray]]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Confirm pivots for one lookback ``size`` (memoised per size).

    ``high_at[i]`` is True when bar ``i-size`` is a confirmed pivot high at
    bar ``i``; ``low_at[i]`` likewise for a pivot low.
    """
    if size in cache:
        return cache[size]

    n = len(high)
    high_at = np.zeros(n, dtype=bool)
    low_at = np.zeros(n, dtype=bool)
    for i in range(size, n):
        p = i - size
        hi_max = -math.inf
        lo_min = math.inf
        for j in range(p + 1, i + 1):
            if high[j] > hi_max:
                hi_max = high[j]
            if low[j] < lo_min:
                lo_min = low[j]
        if high[p] > hi_max:
            high_at[i] = True
        if low[p] < lo_min:
            low_at[i] = True

    cache[size] = (high_at, low_at)
    return high_at, low_at


# ---------------------------------------------------------------------------
# Structure pass (swing or internal)
# ---------------------------------------------------------------------------
def _run_structure(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   times: list[str], parsed_high: np.ndarray,
                   parsed_low: np.ndarray, size: int, internal: bool,
                   swing_levels: dict[str, list[float | None]] | None,
                   cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """One structure pass: pivots, BOS/CHoCH breaks and order blocks."""
    n = len(high)
    high_at, low_at = _find_pivots(high, low, size, cache)

    leg = 0
    p_high: float | None = None
    p_high_time: str | None = None
    p_high_idx: int | None = None
    p_high_crossed = False
    p_low: float | None = None
    p_low_time: str | None = None
    p_low_idx: int | None = None
    p_low_crossed = False
    bias = 0

    breaks: list[dict[str, Any]] = []
    obs: list[dict[str, Any]] = []
    levels_high: list[float | None] = [None] * n
    levels_low: list[float | None] = [None] * n

    for i in range(n):
        # ---- mitigate existing order blocks (deleteOrderBlocks)
        for k in range(len(obs) - 1, -1, -1):
            ob = obs[k]
            if ob["bias"] == BEARISH and high[i] > ob["top"]:
                obs.pop(k)
            elif ob["bias"] == BULLISH and low[i] < ob["bottom"]:
                obs.pop(k)

        # ---- leg / pivot update (getCurrentStructure)
        new_leg = leg
        if high_at[i]:
            new_leg = 0          # BEARISH_LEG -> pivot high
        elif low_at[i]:
            new_leg = 1          # BULLISH_LEG -> pivot low
        changed = new_leg != leg
        leg = new_leg

        if changed and new_leg == 1:
            p = i - size
            p_low = float(low[p])
            p_low_time = times[p]
            p_low_idx = p
            p_low_crossed = False
        if changed and new_leg == 0:
            p = i - size
            p_high = float(high[p])
            p_high_time = times[p]
            p_high_idx = p
            p_high_crossed = False

        levels_high[i] = p_high
        levels_low[i] = p_low

        # ---- bullish break: close crosses above the last pivot high
        if (p_high is not None and not p_high_crossed and i > 0
                and close[i] > p_high and close[i - 1] <= p_high):
            extra = (swing_levels["high"][i] != p_high) if internal else True
            if extra:
                tag = "CHoCH" if bias == BEARISH else "BOS"
                p_high_crossed = True
                bias = BULLISH
                breaks.append({
                    "time": times[i], "fromTime": p_high_time,
                    "price": round(p_high, 4), "tag": tag,
                    "bias": BULLISH, "internal": internal,
                })
                # order block: lowest parsed low between the pivot and this bar
                best, best_val = -1, math.inf
                assert p_high_idx is not None
                for j in range(p_high_idx, i):
                    if parsed_low[j] < best_val:
                        best_val = float(parsed_low[j])
                        best = j
                if best >= 0:
                    obs.insert(0, {
                        "fromTime": times[best],
                        "top": round(float(parsed_high[best]), 4),
                        "bottom": round(float(parsed_low[best]), 4),
                        "bias": BULLISH, "internal": internal,
                    })

        # ---- bearish break: close crosses below the last pivot low
        if (p_low is not None and not p_low_crossed and i > 0
                and close[i] < p_low and close[i - 1] >= p_low):
            extra = (swing_levels["low"][i] != p_low) if internal else True
            if extra:
                tag = "CHoCH" if bias == BULLISH else "BOS"
                p_low_crossed = True
                bias = BEARISH
                breaks.append({
                    "time": times[i], "fromTime": p_low_time,
                    "price": round(p_low, 4), "tag": tag,
                    "bias": BEARISH, "internal": internal,
                })
                best, best_val = -1, -math.inf
                assert p_low_idx is not None
                for j in range(p_low_idx, i):
                    if parsed_high[j] > best_val:
                        best_val = float(parsed_high[j])
                        best = j
                if best >= 0:
                    obs.insert(0, {
                        "fromTime": times[best],
                        "top": round(float(parsed_high[best]), 4),
                        "bottom": round(float(parsed_low[best]), 4),
                        "bias": BEARISH, "internal": internal,
                    })

    return {
        "breaks": breaks,
        "obs": obs,
        "levelsHigh": levels_high,
        "levelsLow": levels_low,
        "bias": bias,
    }


# ---------------------------------------------------------------------------
# Equal highs / equal lows
# ---------------------------------------------------------------------------
def _equal_hl(high: np.ndarray, low: np.ndarray, times: list[str],
              atr: np.ndarray, size: int, threshold: float,
              cache: dict[int, tuple[np.ndarray, np.ndarray]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect EQH / EQL: consecutive pivots within ``threshold * ATR``."""
    n = len(high)
    high_at, low_at = _find_pivots(high, low, size, cache)

    eq_leg = 0
    eq_high: dict[str, Any] | None = None
    eq_low: dict[str, Any] | None = None
    equal_highs: list[dict[str, Any]] = []
    equal_lows: list[dict[str, Any]] = []

    for i in range(n):
        new_leg = eq_leg
        if high_at[i]:
            new_leg = 0
        elif low_at[i]:
            new_leg = 1
        changed = new_leg != eq_leg
        eq_leg = new_leg
        p = i - size

        if changed and new_leg == 1:
            if (eq_low is not None and np.isfinite(atr[i])
                    and abs(eq_low["level"] - low[p]) < threshold * atr[i]):
                equal_lows.append({
                    "fromTime": eq_low["time"], "toTime": times[p],
                    "price": round(float(low[p]), 4), "tag": "EQL",
                })
            eq_low = {"level": float(low[p]), "time": times[p]}

        if changed and new_leg == 0:
            if (eq_high is not None and np.isfinite(atr[i])
                    and abs(eq_high["level"] - high[p]) < threshold * atr[i]):
                equal_highs.append({
                    "fromTime": eq_high["time"], "toTime": times[p],
                    "price": round(float(high[p]), 4), "tag": "EQH",
                })
            eq_high = {"level": float(high[p]), "time": times[p]}

    return equal_highs, equal_lows


# ---------------------------------------------------------------------------
# Trailing extremes -> premium / equilibrium / discount bands
# ---------------------------------------------------------------------------
def _bands(high: np.ndarray, low: np.ndarray, times: list[str],
           swing_size: int,
           cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any] | None:
    """Trailing swing range -> the three shaded zones.

    LuxAlgo draws each zone as ONE box spanning from the pivot bar to the last
    bar, so only the final values matter. The payload therefore carries
    single-element series, which keeps the response small and still matches
    what ``makeBandPrimitive`` expects on the frontend.

    The trailing extremes are *not* reset by every pivot. Pine's ``leg()``
    keeps a stateful leg value that only flips when a bar makes a new leg
    high/low, and ``startOfNewLeg()`` is ``ta.change(leg) != 0`` — so pivots
    strictly alternate high/low/high/low. A run of consecutive pivot lows
    therefore does not keep lowering ``bottom``; only the first low after a
    bearish leg counts. Resetting on every pivot independently collapses the
    range (e.g. AAPL 1Y bottom 309.40 instead of 245.07).
    """
    n = len(high)
    high_at, low_at = _find_pivots(high, low, swing_size, cache)

    # Pine's trailingExtremes.new() leaves top/bottom as na, and
    # math.max(high, na) stays na, so each side only starts tracking once its
    # first pivot of that direction has been seen.
    top: float | None = None
    bottom: float | None = None
    anchor: str | None = None
    last_top_time: str | None = None
    last_bottom_time: str | None = None
    leg = 0  # BEARISH_LEG

    for i in range(n):
        # updateTrailingExtremes(): ratchet, and refresh the time on ties
        # because Pine compares `trailing.top == high` after the max().
        if top is not None and np.isfinite(high[i]):
            if high[i] > top:
                top = float(high[i])
            if top == high[i]:
                last_top_time = times[i]
        if bottom is not None and np.isfinite(low[i]):
            if low[i] < bottom:
                bottom = float(low[i])
            if bottom == low[i]:
                last_bottom_time = times[i]

        # leg(): a new leg high wins over a new leg low, and the leg value only
        # changes when it actually flips.
        new_leg = leg
        if high_at[i]:
            new_leg = 0
        elif low_at[i]:
            new_leg = 1
        if new_leg == leg:
            continue

        # getCurrentStructure(): reset the extreme that the new leg starts from.
        pivot_time = times[i - swing_size]
        if new_leg == 1:  # start of bullish leg -> pivot low
            bottom = float(low[i - swing_size])
            last_bottom_time = pivot_time
        else:             # start of bearish leg -> pivot high
            top = float(high[i - swing_size])
            last_top_time = pivot_time
        anchor = pivot_time
        leg = new_leg

    if top is None or bottom is None or not (top > bottom):
        return None

    t = times[-1]
    return {
        "premiumTop": [{"time": t, "value": round(top, 4)}],
        "premiumBottom": [{"time": t, "value": round(0.95 * top + 0.05 * bottom, 4)}],
        "eqTop": [{"time": t, "value": round(0.525 * top + 0.475 * bottom, 4)}],
        "eqBottom": [{"time": t, "value": round(0.525 * bottom + 0.475 * top, 4)}],
        "discountTop": [{"time": t, "value": round(0.95 * bottom + 0.05 * top, 4)}],
        "discountBottom": [{"time": t, "value": round(bottom, 4)}],
        "anchorTime": [anchor if anchor is not None else t],
        "top": top,
        "bottom": bottom,
        "topTime": last_top_time,
        "bottomTime": last_bottom_time,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full Smart Money Concepts pass over one OHLC DataFrame.

    ``df`` must be indexed by date and carry Open/High/Low/Close columns.
    Returns a JSON-serialisable overlay payload.
    """
    p = {**DEFAULTS, **(params or {})}

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return {
            "bars": 0, "lastTime": None,
            "premiumTop": [], "premiumBottom": [],
            "eqTop": [], "eqBottom": [],
            "discountTop": [], "discountBottom": [],
            "anchorTime": [], "structure": [],
            "equalHighs": [], "equalLows": [], "orderBlocks": [],
            "strongWeak": {"top": None, "bottom": None, "topTime": None,
                           "bottomTime": None, "topLabel": "Weak High",
                           "bottomLabel": "Weak Low"},
        }

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    times = [_fmt_time(t) for t in df.index]
    n = len(high)

    atr = _atr(high, low, close, p["atrLength"])
    parsed_high, parsed_low = _parsed_hl(high, low, atr)

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    swing_size = max(2, int(round(p["swingLength"])))
    internal_size = max(2, int(round(p["internalLength"])))

    swing = _run_structure(high, low, close, times, parsed_high, parsed_low,
                           swing_size, False, None, cache)
    internal = _run_structure(high, low, close, times, parsed_high, parsed_low,
                              internal_size, True,
                              {"high": swing["levelsHigh"],
                               "low": swing["levelsLow"]}, cache)

    eq_size = max(1, int(round(p["equalLength"])))
    equal_highs, equal_lows = _equal_hl(
        high, low, times, atr, eq_size, float(p["equalThreshold"]), cache)

    bands = _bands(high, low, times, swing_size, cache) or {}

    structure = sorted(swing["breaks"] + internal["breaks"],
                       key=lambda b: b["time"])[-200:]

    order_blocks = (
        internal["obs"][: max(1, int(round(p["maxInternalOB"])))]
        + swing["obs"][: max(0, int(round(p["maxSwingOB"])))]
    )

    return {
        "bars": n,
        "lastTime": times[-1],
        "premiumTop": bands.get("premiumTop", []),
        "premiumBottom": bands.get("premiumBottom", []),
        "eqTop": bands.get("eqTop", []),
        "eqBottom": bands.get("eqBottom", []),
        "discountTop": bands.get("discountTop", []),
        "discountBottom": bands.get("discountBottom", []),
        "anchorTime": bands.get("anchorTime", []),
        "structure": structure,
        "equalHighs": equal_highs[-40:],
        "equalLows": equal_lows[-40:],
        "orderBlocks": order_blocks,
        "strongWeak": {
            "top": bands.get("top"),
            "bottom": bands.get("bottom"),
            "topTime": bands.get("topTime"),
            "bottomTime": bands.get("bottomTime"),
            "topLabel": "Strong High" if swing["bias"] == BEARISH else "Weak High",
            "bottomLabel": "Strong Low" if swing["bias"] == BULLISH else "Weak Low",
        },
    }


# ---------------------------------------------------------------------------
# Scanner helper: how far a price sits from the discount zone
# ---------------------------------------------------------------------------
def discount_zone_score(price: float, top: float, bottom: float) -> float | None:
    """Distance of ``price`` from the discount zone, as a -100..+100 score.

    The discount zone spans ``bottom`` .. ``0.95*bottom + 0.05*top`` (see
    :func:`_bands`), so it covers the lowest 5% of the swing range. The score
    is anchored on the middle of that zone and scaled by the height of the
    swing above it:

    * ``0``   -> the price is inside the discount zone (the buy area)
    * ``+100`` -> the price is at the swing high, i.e. deep in the premium zone
    * ``-100`` -> the price is below the swing low

    Values are clamped to ±100 so the column stays a bounded oscillator.
    Returns None when the range is degenerate (no usable swing).
    """
    if not (top > bottom) or not math.isfinite(price):
        return None
    zone_top = 0.95 * bottom + 0.05 * top
    ref = 0.5 * (zone_top + bottom)   # centre of the discount zone
    unit = top - ref                  # swing height above the zone
    if unit <= 0:
        return None
    return max(-100.0, min(100.0, 100.0 * (price - ref) / unit))


def discount_zone(df: pd.DataFrame,
                  params: dict[str, Any] | None = None) -> float | None:
    """Convenience wrapper: run the band pass on ``df`` and score its last close.

    Used by the scan-table builder (Step 3), which only needs the current
    value. Returns None when there is not enough history for a swing pivot.
    """
    p = {**DEFAULTS, **(params or {})}
    df = df.dropna(subset=["High", "Low"])
    if len(df) < 2:
        return None

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    times = [_fmt_time(t) for t in df.index]
    swing_size = max(2, int(round(p["swingLength"])))

    bands = _bands(high, low, times, swing_size, {})
    if not bands:
        return None
    return discount_zone_score(float(df["Close"].iloc[-1]),
                               bands["top"], bands["bottom"])


# ---------------------------------------------------------------------------
# Scanner helper: how far a price sits from the premium zone
# ---------------------------------------------------------------------------
def premium_zone_score(price: float, top: float, bottom: float) -> float | None:
    """Distance of ``price`` from the premium zone, as a -100..+100 score.

    The premium zone spans ``0.95*top + 0.05*bottom`` .. ``top`` (see
    :func:`_bands`), so it covers the highest 5% of the swing range. This is
    the mirror of :func:`discount_zone_score`: same unit (the swing height
    below the zone), opposite anchor and sign.

    * ``0``    -> the price is inside the premium zone (the sell area)
    * ``-100`` -> the price is at the swing low, i.e. deep in the discount zone
    * ``> 0``  -> the price is above the premium zone (only possible on a
      breakout above the swing high; ``+100`` needs a full swing range of
      headroom, exactly as ``-100`` needs one below the zone)

    Values are clamped to ±100 so the column stays a bounded oscillator.
    Returns None when the range is degenerate (no usable swing).
    """
    if not (top > bottom) or not math.isfinite(price):
        return None
    zone_bottom = 0.95 * top + 0.05 * bottom
    ref = 0.5 * (zone_bottom + top)   # centre of the premium zone
    unit = ref - bottom               # swing height below the zone
    if unit <= 0:
        return None
    return max(-100.0, min(100.0, 100.0 * (price - ref) / unit))


def premium_zone(df: pd.DataFrame,
                 params: dict[str, Any] | None = None) -> float | None:
    """Convenience wrapper: run the band pass on ``df`` and score its last close.

    Mirror of :func:`discount_zone`, used by the scan-table builder (Step 3).
    Returns None when there is not enough history for a swing pivot.
    """
    p = {**DEFAULTS, **(params or {})}
    df = df.dropna(subset=["High", "Low"])
    if len(df) < 2:
        return None

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    times = [_fmt_time(t) for t in df.index]
    swing_size = max(2, int(round(p["swingLength"])))

    bands = _bands(high, low, times, swing_size, {})
    if not bands:
        return None
    return premium_zone_score(float(df["Close"].iloc[-1]),
                              bands["top"], bands["bottom"])
