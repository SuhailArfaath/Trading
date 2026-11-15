"""Indicator registry.

Every ``*.py`` module in this folder that exposes both an ``INDICATOR`` dict
and a ``compute(df, params)`` function is auto-discovered and offered to the
UI's **Indicators** menu under its "Custom" group. Dropping a new file in here
is all it takes to add an indicator — no edits to the API or the frontend.

A module's contract
-------------------
``INDICATOR``
    ``{"id", "label", "group", "description", "color", "palette", "params"}``
    — the metadata the dropdown renders and the defaults it applies.

``compute(df, params) -> dict``
    ``df`` is an OHLC DataFrame indexed by date (Open/High/Low/Close columns).
    The return value must be JSON-serialisable and follow the overlay payload
    shape documented in :mod:`src.indicators.smart_money`.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent

#: id -> {"meta": INDICATOR dict, "compute": callable}
_REGISTRY: dict[str, dict[str, Any]] = {}


def _discover() -> None:
    """Import every sibling module and register the ones that follow the contract."""
    for info in pkgutil.iter_modules([str(_PKG_DIR)]):
        name = info.name
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except Exception:  # a broken indicator must not take the app down
            continue
        meta = getattr(mod, "INDICATOR", None)
        fn = getattr(mod, "compute", None)
        if isinstance(meta, dict) and callable(fn) and meta.get("id"):
            _REGISTRY[str(meta["id"])] = {"meta": meta, "compute": fn}


_discover()


def catalogue() -> list[dict[str, Any]]:
    """Every registered indicator's metadata, sorted by group then label."""
    items = [dict(v["meta"]) for v in _REGISTRY.values()]
    items.sort(key=lambda m: (str(m.get("group", "")), str(m.get("label", ""))))
    return items


def get(indicator_id: str) -> dict[str, Any] | None:
    """One registry entry (``{"meta", "compute"}``) or None."""
    return _REGISTRY.get(indicator_id)


def compute(indicator_id: str, df: pd.DataFrame,
            params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Run one indicator over ``df``. None when the id is unknown."""
    entry = _REGISTRY.get(indicator_id)
    if entry is None:
        return None
    fn: Callable[..., dict[str, Any]] = entry["compute"]
    return fn(df, params or {})


__all__ = ["catalogue", "get", "compute"]
