"""Refresh the locally-cached frontend libraries in /cdn from unpkg / jsDelivr.

The UI (Trading.html) loads every third-party library from /cdn/<file>, served
by the FastAPI app, so the page never needs the internet at runtime. This script
is the only thing that does: it resolves the latest published version of each
package on the npm registry, downloads the browser-ready build, and writes it
into /cdn under the stable filename the HTML expects.

React 19 no longer publishes a UMD bundle (only CommonJS + ESM), and this page is
deliberately build-free (no bundler, no JSX). So for React/ReactDOM the script
downloads the official CJS production builds and wraps them in a small IIFE +
module registry, producing the same `window.React` / `window.ReactDOM` globals
the old UMD builds provided.

Usage:
    python src/update_cdn.py            # update everything
    python src/update_cdn.py --check    # report versions, download nothing

After updating, bump the ?v= query strings in Trading.html so browsers drop
their cached copy. The script prints the exact values to use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CDN = ROOT / "cdn"
REGISTRY = "https://registry.npmjs.org"
UNPKG = "https://unpkg.com"
JSDELIVR = "https://cdn.jsdelivr.net/npm"

# Plain single-file assets: local filename -> (npm package, path, version selector).
# A selector of "latest" means the newest published version; "18" means the
# newest 18.x release; "^1.2.3" means the newest release satisfying that caret
# range. React/ReactDOM are handled separately below (they need bundling).
ASSETS: dict[str, tuple[str, str, str]] = {
    "tailwind.browser.js": ("@tailwindcss/browser", "dist/index.global.js", "latest"),
    "tabulator.min.js": ("tabulator-tables", "dist/js/tabulator.min.js", "latest"),
    "tabulator.min.css": ("tabulator-tables", "dist/css/tabulator.min.css", "latest"),
    "lightweight-charts.standalone.production.js": (
        "lightweight-charts",
        "dist/lightweight-charts.standalone.production.js",
        "latest",
    ),
    "talib-web.umd.js": ("talib-web", "lib/index.umd.js", "latest"),
    "talib.wasm": ("talib-web", "lib/talib.wasm", "latest"),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Trading-cdn-updater"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _get_text(url: str) -> str:
    return _get(url).decode("utf-8")


def _strip_sourcemap(text: str) -> str:
    """Drop a trailing //# sourceMappingURL comment (it would swallow appended code)."""
    return re.sub(r"\n?//# sourceMappingURL=\S+\s*$", "\n", text)


def _fetch_min(package: str, version: str, path: str) -> str:
    """Fetch a CJS file, preferring jsDelivr's minified copy over the raw unpkg one."""
    min_path = re.sub(r"\.js$", ".min.js", path)
    for url in (
        f"{JSDELIVR}/{package}@{version}/{min_path}",
        f"{UNPKG}/{package}@{version}/{path}",
    ):
        try:
            return _strip_sourcemap(_get_text(url))
        except Exception:
            continue
    raise RuntimeError(f"could not fetch {package}@{version}/{path}")


def _patch(filename: str, data: bytes) -> bytes:
    """Repair known packaging bugs in upstream builds.

    talib-web@0.1.3 ships lib/index.umd.js with a literal backslash-n after the
    final `};` (i.e. `};\\n})`), which is a syntax error in the browser. Replace
    it with a real newline so the UMD wrapper parses.
    """
    if filename == "talib-web.umd.js":
        text = data.decode("utf-8")
        fixed = re.sub(r"};\s*\\n\}\)\s*$", "};\n})\n", text)
        if fixed != text:
            print(f"  patched trailing literal \\n in {filename}")
            return fixed.encode("utf-8")
    return data


_doc_cache: dict[str, dict] = {}


def _registry_doc(package: str) -> dict:
    if package not in _doc_cache:
        # Scoped names must be escaped: @scope/name -> @scope%2fname
        _doc_cache[package] = json.loads(
            _get(f"{REGISTRY}/{package.replace('/', '%2f')}")
        )
    return _doc_cache[package]


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("-")[0].split("."))


def _satisfies_caret(version: str, base: str) -> bool:
    """Minimal `^base` matcher (enough for the ranges npm packages declare)."""
    v, b = _parse(version), _parse(base)
    if v < b:
        return False
    if b[0] > 0:
        return v[0] == b[0]
    if len(b) > 1 and b[1] > 0:
        return v[0] == 0 and v[1] == b[1]
    return v[:2] == b[:2]


def resolve_version(package: str, selector: str) -> str:
    """Return the concrete version to download for `package`."""
    doc = _registry_doc(package)
    if selector == "latest":
        return doc["dist-tags"]["latest"]

    if selector.startswith("^"):
        base = selector[1:]
        candidates = [
            v for v in doc["versions"] if "-" not in v and _satisfies_caret(v, base)
        ]
        if not candidates:
            raise SystemExit(f"no published version of {package} matches {selector!r}")
        return max(candidates, key=_parse)

    # Newest release matching a major (or major.minor) prefix.
    pattern = re.compile(rf"^{re.escape(selector)}(\.\d+)*$")
    candidates = [v for v in doc["versions"] if pattern.match(v) and "-" not in v]
    if not candidates:
        raise SystemExit(f"no published version of {package} matches {selector!r}")
    return max(candidates, key=_parse)


# ------------------------------------------------------------- react bundles


def _react_bundle(version: str) -> str:
    """Wrap react's CJS production build into a `window.React` global."""
    code = _fetch_min("react", version, "cjs/react.production.js")
    return (
        f"/*! React v{version} | (c) Meta Platforms, Inc. and affiliates. | MIT\n"
        " *  Browser global assembled from the official CommonJS build: React 19\n"
        " *  no longer ships a UMD bundle, so the CJS module is wrapped in an IIFE.\n"
        f" *  Source: {UNPKG}/react@{version}/cjs/react.production.js\n"
        " */\n"
        "(function (global) {\n"
        "  var module = { exports: {} };\n"
        "  var exports = module.exports;\n"
        + code
        + "\n  global.React = module.exports;\n"
        "})(typeof globalThis !== 'undefined' ? globalThis : this);\n"
    )


def _react_dom_bundle(dom_version: str, scheduler_version: str) -> str:
    """Bundle react-dom + react-dom/client + scheduler into a `window.ReactDOM` global."""
    sched = _fetch_min("scheduler", scheduler_version, "cjs/scheduler.production.js")
    dom = _fetch_min("react-dom", dom_version, "cjs/react-dom.production.js")
    client = _fetch_min("react-dom", dom_version, "cjs/react-dom-client.production.js")
    return (
        f"/*! ReactDOM v{dom_version} | (c) Meta Platforms, Inc. and affiliates. | MIT\n"
        " *  Browser global assembled from the official CommonJS builds (react-dom,\n"
        " *  react-dom/client, scheduler). React 19 ships no UMD bundle.\n"
        f" *  Sources: {UNPKG}/react-dom@{dom_version}/cjs/\n"
        " */\n"
        "(function (global) {\n"
        "  var registry = {};\n"
        "  var cache = {};\n"
        "  function require(name) {\n"
        "    if (Object.prototype.hasOwnProperty.call(cache, name)) return cache[name].exports;\n"
        "    var factory = registry[name];\n"
        "    if (!factory) throw new Error('Cannot find module ' + name);\n"
        "    var module = { exports: {} };\n"
        "    cache[name] = module;\n"
        "    factory.call(module.exports, module, module.exports, require);\n"
        "    return module.exports;\n"
        "  }\n"
        "  registry['react'] = function (module) { module.exports = global.React; };\n"
        "  registry['scheduler'] = function (module, exports, require) {\n"
        + sched
        + "\n  };\n"
        "  registry['react-dom'] = function (module, exports, require) {\n"
        + dom
        + "\n  };\n"
        "  registry['react-dom-client'] = function (module, exports, require) {\n"
        + client
        + "\n  };\n"
        "  var out = {};\n"
        "  var domExports = require('react-dom');\n"
        "  var clientExports = require('react-dom-client');\n"
        "  for (var k in domExports) out[k] = domExports[k];\n"
        "  for (var k in clientExports) out[k] = clientExports[k];\n"
        "  global.ReactDOM = out;\n"
        "})(typeof globalThis !== 'undefined' ? globalThis : this);\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="only report the versions that would be downloaded",
    )
    args = ap.parse_args()

    CDN.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    failures: list[str] = []

    # --- React: latest, assembled from CJS because 19 has no UMD build.
    try:
        react_version = resolve_version("react", "latest")
        dom_version = resolve_version("react-dom", "latest")
        dom_doc = _registry_doc("react-dom")
        sched_range = dom_doc["versions"][dom_version]["dependencies"]["scheduler"]
        scheduler_version = resolve_version("scheduler", sched_range)
    except Exception as exc:
        print(f"FAILED to resolve React versions: {exc}", file=sys.stderr)
        return 1

    resolved["react"] = react_version
    resolved["react-dom"] = dom_version
    resolved["scheduler"] = scheduler_version

    if args.check:
        print(f"{'react.umd.min.js':<45} react@{react_version} (CJS -> global)")
        print(f"{'react-dom.umd.min.js':<45} react-dom@{dom_version} (CJS -> global)")
        print(f"{'':<45} scheduler@{scheduler_version} (bundled)")

    # --- Plain assets.
    for filename, (package, inner, selector) in ASSETS.items():
        try:
            version = resolve_version(package, selector)
        except Exception as exc:  # network / registry problem
            failures.append(f"{package}: {exc}")
            continue

        resolved[package] = version
        url = f"{UNPKG}/{package}@{version}/{inner}"
        dest = CDN / filename

        if args.check:
            print(f"{filename:<45} {package}@{version}")
            continue

        try:
            data = _get(url)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue

        data = _patch(filename, data)
        dest.write_bytes(data)
        print(f"{filename:<45} {package}@{version}  {len(data):>10,} bytes")

    # --- React bundles.
    if not args.check:
        try:
            react_js = _react_bundle(react_version).encode("utf-8")
            (CDN / "react.umd.min.js").write_bytes(react_js)
            print(
                f"{'react.umd.min.js':<45} react@{react_version}  "
                f"{len(react_js):>10,} bytes"
            )

            dom_js = _react_dom_bundle(dom_version, scheduler_version).encode("utf-8")
            (CDN / "react-dom.umd.min.js").write_bytes(dom_js)
            print(
                f"{'react-dom.umd.min.js':<45} react-dom@{dom_version}  "
                f"{len(dom_js):>10,} bytes"
            )
        except Exception as exc:
            failures.append(f"react bundle: {exc}")

    if not args.check:
        print("\nBump these query strings in Trading.html:")
        for package, version in resolved.items():
            print(f"  {package} -> ?v={version}")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
