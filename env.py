"""
Load `.env` into the process environment.

`llm_adjudicator.py` reads `os.environ["TENSORMUX_API_KEY"]` directly, and until
now nothing put anything there — a filled-in `.env` had no effect at all, which
is a confusing failure because the file exists and looks correct. This closes
that gap.

Stdlib only, and about twenty lines, rather than a `python-dotenv` import. The
cascade is deliberately dependency-light so it runs anywhere; adding a package
to read six lines of `KEY=value` would be the wrong trade. It also means the
loader itself can never be the reason a run fails to start.

Never logs a value. Secrets get leaked by helpful diagnostics far more often than
by malice, so `describe()` reports only whether something is set and how long it
is — enough to tell "key missing" from "key wrong", which is the only question
this needs to answer.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name(".env")


def load_env(path: Path | str = DEFAULT_PATH, override: bool = False) -> list[str]:
    """Read `path` into `os.environ`. Returns the names of the keys it set.

    A real environment variable wins over the file unless `override` is set:
    CI and production inject secrets directly, and a stale `.env` on a developer
    machine silently replacing them is a genuinely nasty failure to debug.
    Missing file is not an error — the environment may already be populated.
    """
    path = Path(path)
    if not path.is_file():
        return []

    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and (override or key not in os.environ):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def describe(*names: str) -> str:
    """Report presence and length of each name. Never the value."""
    parts = []
    for name in names:
        value = os.environ.get(name, "")
        parts.append(f"{name}={'set (' + str(len(value)) + ' chars)' if value else 'MISSING'}")
    return "  ".join(parts)


def require(*names: str) -> None:
    """Fail early and legibly when a needed secret is absent.

    Better than the KeyError deep inside a provider SDK, which tends to surface
    as a stack trace that names neither the variable nor the file it belongs in.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            f"missing required environment variable(s): {', '.join(missing)}\n"
            f"add them to {DEFAULT_PATH} (it is gitignored), one per line as KEY=value")
