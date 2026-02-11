from __future__ import annotations

import re

_PROVISIONAL_RE = re.compile(r"^(?P<year>\d{4})\s+(?P<code>[A-Za-z]{1,3}\d+[A-Za-z0-9]*)$")
_PERIODIC_COMET_PREFIX_RE = re.compile(r"^(?P<num>\d+)(?P<class>[PpDd])$")
_LONG_FORM_COMET_PREFIXES = ("C/", "P/", "D/", "A/", "I/")


def normalize_designation(value: str) -> str:
    """
    Normalize a designation-like string into a stable key used across pipeline joins.

    This is intended to align:
    - truth `designation` strings,
    - SBDB `object_id` strings (which may include names in parentheses),
    - and comet/satellite naming conventions.

    Examples
    --------
    - "(2018 TH19)" -> "2018 TH19"
    - "2025 ME246" -> "2025 ME246"
    - "C/2017 K2 (PANSTARRS)" -> "C/2017 K2"
    - "11P/Tempel-Swift-LINEAR" -> "11P"
    - "104P/Kowal 2" -> "104P"
    - "S/2017 J 11" -> "S/2017 J 11"
    - "Jupiter 7" -> "Jupiter 7"
    - "191305 (2003 HQ20)" -> "191305"
    """
    s = str(value).strip()
    if not s:
        return ""

    # Provisional designations may be stored as "(2019 NY21)".
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        s = s[1:-1].strip()

    # Drop trailing " (NAME)" suffixes such as "(PANSTARRS)" or "(2003 HQ20)".
    # Important: do this after stripping outer parens.
    s = s.split("(", 1)[0].strip()
    if not s:
        return ""

    tokens = s.split()
    if not tokens:
        return ""

    t0 = tokens[0]

    # Periodic comets often appear as "11P/..." or "104P/..." (optionally followed by tokens).
    if "/" in t0:
        prefix = t0.split("/", 1)[0].strip()
        if _PERIODIC_COMET_PREFIX_RE.fullmatch(prefix) is not None:
            return prefix.upper()

    # Long-form comet designations: "C/2017 K2", "P/2019 LD2", etc.
    if any(t0.startswith(p) for p in _LONG_FORM_COMET_PREFIXES):
        if len(tokens) >= 2:
            return f"{t0} {tokens[1]}"
        return t0

    # Natural satellites: "S/2017 J 11" (three tokens).
    if t0.startswith("S/") and len(tokens) >= 3:
        return " ".join(tokens[:3])

    # Planet satellites like "Jupiter 7".
    if len(tokens) >= 2 and tokens[1].isdigit():
        return f"{tokens[0]} {tokens[1]}"

    # Provisional asteroid designations: "2025 ME246" (two tokens).
    if len(tokens) >= 2 and _PROVISIONAL_RE.fullmatch(" ".join(tokens[:2])) is not None:
        return f"{tokens[0]} {tokens[1]}".upper()

    # Default: numbered asteroid or already-normalized key.
    return tokens[0].strip()

