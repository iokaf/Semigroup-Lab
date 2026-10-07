"""Number formatting and reliability tags.

Complex scalars from the engine arrive as [re, im]; complex arrays as {"re": [...], "im": [...]};
NaN as None and +inf as the string "inf".
"""
from __future__ import annotations

import math
import re


def num(x, p: int = 7) -> str:
    if x is None:
        return "—"
    if x == "inf":
        return "+∞"
    if x == "-inf":
        return "−∞"
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    a = abs(x)
    if 1e-4 <= a < 1e7:
        v = float(f"{x:.{p}g}")          # round to p significant digits, then print without exponent
        s = str(int(v)) if v == int(v) else repr(v)
    else:
        m, e = f"{x:.3e}".split("e")
        s = f"{m}·10^{int(e)}"
    return s.replace("-", "−")


def cnum(c, p: int = 7, tol: float = 1e-13) -> str:
    if c is None:
        return "—"
    if c == "inf":
        return "+∞"
    re_, im_ = c
    if re_ is None or im_ is None:
        return "—"
    scale = max(1.0, abs(re_), abs(im_))
    if abs(im_) < tol * scale:
        return num(re_, p)
    if abs(re_) < tol * scale:
        if im_ == 1:
            return "i"
        if im_ == -1:
            return "−i"
        return num(im_, p) + "i"
    sign = " − " if im_ < 0 else " + "
    return f"{num(re_, p)}{sign}{num(abs(im_), p)}i"


def to_complex(c) -> complex:
    return complex(c[0], c[1])


def tier_color(t: str | None) -> str:
    if not t:
        return "gray"
    s = t.lower()
    if "exact" in s or "theorem" in s or "given" in s:
        return "green"
    if "tier 3" in s or "heuristic" in s or "estimate" in s or "undecided" in s:
        return "orange"
    return "blue"


def tier_label(t: str | None) -> str:
    if not t:
        return ""
    s = t.lower()
    if "theorem" in s:
        return "theorem"
    if "given" in s:
        return "given"
    if "exact" in s:
        return "exact"
    if "tier 3" in s or "estimate" in s:
        return "numerical estimate"
    if "undecided" in s:
        return "undecided"
    if "tier 2" in s:
        return "numerical limit"
    return "numerical"


def badge(t: str | None) -> str:
    """Streamlit markdown badge for a reliability string."""
    if not t:
        return ""
    return f":{tier_color(t)}-badge[{tier_label(t)}]"


_COMPLEX_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)?(?:([+-](?:\d+\.?\d*|\.\d+)?(?:e[+-]?\d+)?)i)?$", re.I)


def parse_complex(text: str) -> complex:
    """Read '0.3+0.4i', '-0.5i', '0.2', '0.1 - 0.7 I' etc."""
    s = (text or "0").replace(" ", "").replace("−", "-").replace("*", "")
    s = re.sub(r"[Ij]", "i", s)
    if not s:
        return 0j
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)?(?:e[+-]?\d+)?i", s, re.I):  # pure imaginary
        v = s[:-1]
        return complex(0, 1 if v in ("", "+") else -1 if v == "-" else float(v))
    m = _COMPLEX_RE.fullmatch(s)
    if not m or (m.group(1) is None and m.group(2) is None):
        raise ValueError(f'Cannot read "{text}" as a complex number; write e.g. 0.3+0.4i.')
    re_ = float(m.group(1)) if m.group(1) else 0.0
    im_ = 0.0
    if m.group(2) is not None:
        g = m.group(2)
        im_ = 1.0 if g == "+" else -1.0 if g == "-" else float(g)
    return complex(re_, im_)


def md_escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
