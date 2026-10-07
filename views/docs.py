"""Documentation page: theory, algorithms, thresholds and the live source of each step.

The text lives in docs.md. Two kinds of HTML comments structure it:
  <!-- section: anchor -->   starts a section (its first "## " line becomes the header)
  <!-- code: key -->         shows the source of the function registered under key in
                              engine/sourcemap.py, read live with inspect
"""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from engine import sourcemap
from ui import compat

DOC_PATH = Path(__file__).resolve().parents[1] / "docs.md"
SECTION_RE = re.compile(r"^<!-- section: ([a-z0-9-]+) -->$")
CODE_RE = re.compile(r"^<!-- code: ([A-Za-z_]+) -->$")
BADGE_RE = re.compile(r":[a-z]+-badge\[([^\]]*)\]")


def parse(text: str):
    """Return (intro_blocks, [(anchor, title, blocks)]); a block is ("md", text) or ("code", key)."""
    intro, sections = [], []
    current = intro
    buf: list[str] = []

    def flush():
        if any(line.strip() for line in buf):
            current.append(("md", "\n".join(buf).strip("\n")))
        buf.clear()

    pending_anchor = None
    for line in text.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            flush()
            pending_anchor = m.group(1)
            continue
        if pending_anchor and line.startswith("## "):
            blocks: list = []
            sections.append((pending_anchor, line[3:].strip(), blocks))
            current = blocks
            pending_anchor = None
            continue
        m = CODE_RE.match(line.strip())
        if m:
            flush()
            current.append(("code", m.group(1)))
            continue
        buf.append(line)
    flush()
    return intro, sections


@st.cache_data(show_spinner=False)
def _source(key: str):
    return sourcemap.resolve(key)


def render_blocks(blocks):
    for kind, value in blocks:
        if kind == "md":
            st.markdown(value)
        else:
            info = _source(value)
            label = (f"Implementation: `{info['qualname']}` ({info['file']}, "
                     f"lines {info['start_line']}–{info['end_line']})")
            with st.expander(label):
                st.code(info["source"], language="python", line_numbers=True)
                st.caption(f"Line 1 above is line {info['start_line']} of {info['file']}.")


def main():
    st.html("<style>.block-container{max-width:56rem} [data-testid='stMarkdownContainer'] p,"
            "[data-testid='stMarkdownContainer'] li{font-family:Spectral,Georgia,serif;font-size:1.06rem;"
            "line-height:1.65} [data-testid='stMarkdownContainer'] table{font-size:.9rem}"
            ".st-key-toc a{text-decoration:none}</style>")
    intro, sections = parse(DOC_PATH.read_text(encoding="utf-8"))
    with st.sidebar:
        with st.container(key="toc"):
            st.markdown("**Contents**")
            st.markdown("\n".join(f"1. [{BADGE_RE.sub('', title).strip()}](#{anchor})"
                                  for anchor, title, _ in sections))
    render_blocks(intro)
    for anchor, title, blocks in sections:
        st.header(title, anchor=anchor, divider="gray")
        render_blocks(blocks)
    # Formulas above a deep-linked section finish rendering after the browser has jumped to it,
    # which pushes the target down; jump again a few times while the page settles.
    compat.run_script("""
      const h = decodeURIComponent(window.location.hash.slice(1));
      if (h) { let n = 0; const t = setInterval(() => {
        const el = window.document.getElementById(h); if (el) el.scrollIntoView();
        if (++n >= 5) clearInterval(t); }, 400); }
    """)


main()
