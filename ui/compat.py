"""Small shims so the app runs without errors or deprecation warnings on Streamlit 1.46 and later.

Streamlit renamed and added layout arguments across releases:

* ``st.plotly_chart`` gained ``width=`` in 1.52. Before that it collected unknown keyword
  arguments and warned "The keyword arguments have been deprecated ... Use `config` instead",
  and from 1.52 on ``use_container_width=`` is itself deprecated.
* ``st.form_submit_button`` and ``st.download_button`` gained ``width=`` in 1.48.
* ``st.html`` gained ``unsafe_allow_javascript=`` in 1.52 (``components.html`` is deprecated later).

Each helper inspects the installed Streamlit once and passes only arguments it supports.
"""
from __future__ import annotations

import inspect

import streamlit as st


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_PLOTLY_WIDTH = _accepts(st.plotly_chart, "width")
_SUBMIT_WIDTH = _accepts(st.form_submit_button, "width")
_DOWNLOAD_WIDTH = _accepts(st.download_button, "width")
_HTML_JS = _accepts(st.html, "unsafe_allow_javascript")


def _full_width(kwargs: dict, has_width: bool) -> dict:
    kwargs = dict(kwargs)
    if has_width:
        kwargs.setdefault("width", "stretch")
    else:
        kwargs.setdefault("use_container_width", True)
    return kwargs


def plotly_chart(fig, container=None, **kwargs):
    """Full-width Plotly chart; ``container`` lets you draw into a column (``col.plotly_chart``)."""
    target = container if container is not None else st
    return target.plotly_chart(fig, **_full_width(kwargs, _PLOTLY_WIDTH))


def form_submit_button(label: str, **kwargs):
    return st.form_submit_button(label, **_full_width(kwargs, _SUBMIT_WIDTH))


def download_button(label: str, **kwargs):
    if _DOWNLOAD_WIDTH:
        kwargs.setdefault("width", "content")
    return st.download_button(label, **kwargs)


def run_script(js: str) -> None:
    """Run a small trusted script in the page (never pass user input here)."""
    if _HTML_JS:
        st.html(f"<script>{js}</script>", unsafe_allow_javascript=True)
    else:  # Streamlit < 1.52: the script runs in an iframe and reaches the page through window.parent
        import streamlit.components.v1 as components

        components.html(f"<script>const window_ = window.parent;\n{js.replace('window.', 'window_.')}"
                        "</script>", height=0)
