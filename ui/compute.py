"""Cached wrappers around the engine.

Each analysis step is cached separately on (input, its own parameters), so changing a
picture layer, the time slider or a tab never recomputes, and changing one parameter only
recomputes the step that uses it. Results are the JSON-safe dicts produced by
``engine.report``.
"""
from __future__ import annotations

import json

import streamlit as st

from engine import report
from engine.expr import ExpressionError
from engine.model import ModelError

USER_ERRORS = (ExpressionError, ModelError, ValueError, ZeroDivisionError, OverflowError)


def spec_key(spec: dict) -> str:
    return report.spec_key(spec)


def _spec(key: str) -> dict:
    return json.loads(key)


@st.cache_data(show_spinner=False, max_entries=64)
def summary(key: str):
    return report.summary(_spec(key))


@st.cache_data(show_spinner=False, max_entries=32)
def dynamics(key: str, T: float | None):
    return report.dynamics(_spec(key), T=T)


@st.cache_data(show_spinner=False, max_entries=32)
def backward(key: str, n: int):
    return report.backward(_spec(key), n=n)


@st.cache_data(show_spinner=False, max_entries=32)
def koenigs(key: str, T: float | None):
    return report.koenigs_view(_spec(key), T=T)


@st.cache_data(show_spinner=False, max_entries=32)
def asymptotics(key: str, z0_re: float, z0_im: float, T: float | None):
    return report.asymptotics(_spec(key), z0=complex(z0_re, z0_im), T=T)


@st.cache_data(show_spinner=False, max_entries=256)
def orbit(key: str, z0_re: float, z0_im: float, T: float | None):
    return report.point_orbits(_spec(key), complex(z0_re, z0_im), T=T)


def run(fn, *args):
    """Call a cached step; return (result, error message)."""
    try:
        return fn(*args), None
    except USER_ERRORS as exc:
        return None, str(exc)
