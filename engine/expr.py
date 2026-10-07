"""Safe parsing and compilation of user-supplied holomorphic expressions.

Input strings are checked against a whitelist of identifiers and characters
*before* they reach SymPy (whose parser uses ``eval``), so arbitrary Python
cannot be smuggled in.  Decimal literals are converted to exact rationals so
that ``0.5`` is treated as ``1/2`` by the exact engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import mpmath as mp
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

z, t, w, u = sp.symbols("z t w u")

FUNCTIONS = {
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh,
}
CONSTANTS = {"I": sp.I, "i": sp.I, "j": sp.I, "pi": sp.pi, "E": sp.E, "e": sp.E}

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^(). ,\t]*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
MAX_LEN = 600


class ExpressionError(ValueError):
    """Raised for input that cannot be parsed or is not allowed."""


def parse(text: str, variables: tuple[str, ...] = ("z",)) -> sp.Expr:
    if text is None or not str(text).strip():
        raise ExpressionError("Empty expression.")
    text = str(text).strip()
    if len(text) > MAX_LEN:
        raise ExpressionError(f"Expression longer than {MAX_LEN} characters.")
    if not _ALLOWED_CHARS.match(text):
        bad = sorted(set(c for c in text if not _ALLOWED_CHARS.match(c)))
        raise ExpressionError(f"Characters not allowed: {' '.join(bad)}")
    if "__" in text:
        raise ExpressionError("Double underscores are not allowed.")
    allowed = set(FUNCTIONS) | set(CONSTANTS) | set(variables)
    for name in _IDENT.findall(text):
        if name not in allowed:
            raise ExpressionError(
                f"Unknown name '{name}'. Allowed variables: {', '.join(variables)}; "
                f"functions: {', '.join(sorted(FUNCTIONS))}; constants: I, pi, E."
            )
    local = {**FUNCTIONS, **CONSTANTS, **{v: sp.Symbol(v) for v in variables}}
    glob = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
            "Symbol": sp.Symbol, "Function": sp.Function, "Mul": sp.Mul,
            "Add": sp.Add, "Pow": sp.Pow}
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        # First an unevaluated parse, so sizes can be checked before SymPy evaluates anything
        # (9^9^9^9 or (1+z)^100000 would otherwise hang the server).
        raw = parse_expr(text, local_dict=local, global_dict=glob, transformations=transformations,
                         evaluate=False)
    except Exception as exc:  # noqa: BLE001 - surface parser errors to the user
        raise ExpressionError(f"Could not parse '{text}': {exc}") from None
    check_size(raw)
    try:
        expr = parse_expr(text, local_dict=local, global_dict=glob, transformations=transformations,
                          evaluate=True)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"Could not parse '{text}': {exc}") from None
    if not isinstance(expr, sp.Expr):
        raise ExpressionError("Input is not a scalar expression.")
    floats = expr.atoms(sp.Float)
    if floats:
        expr = expr.xreplace({f: sp.Rational(str(f)) for f in floats})
    extra = {s.name for s in expr.free_symbols} - set(variables)
    if extra:
        raise ExpressionError(f"Unexpected symbols: {', '.join(sorted(extra))}")
    return expr


MAX_EXPONENT = 100
MAX_INTEGER_DIGITS = 50


def check_size(raw) -> None:
    """Reject numeric exponents above MAX_EXPONENT and integers longer than MAX_INTEGER_DIGITS.

    ``raw`` is the unevaluated parse tree. It is walked children-first, so the exponent of every
    power has itself been checked before it is evaluated here.
    """
    if not isinstance(raw, sp.Basic):
        return
    for node in sp.postorder_traversal(raw):
        if isinstance(node, sp.Integer) and len(str(abs(int(node)))) > MAX_INTEGER_DIGITS:
            raise ExpressionError(f"Integers may have at most {MAX_INTEGER_DIGITS} digits.")
        if isinstance(node, sp.Pow) and not node.exp.free_symbols:
            try:
                size = abs(complex(sp.N(node.exp, 15)))
            except (TypeError, ValueError, OverflowError):
                size = float("inf")
            if not size <= MAX_EXPONENT:
                raise ExpressionError(
                    f"Numeric exponents must be at most {MAX_EXPONENT} in absolute value "
                    "(rational generators are limited to degree 40 anyway).")


def is_rational_in(expr: sp.Expr, var: sp.Symbol) -> bool:
    try:
        return bool(expr.is_rational_function(var))
    except Exception:  # noqa: BLE001
        return False


def cancel_rational_parts(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
    """Cancel every rational-in-``var`` subexpression (bottom-up).

    After substituting a Möbius change of variables this removes the
    catastrophic cancellation ``1 - z`` with ``z ~ 1`` (it becomes ``2/(w+1)``).
    """
    if is_rational_in(expr, var):
        return sp.cancel(sp.together(expr))
    if expr.is_Atom:
        return expr
    args = [cancel_rational_parts(a, var) for a in expr.args]
    return expr.func(*args)


@dataclass
class Compiled:
    expr: sp.Expr
    var: sp.Symbol
    np_fn: object
    mp_fn: object

    def __call__(self, x):
        x = np.asarray(x, dtype=complex)
        with np.errstate(all="ignore"):
            out = self.np_fn(x)
        out = np.asarray(out, dtype=complex)
        if out.shape != x.shape:
            out = np.broadcast_to(out, x.shape).copy()
        return out

    def mp(self, x, dps: int = 40):
        with mp.workdps(dps):
            return mp.mpc(self.mp_fn(mp.mpc(x)))


def compile_expr(expr: sp.Expr, var: sp.Symbol) -> Compiled:
    np_fn = sp.lambdify(var, expr, modules=["numpy"])
    mp_fn = sp.lambdify(var, expr, modules=["mpmath"])
    return Compiled(expr=expr, var=var, np_fn=np_fn, mp_fn=mp_fn)


def to_complex(value) -> complex:
    return complex(sp.N(value, 30))


def latex(expr) -> str:
    try:
        return sp.latex(expr)
    except Exception:  # noqa: BLE001
        return str(expr)
