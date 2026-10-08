"""Backward invariant set W = ∩ φ_t(D) and its petals.

Every grid point is classified by ``backward.classify`` (escape, petal of a repelling point σ,
parabolic petal, undecided); see that module for the method. Then:

* Petals are grouped by their α-point, the point the backward orbits converge to, not by which
  pixels touch: every repelling boundary fixed point is the α-point of exactly one hyperbolic
  petal, and the parabolic petal (at most one) consists of the backward orbits converging to τ.
  Two petals separated by a curve thinner than the grid therefore stay apart.
* The grid reaches the unit circle (cells with |z| < 1 − 10⁻⁷), and areas are shares of the grid
  cells that lie in D, so they are not biased by an unsampled ring.
* The outcome is cross-checked against theorems, and every failed check is reported.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import backward as B

CODES = {
    B.ESCAPE: "escapes in finite time",
    "petal": "converges to a repelling boundary fixed point",
    B.TAU: "converges to τ (parabolic petal)",
    B.UNDECIDED: "undecided",
    B.FIXED: "Denjoy–Wolff point (fixed)",
    B.IN_W: "in W (group of rotations)",
    B.FAILED: "numerical failure",
}


def code_name(label: int) -> str:
    return CODES["petal"] if label >= 0 else CODES.get(label, "?")


def _fmt_c(c, tol=1e-9):
    re_, im_ = (0.0 if abs(c.real) < tol else c.real), (0.0 if abs(c.imag) < tol else c.imag)
    if im_ == 0:
        return f"{re_:.6g}"
    if re_ == 0:
        return f"{im_:.6g}i"
    return f"{re_:.6g} {'+' if im_ > 0 else '-'} {abs(im_):.6g}i"


def backward_map(model, boundary_points, n=101, step_kind=None):
    """Classify a grid of points; return labels, petals, escape times and theorem checks."""
    n = int(np.clip(n, 21, 201))
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    inside = np.abs(Z) < 1 - 1e-7
    pts = Z[inside]
    chart = B.Chart(model, boundary_points)
    lab = np.full(pts.shape, B.UNDECIDED, dtype=int)
    exit_t = np.full(pts.shape, np.nan)
    beta_est = np.full(pts.shape, np.nan)
    t_max = 0.0
    if model.kind == "elliptic" and model.group:
        lab[:] = B.IN_W                       # rotations: every point has a backward orbit
    else:
        is_tau = (np.abs(pts - model.tau) < 1e-12) if model.kind == "elliptic" else np.zeros(pts.size, bool)
        res = B.classify(chart, pts[~is_tau], parabolic_allowed=(model.kind == "parabolic"))
        lab[~is_tau] = res["label"]
        exit_t[~is_tau] = res["exit_time"]
        beta_est[~is_tau] = res["beta_est"]
        lab[is_tau] = B.FIXED
        t_max = res["t_max"]
    labels = np.full(Z.shape, -9, dtype=int)
    labels[inside] = lab
    exit_grid = np.full(Z.shape, np.nan)
    exit_grid[inside] = exit_t
    n_in = int(inside.sum())

    petals = []
    for k, (sig, beta) in enumerate(zip(chart.sigmas, chart.betas)):
        cells = labels == k
        cnt = int(cells.sum())
        comps = _components(cells)
        be = beta_est[lab == k]
        be = be[np.isfinite(be)]
        petals.append({"key": k, "kind": "hyperbolic petal", "alpha_point": sig, "beta": beta,
                       "cells": cnt, "area_fraction": cnt / n_in, "components": comps,
                       "beta_estimate_median": float(np.median(be)) if be.size else None})
    cells = labels == B.TAU
    if cells.any():
        petals.append({"key": "tau", "kind": "parabolic petal", "alpha_point": model.tau, "beta": None,
                       "cells": int(cells.sum()), "area_fraction": float(cells.sum()) / n_in,
                       "components": _components(cells), "beta_estimate_median": None})
    if model.kind == "elliptic" and model.group:
        petals = [{"key": "W", "kind": "whole disc (group of rotations)", "alpha_point": None, "beta": None,
                   "cells": n_in, "area_fraction": 1.0, "components": 1, "beta_estimate_median": None}]
    counts = {}
    for l in np.unique(lab):
        name = code_name(int(l))
        counts[name] = counts.get(name, 0) + int((lab == l).sum())
    checks = petal_checks(model, petals, counts, n_in, step_kind)
    summary = _summary(model, petals)
    return {"n": n, "grid": g, "labels": labels, "exit_time": exit_grid, "petals": petals,
            "counts": counts, "checks": checks, "summary": summary, "t_max": t_max,
            "time_unit": 20.0 / chart.typical_speed(),
            "capture_radii": chart.r_cap}


def _components(cells):
    labelled, num = ndimage.label(cells, structure=np.ones((3, 3)))
    if num == 0:
        return 0
    sizes = ndimage.sum(cells, labelled, range(1, num + 1))
    return int((np.asarray(sizes) >= 3).sum())


def _summary(model, petals):
    if model.kind == "elliptic" and model.group:
        return ["Group of rotations: every point has a backward orbit, W = D."]
    real = [p for p in petals if p["cells"] > 0]
    if not real:
        return ["No petals: numerically W has empty interior."]
    out = []
    for p in real:
        ap = _fmt_c(p["alpha_point"]) if p["alpha_point"] is not None else "undetermined"
        out.append(f"A {p['kind']} with α-point {ap}, covering about {100 * p['area_fraction']:.1f}% of D.")
    if model.group:
        out.insert(0, "Group of automorphisms: every point has a backward orbit, W = D.")
    return out


def petal_checks(model, petals, counts, n_in, step_kind=None):
    """Compare the computed petals with what the theory requires. Each entry: {ok, text}."""
    checks = []
    if model.kind == "elliptic" and model.group:
        return [{"ok": True, "text": "Group of rotations: W = D, as it must be."}]
    for p in petals:
        if p["kind"] != "hyperbolic petal":
            continue
        s = _fmt_c(p["alpha_point"])
        if p["cells"] == 0:
            checks.append({"ok": False, "text": (
                f"No grid point converged to the repelling point σ = {s}, yet every repelling point is "
                "the α-point of exactly one petal: that petal is thinner than the grid here. "
                "Raise the resolution.")})
            continue
        checks.append({"ok": True, "text": f"The repelling point σ = {s} has its petal, as it must."})
        if p["components"] > 1:
            checks.append({"ok": False, "text": (
                f"The petal of σ = {s} appears in {p['components']} pieces. Petals are connected, so the "
                "pieces are joined by strips thinner than the grid.")})
        be = p["beta_estimate_median"]
        if be is not None and p["beta"]:
            rel = abs(be - p["beta"]) / p["beta"]
            checks.append({"ok": rel < 0.1, "text": (
                f"Rate of the backward orbits into σ = {s}: {be:.5g}, against β(σ) = {p['beta']:.5g} "
                f"({100 * rel:.2g}% apart){'' if rel < 0.1 else '; expected to agree'}.")})
    has_tau = any(p["kind"] == "parabolic petal" and p["cells"] for p in petals)
    if model.kind == "parabolic":
        if step_kind == "positive":
            checks.append({"ok": has_tau, "text": (
                "Positive hyperbolic step and a parabolic petal was found, as the theory requires."
                if has_tau else
                "Positive hyperbolic step, so a parabolic petal must exist, but none was found on the grid.")})
        elif step_kind == "zero":
            checks.append({"ok": not has_tau, "text": (
                "Zero hyperbolic step and no parabolic petal, as the theory requires." if not has_tau else
                "A parabolic petal was found although the step is zero; this contradicts the theory.")})
        else:
            checks.append({"ok": True, "text": (
                "Hyperbolic step undecided, so the parabolic-petal check (a parabolic petal exists exactly "
                "for positive step) was not applied.")})
    bad = counts.get(CODES[B.UNDECIDED], 0) + counts.get(CODES[B.FAILED], 0)
    if bad:
        share = bad / n_in
        checks.append({"ok": share < 0.01, "text": (
            f"{bad} grid points ({100 * share:.2g}%) could not be classified "
            "(undecided or numerical failure).")})
    return checks


def backward_orbit(model, z0, boundary_points):
    """Backward orbit of one point with its classification (for the point inspector)."""
    chart = B.Chart(model, boundary_points)
    if model.kind == "elliptic" and model.group:
        from .integrate import disk_margin, integrate
        T = 2 * np.pi / max(abs(model.lam.imag), 1e-9)
        tt = np.linspace(0, T, 400)
        r = integrate(lambda w: -model.G(w), np.array([z0]), tt, rtol=1e-10, margin=disk_margin,
                      eps_levels=(1e-14,))
        return {"t": tt, "z": r.y_out[:, 0], "code": B.IN_W, "code_name": code_name(B.IN_W),
                "reason": "group of rotations: the backward orbit is periodic"}
    if model.kind == "elliptic" and abs(z0 - model.tau) < 1e-12:
        return {"t": np.array([0.0]), "z": np.array([z0]), "code": B.FIXED, "code_name": code_name(B.FIXED)}
    pts = np.array([z0])
    res = B.classify(chart, pts, parabolic_allowed=(model.kind == "parabolic"), record_path=True, rtol=1e-10)
    lab = int(res["label"][0])
    out = {"t": res["path_t"][0], "z": res["path_z"][0], "code": lab, "code_name": code_name(lab)}
    if lab >= 0:
        out["sigma"] = chart.sigmas[lab]
        out["landing"] = chart.sigmas[lab]
        out["beta_expected"] = chart.betas[lab]
        if np.isfinite(res["beta_est"][0]):
            out["beta_est"] = float(res["beta_est"][0])
        out["reason"] = "enters the capture ball of σ, so it converges to σ"
    elif lab == B.ESCAPE:
        out["exit_time"] = float(res["exit_time"][0])
        out["landing"] = complex(res["exit_point"][0])
        out["reason"] = "reaches the unit circle in finite time"
    elif lab == B.TAU:
        out["landing"] = model.tau
        out["reason"] = "runs to τ with Re w tending to a positive limit (parabolic petal)"
    elif lab == B.UNDECIDED:
        out["reason"] = f"still unclassified at t = {res['t_max']:.4g}"
    else:
        out["reason"] = "numerical failure (step budget or step size)"
    zz = out["z"]
    with np.errstate(all="ignore"):
        rho = np.abs(zz - z0) / np.abs(1 - np.conj(z0) * zz)
    out["k_from_start"] = np.arctanh(np.clip(rho, 0, 1 - 1e-16))
    return out
