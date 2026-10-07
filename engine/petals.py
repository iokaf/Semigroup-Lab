"""Backward invariant set W = ∩ phi_t(D) and petals (connected components of int W).

z ∈ phi_t(D)  <=>  the backward flow  dz/ds = -G(z)  from z exists up to time t.
Each grid point is integrated backwards up to T and classified (see
``orbits.classify_backward``):  points that hit the circle in finite time are
*not* in W (their escape time T* is recorded, so {T* > t} = phi_t(D) ∩ grid);
points whose orbits approach a boundary fixed point asymptotically (detected
from the time between the crossings 1-|z| = 1e-5 and 1e-9) belong to a petal.
The output is an *outer approximation* of W: survivors up to T are undecided.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .integrate import disk_margin, disk_margin_rate, integrate
from .orbits import classify_backward

CODES = {0: "escapes in finite time", 1: "converges to a repelling boundary fixed point",
         2: "converges to tau (parabolic petal)", 3: "survives up to T (undecided)",
         4: "Denjoy–Wolff point (fixed)", 5: "numerical failure"}


def _fmt_c(c, tol=1e-9):
    re_, im_ = (0.0 if abs(c.real) < tol else c.real), (0.0 if abs(c.imag) < tol else c.imag)
    if im_ == 0:
        return f"{re_:.6g}"
    if re_ == 0:
        return f"{im_:.6g}i"
    return f"{re_:.6g} {'+' if im_ > 0 else '-'} {abs(im_):.6g}i"


def backward_map(model, boundary_points, n=101, T=60.0, rtol=1e-8):
    n = int(np.clip(n, 21, 201))
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    inside = np.abs(Z) < 1 - 1.0 / n
    pts = Z[inside]
    f = lambda zz: -model.G(zz)  # noqa: E731
    res = integrate(f, pts, [T], margin=disk_margin, eps_levels=(1e-5, 1e-9), rtol=rtol,
                    atol=1e-13, max_iter=30000, max_steps_per_point=6000,
                    margin_rate=disk_margin_rate)
    codes = np.full(pts.shape, 5, dtype=int)
    exit_t = np.full(pts.shape, np.nan)
    landing = np.full(pts.shape, np.nan + 1j * np.nan)
    beta_est = np.full(pts.shape, np.nan)
    is_tau = (np.abs(pts - model.tau) < 1e-12) if not model.tau_on_boundary else np.zeros(pts.size, bool)
    for i in range(pts.size):
        if is_tau[i]:
            codes[i] = 4
            continue
        code, info = classify_backward(model, res.last[i], res.crossings[:, i], bool(res.alive[i]),
                                       res.last[i], bool(res.stalled[i]), boundary_points, T,
                                       t_stop=float(res.exit_time[i]))
        codes[i] = code
        if "exit_time" in info:
            exit_t[i] = info["exit_time"]
        if "landing" in info:
            landing[i] = info["landing"]
        if "beta_est" in info:
            beta_est[i] = info["beta_est"]
    code_grid = np.full(Z.shape, -1, dtype=int)
    code_grid[inside] = codes
    exit_grid = np.full(Z.shape, np.nan)
    exit_grid[inside] = exit_t
    land_grid = np.full(Z.shape, np.nan + 1j * np.nan)
    land_grid[inside] = landing

    # petals = connected components of the points with asymptotic backward orbits
    petal_mask = (code_grid == 1) | (code_grid == 2)
    if model.group:
        petal_mask |= code_grid == 3
    labels, nlab = ndimage.label(petal_mask, structure=np.ones((3, 3)))
    cell_area = (g[1] - g[0]) ** 2
    petals = []
    for lab in range(1, nlab + 1):
        sel = labels == lab
        cnt = int(sel.sum())
        if cnt < 3:
            labels[sel] = 0
            continue
        kinds = code_grid[sel]
        lands = land_grid[sel]
        lands = lands[np.isfinite(lands)]
        target = None
        if lands.size:
            ang = np.round(np.angle(lands) / np.pi, 2)
            vals, counts = np.unique(ang, return_counts=True)
            a = vals[np.argmax(counts)] * np.pi
            target = complex(np.exp(1j * a))
            for bp in boundary_points:
                if abs(bp["sigma"] - target) < 3e-2:
                    target = bp["sigma"]
                    break
        frac_parabolic = float(np.mean(kinds == 2))
        be = beta_est[labels[inside] == lab]
        be = be[np.isfinite(be)]
        petals.append({
            "label": lab,
            "kind": "parabolic petal" if frac_parabolic > 0.5 else "hyperbolic petal",
            "alpha_point": target,
            "area_fraction": float(cnt * cell_area / np.pi),
            "pixels": cnt,
            "beta_estimate_median": float(np.median(be)) if be.size else None,
        })
    counts = {CODES[c]: int((codes == c).sum()) for c in CODES}
    summary = []
    if model.group:
        summary.append("Group of automorphisms: every point has a backward orbit (W = D).")
    elif not petals:
        if counts[CODES[3]] == 0:
            summary.append("No petals found: numerically W has empty interior.")
        else:
            summary.append("No petals identified; some points survive to T (increase T).")
    else:
        for p in petals:
            ap = _fmt_c(p["alpha_point"]) if p["alpha_point"] is not None else "undetermined"
            summary.append(f"A {p['kind']} with α-point {ap}, covering about "
                           f"{100 * p['area_fraction']:.1f}% of D.")
    return {
        "n": n, "T": T, "grid": g, "codes": code_grid, "exit_time": exit_grid,
        "petal_labels": np.where(inside, labels, -1), "petals": petals, "counts": counts,
        "code_names": CODES, "summary": summary,
        "stalled": int(res.stalled.sum()),
    }
