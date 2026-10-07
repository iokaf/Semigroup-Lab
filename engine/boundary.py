"""Boundary null points of the generator (boundary fixed points of the semigroup).

For a boundary point sigma with radial limit G(sigma) = 0 the quantity

    beta(sigma) = angular-lim  G(z) / (z - sigma)

is real.  beta <= 0 only at the Denjoy–Wolff point (beta = -lambda);
0 < beta < inf  : repelling (regular) boundary fixed point, phi_t'(sigma) = e^{beta t};
beta = +inf     : non-regular ("super-repelling") boundary fixed point.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from .expr import is_rational_in, z


def poly_roots(poly_expr, dps=40):
    """Numerical roots (``dps`` digits) of a polynomial in z with numeric coefficients."""
    import sympy as sp

    P = sp.Poly(sp.expand(poly_expr), z)
    if P.degree() <= 0:
        return []
    coeffs = [complex(sp.N(c, dps)) for c in P.all_coeffs()]
    with mp.workdps(dps):
        mpc = [mp.mpc(sp.N(sp.re(c), dps), sp.N(sp.im(c), dps)) for c in P.all_coeffs()]
        try:
            roots = mp.polyroots(mpc, maxsteps=400, extraprec=200)
        except mp.libmp.NoConvergence:
            roots = [mp.mpc(r) for r in np.roots(coeffs)]
    return [mp.mpc(r) for r in roots]


def radial_beta(G_mp, sigma, ks=range(3, 10), dps=50):
    """Estimate beta(sigma) from the radial sequence z_k = (1-10^-k) sigma."""
    vals = []
    with mp.workdps(dps):
        sig = mp.mpc(sigma)
        for k in ks:
            s = mp.mpf(10) ** (-k)
            zz = (1 - s) * sig
            try:
                vals.append(complex(G_mp(zz) / (zz - sig)))
            except Exception:  # noqa: BLE001
                vals.append(complex("nan"))
    vals = np.array(vals)
    if not np.all(np.isfinite(vals)):
        return {"beta": None, "err": None, "infinite": False, "sequence": vals.tolist(), "ok": False}
    mags = np.abs(vals)
    growing = len(mags) >= 4 and np.all(mags[-3:] / np.maximum(mags[-4:-1], 1e-300) > 1.5) and mags[-1] > 1e2
    if growing:
        return {"beta": float("inf"), "err": None, "infinite": True, "sequence": vals.tolist(), "ok": True}
    est = vals[-1]
    err = float(np.abs(vals[-1] - vals[-2]))
    return {"beta": est, "err": err, "infinite": False, "sequence": vals.tolist(), "ok": True}


def beta_via_halfplane(G_expr, sigma_mp, dps=60):
    """beta(sigma) = -lim_{x->+inf} Gamma_sigma(x)/x, where Gamma_sigma is G conjugated by
    w = (sigma + z)/(sigma - z).  Rational subexpressions are cancelled symbolically, so the
    evaluation at huge x does not suffer from the cancellation 1 - conj(sigma) z ~ 0."""
    import sympy as sp

    from .expr import cancel_rational_parts, w

    with mp.workdps(45):
        sig = sp.Float(mp.nstr(mp.re(sigma_mp), 40), 40) + sp.I * sp.Float(mp.nstr(mp.im(sigma_mp), 40), 40)
    expr = (w + 1) ** 2 / (2 * sig) * G_expr.subs(z, sig * (w - 1) / (w + 1))
    expr = cancel_rational_parts(expr, w)
    f = sp.lambdify(w, expr, "mpmath")
    ks = list(range(2, 33, 2))
    vals = []
    with mp.workdps(dps):
        for k in ks:
            x = mp.mpf(10) ** k
            try:
                vals.append(complex(-f(x) / x))
            except Exception:  # noqa: BLE001
                vals.append(complex("nan"))
    vals = np.array(vals)
    finite = np.isfinite(vals)
    if finite.sum() < 4:
        return {"beta": None, "err": None, "infinite": False, "sequence": vals.tolist(), "ok": False}
    v = vals[finite]
    mags = np.abs(v)
    if np.all(mags[-3:] / np.maximum(mags[-4:-1], 1e-300) > 1.5) and mags[-1] > 1e2:
        return {"beta": float("inf"), "err": None, "infinite": True, "sequence": vals.tolist(), "ok": True}
    return {"beta": v[-1], "err": float(abs(v[-1] - v[-2])), "infinite": False,
            "sequence": vals.tolist(), "ok": True}


def _refine_on_circle(G, theta_lo, theta_hi, r=1 - 1e-12, iters=80):
    gr = (np.sqrt(5) - 1) / 2
    a, b = theta_lo, theta_hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    f = lambda th: abs(G(np.array([r * np.exp(1j * th)]))[0])  # noqa: E731
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    th = (a + b) / 2
    return th, f(th)


def _refine_mp(G_mp, sigma0, dps=50):
    """Refine a boundary zero to high precision (Newton if G is analytic there)."""
    with mp.workdps(dps):
        try:
            root = mp.findroot(G_mp, mp.mpc(sigma0), tol=mp.mpf(10) ** (-dps + 10), maxsteps=60)
            if abs(root - mp.mpc(sigma0)) < 1e-4 and abs(abs(root) - 1) < mp.mpf(10) ** (-20):
                return root / abs(root), "Newton (high precision)"
        except Exception:  # noqa: BLE001
            pass
        # golden-section minimisation of |G| on the circle r = 1 - 1e-30
        th0 = mp.arg(mp.mpc(sigma0))
        a, b = th0 - mp.mpf("1e-6"), th0 + mp.mpf("1e-6")
        r = 1 - mp.mpf(10) ** (-30)
        f = lambda th: abs(G_mp(r * mp.expj(th)))  # noqa: E731
        gr = (mp.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(110):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = f(d)
        return mp.expj((a + b) / 2), "golden-section minimisation of |G| on the circle"


def find_boundary_null_points(model, n_angles=4096):
    """Return list of dicts {sigma, method} (unclassified)."""
    pts = []
    if model.rational:
        for r in model.numerator_roots:
            if abs(abs(r) - 1) < mp.mpf(10) ** (-15):
                if model.denominator_vanishes(r):
                    continue
                pts.append({"sigma": complex(r / abs(r)), "sigma_mp": r / abs(r),
                            "method": "root of the numerator of G (40 digits)"})
        return _dedupe(pts)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    rb = 1 - 1e-10
    vals = np.abs(model.G(rb * np.exp(1j * theta)))
    vals = np.where(np.isfinite(vals), vals, np.inf)
    med = np.median(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 1.0
    med = max(med, 1e-300)
    prev, nxt = np.roll(vals, 1), np.roll(vals, -1)
    cand = np.nonzero((vals <= prev) & (vals <= nxt) & np.isfinite(vals))[0]
    # keep the deepest candidates
    cand = cand[np.argsort(vals[cand])][:60]
    step = 2 * np.pi / n_angles
    for i in cand:
        th, val = _refine_on_circle(model.G, theta[i] - step, theta[i] + step)
        if not np.isfinite(val) or val > 1e-4 * med:
            continue
        # radial test: |G| must tend to zero along the radius
        s = np.array([1e-4, 1e-8, 1e-12])
        radial = np.abs(model.G((1 - s) * np.exp(1j * th)))
        if not (np.all(np.isfinite(radial)) and radial[-1] < radial[0] and radial[-1] < 1e-3 * med):
            continue
        sig_mp, how = _refine_mp(model.G.mp_fn, np.exp(1j * th))
        pts.append({"sigma": complex(sig_mp), "sigma_mp": sig_mp,
                    "method": f"circle scan + {how}"})
    return _dedupe(pts)


def _dedupe(pts, tol=1e-7):
    out = []
    for p in pts:
        if all(abs(p["sigma"] - q["sigma"]) > tol for q in out):
            out.append(p)
    return out


def classify_boundary_points(model, pts):
    res = []
    for p in pts:
        sig = p["sigma"]
        beta = None
        exact = None
        if model.rational and not model.denominator_vanishes(p["sigma_mp"]):
            with mp.workdps(50):
                b = complex(model.dG.mp(p["sigma_mp"], dps=50))
            info = {"beta": b, "err": 1e-25, "infinite": False, "ok": True,
                    "method": "beta = G'(sigma), G analytic at sigma"}
            exact = model.exact_value_at(sig, kind="dG")
        else:
            info = beta_via_halfplane(model.G_expr, p["sigma_mp"])
            info["method"] = "-lim Gamma_sigma(x)/x in the half-plane chart at sigma, x = 10^2..10^32"
            if not info["ok"]:
                info = radial_beta(model.G.mp_fn, p["sigma_mp"])
                info["method"] = "radial limit of G(z)/(z-sigma)"
        beta = info.get("beta")
        is_dw = model.tau_on_boundary and abs(sig - model.tau) < 1e-6
        if is_dw:
            beta, exact = -model.lam, ({"sigma": model.summary_tau_exact(), "value": model.summary_lam_exact()}
                                      if model.lam_exact is not None else exact)
            info = {"beta": beta, "err": model.lam_err, "infinite": False, "ok": True,
                    "method": "beta = -lambda at the Denjoy–Wolff point"}
        if info.get("infinite"):
            kind = "super-repelling (non-regular)"
        elif beta is None:
            kind = "undetermined"
        elif is_dw:
            kind = "Denjoy–Wolff point"
        elif beta.real > 1e-9:
            kind = "repelling"
        else:
            kind = "inconsistent (beta <= 0 away from the DW point)"
        res.append({
            "sigma": sig,
            "angle_over_pi": float(np.angle(sig) / np.pi),
            "beta": beta if not info.get("infinite") else float("inf"),
            "beta_exact": exact,
            "beta_err": info.get("err"),
            "beta_imag_part": (float(beta.imag) if isinstance(beta, complex) else None),
            "kind": kind,
            "is_dw": bool(is_dw),
            "locate_method": p["method"],
            "beta_method": info["method"],
            "tier": 1 if model.rational else 2,
        })
    return res
