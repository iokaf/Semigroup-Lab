"""Checks that the input really defines a semigroup of holomorphic self-maps of D.

* tau-free generator criterion:  Re(G(z) conj z) <= (1 - |z|^2) Re(G(0) conj z)  on D
  (characterises infinitesimal generators of semigroups of holomorphic self-maps).
* Berkson–Porta: Re p >= 0 for p = G / ((z - tau)(conj(tau) z - 1)).
* Holomorphy: finite values and a Cauchy–Riemann consistency test (catches branch cuts
  of principal-branch functions inside D).
* Semigroup law: |phi_{t+s} - phi_t ∘ phi_s| (closed form, or the numerical flow).
Sampling gives evidence, not proof: values are reported with the location of the worst case.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from .integrate import disk_margin, disk_margin_rate, integrate


def _sample(n_r=70, n_t=240):
    r = np.concatenate([np.linspace(0.02, 0.9, 30), 1 - np.geomspace(0.1, 1e-7, n_r - 30)])
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False) + 1e-3
    return (r[:, None] * np.exp(1j * th[None, :])).ravel()


def generator_inequality(model):
    Z = _sample()
    G = model.G(Z)
    G0 = model.G(np.array([0j]))[0]
    lhs = np.real(G * np.conj(Z))
    rhs = (1 - np.abs(Z) ** 2) * np.real(G0 * np.conj(Z))
    scale = np.abs(G) * np.abs(Z) + abs(G0) + 1e-300
    viol = (lhs - rhs) / scale
    ok = np.isfinite(viol)
    if not ok.any():
        return {"ok": False, "max_violation": None, "where": None, "text": "G is not finite on the sample."}
    i = np.nanargmax(np.where(ok, viol, -np.inf))
    v = float(viol[i])
    passed = v <= 1e-9
    return {"ok": bool(passed), "max_violation": v, "where": complex(Z[i]),
            "text": ("Re(G(z) z̄) ≤ (1-|z|²) Re(G(0) z̄) holds on the sample"
                     if passed else f"violated at z ≈ {Z[i]:.4g} (relative violation {v:.3g})")}


def berkson_porta(model):
    Z = _sample()
    tau = model.tau
    far = np.abs(Z - tau) > 1e-6
    Z = Z[far]
    with np.errstate(all="ignore"):
        p = model.G(Z) / ((Z - tau) * (np.conj(tau) * Z - 1))
    ok = np.isfinite(p)
    if not ok.any():
        return {"ok": False, "min_re_p": None, "where": None}
    rp = np.where(ok, p.real / np.maximum(1.0, np.abs(p)), np.inf)
    i = int(np.argmin(rp))
    passed = rp[i] >= -1e-9
    return {"ok": bool(passed), "min_re_p_relative": float(rp[i]), "where": complex(Z[i]),
            "p_at_worst": complex(p[i]),
            "text": ("Re p ≥ 0 on the sample" if passed
                     else f"Re p < 0 at z ≈ {Z[i]:.4g} (p = {p[i]:.4g})")}


def _branch_arguments(expr):
    """Arguments of principal-branch functions with their cut sets."""
    from .expr import z as zsym

    out = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and not node.exp.is_integer and node.base.has(zsym):
            out.append(("neg_real", node.base, f"({node.base})^({node.exp})"))
        elif isinstance(node, sp.log) and node.args[0].has(zsym):
            out.append(("neg_real", node.args[0], f"log({node.args[0]})"))
        elif isinstance(node, (sp.asin, sp.acos, sp.atanh)) and node.args[0].has(zsym):
            out.append(("real_gt1", node.args[0], f"{node.func.__name__}({node.args[0]})"))
        elif isinstance(node, (sp.atan, sp.asinh)) and node.args[0].has(zsym):
            out.append(("imag_gt1", node.args[0], f"{node.func.__name__}({node.args[0]})"))
        elif isinstance(node, sp.acosh) and node.args[0].has(zsym):
            out.append(("real_lt1", node.args[0], f"acosh({node.args[0]})"))
    return out


def branch_cut_scan(model, n_r=220, n_t=480):
    """Look for principal-branch cuts inside D across which G actually jumps."""
    from .expr import compile_expr
    from .expr import z as zsym

    args = _branch_arguments(model.G_expr)
    if not args:
        return []
    r = np.concatenate([np.linspace(0, 0.98, n_r - 40), 1 - np.geomspace(0.02, 1e-6, 40)])
    th = np.linspace(0, 2 * np.pi, n_t + 1)
    Z = r[:, None] * np.exp(1j * th[None, :])
    found = []
    for kind, arg, label in args:
        A = compile_expr(arg, zsym)(Z)
        if kind in ("neg_real", "real_gt1", "real_lt1"):
            s_ = np.sign(A.imag)
        else:
            s_ = np.sign(A.real)
        pairs = []
        for axis in (0, 1):
            a1 = np.take(A, range(A.shape[axis] - 1), axis=axis)
            a2 = np.take(A, range(1, A.shape[axis]), axis=axis)
            z1 = np.take(Z, range(Z.shape[axis] - 1), axis=axis)
            z2 = np.take(Z, range(1, Z.shape[axis]), axis=axis)
            s1 = np.take(s_, range(s_.shape[axis] - 1), axis=axis)
            s2 = np.take(s_, range(1, s_.shape[axis]), axis=axis)
            cross = (s1 * s2 < 0)
            mid = (a1 + a2) / 2
            if kind == "neg_real":
                cross &= mid.real < 0
            elif kind == "real_gt1":
                cross &= np.abs(mid.real) > 1
            elif kind == "real_lt1":
                cross &= mid.real < 1
            else:
                cross &= np.abs(mid.imag) > 1
            idx = np.nonzero(cross)
            pairs += list(zip(z1[idx], z2[idx]))
        if not pairs:
            continue
        p1 = np.array([a for a, _ in pairs[:400]])
        p2 = np.array([b for _, b in pairs[:400]])
        jump = np.abs(model.G(p1) - model.G(p2))
        smooth = np.abs(model.dG((p1 + p2) / 2)) * np.abs(p1 - p2)
        real_jump = jump > 10 * smooth + 1e-9 * (1 + np.abs(model.G(p1)))
        if real_jump.any():
            k = int(np.argmax(real_jump))
            found.append({"function": label, "example": complex((p1[k] + p2[k]) / 2),
                          "count": int(real_jump.sum())})
    return found


def holomorphy(model):
    rng = np.random.default_rng(7)
    n = 6000
    Z = (1 - 1e-4) * np.sqrt(rng.random(n)) * np.exp(2j * np.pi * rng.random(n))
    G = model.G(Z)
    finite = np.isfinite(G)
    h = 1e-6 * np.exp(2j * np.pi * rng.random(n))
    Gh = model.G(Z + h)
    dG = model.dG(Z)
    with np.errstate(all="ignore"):
        mism = np.abs(Gh - G - dG * h) / (np.abs(dG * h) + 1e-12 * (1 + np.abs(G)))
    bad = finite & np.isfinite(mism) & (mism > 1e-2)
    cuts = branch_cut_scan(model)
    out = {"finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
           "branch_cut_suspects": int(bad.sum()), "branch_cuts": cuts}
    if bad.any():
        out["example"] = complex(Z[np.argmax(np.where(bad, mism, 0))])
    out["ok"] = out["finite"] and out["branch_cut_suspects"] == 0 and not cuts
    if out["ok"]:
        out["text"] = "G finite and complex-differentiable on the sample; no branch cut inside D"
    elif cuts:
        c = cuts[0]
        out["text"] = (f"the principal branch of {c['function']} has a cut inside D across which G jumps "
                       f"(e.g. near z ≈ {c['example']:.4g}); G is not holomorphic in D")
    else:
        out["text"] = (f"{out['nonfinite_count']} non-finite values, {out['branch_cut_suspects']} points "
                       "where G is not complex-differentiable (branch cut or pole inside D?)")
    return out


def semigroup_law(model):
    rng = np.random.default_rng(3)
    Z = 0.95 * np.sqrt(rng.random(40)) * np.exp(2j * np.pi * rng.random(40))
    s_, t_ = 0.37, 0.81
    out = {}
    num = integrate(model.G, Z, [s_, t_, s_ + t_], rtol=1e-11, atol=1e-14, margin=disk_margin,
                    eps_levels=(1e-15,), margin_rate=disk_margin_rate)
    ys, yt, yst = num.y_out
    comp = integrate(model.G, ys, [t_], rtol=1e-11, atol=1e-14, margin=disk_margin,
                     eps_levels=(1e-15,), margin_rate=disk_margin_rate).y_out[0]
    with np.errstate(all="ignore"):
        diff = np.abs(comp - yst)
    diff = diff[np.isfinite(diff)]
    # no finite pair at all means the flow could not be followed: report it as a failed check
    out["numerical_flow_residual"] = float(diff.max()) if diff.size else float("inf")
    if getattr(model, "phi_expr", None) is not None:
        from .expr import t as tsym
        from .expr import z as zsym
        f = sp.lambdify((zsym, tsym), model.phi_expr, "numpy")
        with np.errstate(all="ignore"):
            ex_s = np.asarray(f(Z, s_), dtype=complex) * np.ones_like(Z)
            ex_t_of_s = np.asarray(f(ex_s, t_), dtype=complex) * np.ones_like(Z)
            ex_st = np.asarray(f(Z, s_ + t_), dtype=complex) * np.ones_like(Z)
            out["closed_form_semigroup_residual"] = float(np.nanmax(np.abs(ex_t_of_s - ex_st)))
            out["closed_form_vs_flow"] = float(np.nanmax(np.abs(ex_st - yst)))
            Zb = _sample(30, 120)
            vals = [np.nanmax(np.abs(np.asarray(f(Zb, tv), dtype=complex) * np.ones_like(Zb))) for tv in (0.1, 1, 5)]
        out["closed_form_max_abs"] = float(max(vals))
        out["ok"] = (out["closed_form_semigroup_residual"] < 1e-8 and out["closed_form_vs_flow"] < 1e-6
                     and out["closed_form_max_abs"] <= 1 + 1e-12)
    else:
        out["ok"] = out["numerical_flow_residual"] < 1e-6
    return out


def run_all(model):
    gi = generator_inequality(model)
    bp = berkson_porta(model)
    ho = holomorphy(model)
    sg = semigroup_law(model)
    verdict = gi["ok"] and bp["ok"] and ho["ok"] and sg["ok"]
    return {"generator_inequality": gi, "berkson_porta": bp, "holomorphy": ho,
            "semigroup_law": sg, "ok": bool(verdict)}
