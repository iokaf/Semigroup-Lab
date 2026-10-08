"""Forward / backward orbits, speeds of convergence, slopes and the hyperbolic step.

Hyperbolic distance normalisation: k_D(z, w) = artanh |(z - w)/(1 - conj(w) z)|,
so k_D(0, z) = artanh |z|.  In the right half-plane H the same metric is
k_H(a, b) = artanh |(a - b)/(a + conj(b))|.  All long-time quantities of
non-elliptic semigroups are computed in H, where they are numerically stable.
"""
from __future__ import annotations

import numpy as np

from .integrate import disk_margin, disk_margin_rate, integrate


# ----------------------------------------------------------- distances
def k_H(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    num = np.abs(a - b)
    den = np.abs(a + np.conj(b))
    with np.errstate(all="ignore"):
        rho = num / den
        one_minus = 4 * a.real * b.real / (den * (den + num))
        k = 0.5 * np.log((1 + rho) / one_minus)
    return np.where(rho < 0.5, np.arctanh(np.clip(rho, 0, 0.999999)), k)


def k_D(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    with np.errstate(all="ignore"):
        rho = np.abs(a - b) / np.abs(1 - np.conj(b) * a)
    return np.arctanh(np.clip(rho, 0, 1 - 1e-16))


def _fit_line(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None, None
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    coef, res, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    pred = A @ coef
    resid = float(np.sqrt(np.mean((pred - y[ok]) ** 2)))
    return coef, resid


def _clean(arr):
    arr = np.asarray(arr)
    return arr


# ----------------------------------------------------- visual horizons
def default_horizon(model):
    lam = model.lam
    if model.kind == "elliptic":
        if model.group:
            return 2 * np.pi / max(abs(lam.imag), 1e-3) if abs(lam.imag) > 0 else 10.0
        return float(np.clip(5.0 / max(lam.real, 1e-6), 1.0, 60.0))
    if model.kind == "hyperbolic":
        return float(np.clip(5.0 / max(lam.real, 1e-6), 1.0, 60.0))
    return 40.0


# ---------------------------------------------------------- disc orbits
def forward_orbit_disk(model, z0, T, n=300):
    tt = np.unique(np.concatenate([np.linspace(0, T, n), np.geomspace(min(1e-3, T / 10), T, n // 2)]))
    res = integrate(model.G, np.array([z0]), tt, margin=disk_margin, eps_levels=(1e-15,),
                    rtol=1e-10, atol=1e-14, max_iter=100000)
    zz = res.y_out[:, 0]
    ok = np.isfinite(zz)
    return tt[ok], zz[ok]


# ------------------------------------------------- long-time statistics
def _tail(x, frac=0.3):
    n = len(x)
    return slice(int(n * (1 - frac)), n)


def nonelliptic_asymptotics(model, z0=0j, T=None, n=320):
    lam = model.lam.real
    if T is None:
        T = 35.0 / lam if model.kind == "hyperbolic" else 1e4
    T = float(min(T, 1e6))
    if model.kind == "hyperbolic":
        T = float(min(T, 60.0 / max(lam, 1e-12)))
    tt = np.geomspace(1e-2, T, n)
    tt_all = np.unique(np.concatenate([[0.0], tt, tt + 1.0]))
    w0 = complex(model.to_H(z0))
    res = integrate(model.Gamma, np.array([w0]), tt_all, rtol=1e-11, atol=1e-300,
                    margin=lambda ww: np.minimum(ww.real / np.abs(ww), 1e30 / np.abs(ww)),
                    eps_levels=(1e-14,), cap=0, max_iter=200000, h0=1e-3)
    ww = res.y_out[:, 0]
    ok = np.isfinite(ww)
    t_ok, w_ok = tt_all[ok], ww[ok]
    if t_ok.size < 10:
        return {"ok": False, "reason": "orbit could not be integrated in the half-plane model"}
    idx = {round(v, 12): i for i, v in enumerate(t_ok)}
    base = np.array([1.0 + 0j])  # image of z = 0
    dist = 2 / np.abs(w_ok + 1)
    margin = 4 * w_ok.real / np.abs(w_ok + 1) ** 2  # 1 - |z|^2
    v_tot = k_H(base, w_ok)
    v_o = 0.5 * np.abs(np.log(np.abs(w_ok)))
    eps = np.arctan2(w_ok.real, np.abs(w_ok.imag))
    with np.errstate(divide="ignore"):
        v_T = -0.5 * np.log(np.tan(np.clip(eps, 1e-300, np.pi / 2) / 2))
    slope = -np.angle(w_ok + 1)  # arg(1 - conj(tau) phi_t(z))
    # hyperbolic step s(t) = k(phi_t z, phi_{t+1} z)
    st, sv = [], []
    for i, tv in enumerate(t_ok):
        j = idx.get(round(tv + 1.0, 12))
        if j is not None:
            st.append(tv)
            sv.append(float(k_H(w_ok[i], w_ok[j])))
    st, sv = np.array(st), np.array(sv)
    out = {
        "ok": True, "z0": complex(z0), "T": float(t_ok[-1]),
        "t": t_ok, "w": w_ok, "dist_tau": dist, "one_minus_abs2": margin,
        "v_total": v_tot, "v_orth": v_o, "v_tan": v_T, "slope": slope,
        "step_t": st, "step_v": sv,
        "reached_T": bool(res.alive[0]),
    }
    tl = _tail(t_ok)
    fits = {}
    if model.kind == "hyperbolic":
        c, r = _fit_line(t_ok[tl], -np.log(dist[tl]))
        if c is not None:
            fits["rate_dist"] = {"value": float(c[0]), "resid": r,
                                 "meaning": "-log|phi_t(z) - tau| ~ rate·t (expected rate = lambda)"}
        c, r = _fit_line(t_ok[tl], v_o[tl])
        if c is not None:
            fits["v_orth_slope"] = {"value": float(c[0]), "resid": r,
                                    "meaning": "v_o(t) ~ slope·t (expected lambda/2)"}
        c, r = _fit_line(t_ok[tl], v_tot[tl])
        if c is not None:
            fits["v_total_slope"] = {"value": float(c[0]), "resid": r,
                                     "meaning": "v(t) ~ slope·t (expected lambda/2)"}
    else:
        lt = np.log(t_ok[tl])
        c, r = _fit_line(lt, np.log(dist[tl]))
        if c is not None:
            fits["dist_power"] = {"value": float(-c[0]), "resid": r,
                                  "meaning": "|phi_t(z) - tau| ~ C t^(-alpha); alpha shown"}
        c, r = _fit_line(lt, v_tot[tl])
        if c is not None:
            fits["v_total_log"] = {"value": float(c[0]), "resid": r,
                                   "meaning": "v(t) ~ c·log t; c shown"}
        c, r = _fit_line(lt, v_o[tl])
        if c is not None:
            fits["v_orth_log"] = {"value": float(c[0]), "resid": r,
                                  "meaning": "v_o(t) ~ c·log t; c shown"}
        c, r = _fit_line(lt, v_T[tl])
        if c is not None:
            fits["v_tan_log"] = {"value": float(c[0]), "resid": r,
                                 "meaning": "v_T(t) ~ c·log t; c shown"}
    out["fits"] = fits
    tail_s = slope[tl]
    out["slope_tail"] = {"min": float(np.min(tail_s)), "max": float(np.max(tail_s)),
                         "last": float(slope[-1]),
                         "tangential": bool(np.max(np.abs(tail_s)) > np.pi / 2 - 1e-3)}
    out["step_numeric"] = numeric_step_verdict(model, st, sv)
    return out


def numeric_step_verdict(model, st, sv):
    if st.size < 10:
        return {"kind": None, "confidence": "insufficient data"}
    tl = _tail(st, 0.35)
    x, y = np.log(st[tl]), np.log(np.maximum(sv[tl], 1e-300))
    c, _ = _fit_line(x, y)
    slope = float(c[0]) if c is not None else np.nan
    last = float(sv[-1])
    if model.kind == "hyperbolic":
        kind = "positive"
    elif slope < -0.2 or last < 1e-6:
        kind = "zero"
    elif abs(slope) < 0.05 and last > 1e-3:
        kind = "positive"
    else:
        kind = None
    return {"kind": kind, "loglog_tail_slope": slope, "last_value": last, "t_last": float(st[-1]),
            "confidence": "numerical estimate (limit t -> inf cannot be decided numerically)",
            "rule": "tail log-log slope < -0.2 -> zero; |slope| < 0.05 and value > 1e-3 -> positive"}


def elliptic_asymptotics(model, z0=0.5, T=None, n=320):
    lam = model.lam
    if T is None:
        T = 35.0 / lam.real if lam.real > 1e-9 else 4 * np.pi / max(abs(lam.imag), 1e-3)
    T = float(min(T, 1e5))
    if abs(z0 - model.tau) < 1e-12:
        z0 = model.tau + 0.5 * (1 - abs(model.tau))
    tt = np.unique(np.concatenate([[0.0], np.linspace(T / n, T, n), np.geomspace(1e-3, T, n)]))
    u0 = complex(model.to_U(z0))
    res = integrate(model.G0, np.array([u0]), tt, rtol=1e-11, atol=1e-300, max_iter=200000,
                    margin=lambda uu: np.where(np.abs(uu) > 1e-280, 1 - np.abs(uu), -1.0),
                    eps_levels=(1e-15,), cap=0.5)
    uu = res.y_out[:, 0]
    ok = np.isfinite(uu)
    t_ok, u_ok = tt[ok], uu[ok]
    tau = model.tau
    dz = u_ok * (1 - abs(tau) ** 2) / (1 + np.conj(tau) * u_ok)
    k_tau = np.arctanh(np.abs(u_ok))
    arg = np.unwrap(np.angle(u_ok))
    out = {"ok": True, "z0": complex(z0), "T": float(t_ok[-1]) if t_ok.size else 0.0, "t": t_ok, "u": u_ok,
           "dist_tau": np.abs(dz), "k_to_tau": k_tau, "arg": arg}
    fits = {}
    if t_ok.size > 10:
        tl = _tail(t_ok)
        if not model.group:
            c, r = _fit_line(t_ok[tl], -np.log(np.abs(u_ok[tl])))
            if c is not None:
                fits["rate"] = {"value": float(c[0]), "resid": r,
                                "meaning": "-log|phi_t(z) - tau| ~ rate·t (expected Re lambda)"}
        c, r = _fit_line(t_ok[tl], -arg[tl])
        if c is not None:
            fits["rotation"] = {"value": float(c[0]), "resid": r,
                                "meaning": "-arg(phi_t(z) - tau) ~ omega·t (expected Im lambda)"}
    out["fits"] = fits
    return out
