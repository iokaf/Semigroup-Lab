"""Vectorised adaptive Dormand–Prince 5(4) integrator for complex ODEs.

Every point carries its own time and step size, so thousands of orbits
(a whole grid of initial conditions) advance together in NumPy.

Features needed by the analysis:
* outputs at prescribed times (steps are shortened to land on them);
* a *margin* function (e.g. ``1-|z|`` in the disc) with crossing times for
  several levels ``eps`` — the difference of crossing times is what separates
  finite-time escape from asymptotic approach to a boundary fixed point;
* a step cap ``h <= cap * margin / |f|`` so steps never jump far outside the
  domain where the vector field may be singular.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Dormand–Prince coefficients
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
E = B5 - B4


@dataclass
class IntegrationResult:
    t_out: np.ndarray          # (M,)
    y_out: np.ndarray          # (M, N) complex, NaN where not reached
    alive: np.ndarray          # (N,) reached final output time
    exit_time: np.ndarray      # (N,) time of the stop event (inf if alive)
    crossings: np.ndarray      # (L, N) first time margin < eps_l (inf if never)
    last: np.ndarray           # (N,) last valid state
    last_t: np.ndarray         # (N,)
    stalled: np.ndarray        # (N,) step size underflow / iteration budget hit
    path_t: list | None = None
    path_y: list | None = None


def integrate(f, y0, t_out, *, rtol=1e-9, atol=1e-12, margin=None, eps_levels=(),
              cap=0.5, h0=1e-2, max_iter=20000, record_path=False, h_max=None,
              max_steps_per_point=None, margin_rate=None):
    y = np.array(y0, dtype=complex).ravel().copy()
    n = y.size
    t_out = np.asarray(t_out, dtype=float)
    m = t_out.size
    t = np.zeros(n)
    h = np.full(n, float(h0))
    y_out = np.full((m, n), np.nan + 1j * np.nan)
    out_idx = np.zeros(n, dtype=int)
    eps_levels = tuple(sorted(eps_levels, reverse=True))
    crossings = np.full((len(eps_levels), n), np.inf)
    exit_time = np.full(n, np.inf)
    stalled = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    nsteps = np.zeros(n, dtype=int)
    stop_level = eps_levels[-1] if eps_levels else None

    # record outputs at t=0
    while True:
        mask = alive & (out_idx < m)
        hit = mask & (np.abs(t_out[np.minimum(out_idx, m - 1)] - t) <= 1e-14)
        if not hit.any():
            break
        idx = np.nonzero(hit)[0]
        y_out[out_idx[idx], idx] = y[idx]
        out_idx[idx] += 1
    done = out_idx >= m
    alive &= ~done
    finished = done.copy()

    if margin is not None:
        mg0 = margin(y)
        bad = ~np.isfinite(mg0) | ((mg0 <= 0) if stop_level is None else (mg0 < stop_level))
        bad &= alive
        exit_time[bad] = 0.0
        alive[bad] = False

    paths_t = [[0.0] for _ in range(n)] if record_path else None
    paths_y = [[y[i]] for i in range(n)] if record_path else None

    it = 0
    while alive.any() and it < max_iter:
        it += 1
        idx = np.nonzero(alive)[0]
        if max_steps_per_point is not None:
            nsteps[idx] += 1
            over = nsteps[idx] > max_steps_per_point
            if over.any():
                o_idx = idx[over]
                stalled[o_idx] = True
                alive[o_idx] = False
                exit_time[o_idx] = t[o_idx]
                idx = idx[~over]
                if idx.size == 0:
                    break
        yi, ti, hi = y[idx], t[idx], h[idx]
        target = t_out[out_idx[idx]]
        hi = np.minimum(hi, target - ti)
        if h_max is not None:
            hi = np.minimum(hi, h_max)
        k = [None] * 7
        k[0] = f(yi)
        if margin is not None and cap:
            mg = margin(yi)
            if margin_rate is not None:
                speed = np.maximum(-margin_rate(yi, k[0]), 0.0)  # only approach matters
            else:
                speed = np.abs(k[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.where(speed > 0, cap * np.abs(mg) / speed, np.inf)
            hi = np.minimum(hi, np.maximum(lim, 1e-14))
        for s in range(1, 7):
            acc = np.zeros_like(yi)
            for j, a in enumerate(A[s]):
                if a:
                    acc = acc + a * k[j]
            k[s] = f(yi + hi * acc)
        y5 = yi + hi * sum(B5[j] * k[j] for j in range(7) if B5[j])
        err_vec = hi * sum(E[j] * k[j] for j in range(7) if E[j])
        scale = atol + rtol * np.maximum(np.abs(yi), np.abs(y5))
        with np.errstate(all="ignore"):
            err = np.abs(err_vec) / scale
        finite = np.isfinite(y5) & np.isfinite(err)
        err = np.where(finite, err, np.inf)
        accept = err <= 1.0
        # step size update
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.where(err > 0, 0.9 * err ** (-0.2), 5.0)
        fac = np.clip(np.nan_to_num(fac, nan=0.2, posinf=5.0), 0.2, 5.0)
        new_h = hi * fac
        # rejected steps: shrink
        rej = ~accept
        if rej.any():
            r_idx = idx[rej]
            h[r_idx] = new_h[rej]
            tiny = h[r_idx] < 1e-13 * np.maximum(1.0, np.abs(t[r_idx]))
            if tiny.any():
                s_idx = r_idx[tiny]
                stalled[s_idx] = True
                alive[s_idx] = False
                exit_time[s_idx] = t[s_idx]
        if not accept.any():
            continue
        a_idx = idx[accept]
        y_old = y[a_idx]
        y_new = y5[accept]
        t_old = t[a_idx]
        t_new = t_old + hi[accept]
        stop_now = np.zeros(a_idx.size, dtype=bool)
        if margin is not None:
            m_old = margin(y_old)
            m_new = margin(y_new)
            bad = ~np.isfinite(m_new)
            for li, eps in enumerate(eps_levels):
                cr = (m_new < eps) & np.isinf(crossings[li, a_idx])
                if cr.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        frac = np.where(np.isfinite(m_new) & (m_old != m_new),
                                        (m_old - eps) / (m_old - m_new), 1.0)
                    frac = np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0)
                    tc = t_old + frac * (t_new - t_old)
                    crossings[li, a_idx[cr]] = tc[cr]
            if stop_level is not None:
                stop_now = bad | (m_new < stop_level)
            else:
                stop_now = bad | (m_new <= 0)
        y[a_idx] = np.where(np.isfinite(y_new), y_new, y_old)
        t[a_idx] = t_new
        h[a_idx] = new_h[accept]
        if record_path:
            for jj, i in enumerate(a_idx):
                if np.isfinite(y_new[jj]):
                    paths_t[i].append(t_new[jj])
                    paths_y[i].append(y_new[jj])
        if stop_now.any():
            s_idx = a_idx[stop_now]
            alive[s_idx] = False
            exit_time[s_idx] = t[s_idx]
        # outputs
        hitmask = (~stop_now) & (np.abs(t_new - t_out[out_idx[a_idx]]) <= 1e-12 * np.maximum(1.0, np.abs(t_new)))
        if hitmask.any():
            h_idx = a_idx[hitmask]
            y_out[out_idx[h_idx], h_idx] = y[h_idx]
            t[h_idx] = t_out[out_idx[h_idx]]
            out_idx[h_idx] += 1
            fin = out_idx[h_idx] >= m
            if fin.any():
                f_idx = h_idx[fin]
                alive[f_idx] = False
                finished[f_idx] = True
    if alive.any():  # iteration budget exhausted
        r = np.nonzero(alive)[0]
        stalled[r] = True
        exit_time[r] = t[r]
        alive[r] = False
    return IntegrationResult(
        t_out=t_out, y_out=y_out, alive=finished, exit_time=exit_time,
        crossings=crossings, last=y, last_t=t, stalled=stalled,
        path_t=[np.array(p) for p in paths_t] if record_path else None,
        path_y=[np.array(p) for p in paths_y] if record_path else None,
    )


def disk_margin(zz):
    return 1.0 - np.abs(zz)


def disk_margin_rate(zz, vel):
    """d/dt (1 - |z|) along the velocity ``vel``."""
    a = np.abs(zz)
    with np.errstate(all="ignore"):
        return np.where(a > 0, -np.real(np.conj(zz) * vel) / a, -np.abs(vel))
