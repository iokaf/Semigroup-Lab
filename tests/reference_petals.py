"""Independent reference for W and its petals, used only by the tests.

It shares nothing with the app's backward-orbit code: each point is integrated separately with
SciPy's DOP853 (rtol 1e-11) and exact event location.

* chart: H = {Re w > 0} with τ at infinity (non-elliptic), or the disc (elliptic);
* event "leave the domain" (Re w = 0 or |z| = 1)                    -> ESCAPE;
* event "enter |w − w_σ| < 1e-6 (scale) around a repelling point σ"  -> petal of σ;
* alive after a long horizon: parabolic petal if Re w tends to a positive limit (Richardson from
  T/4, T/2, T), otherwise UNDECIDED.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import solve_ivp

ESC, UND, TAU = -1, -2, -3


def reference_labels(model, n=31, T=4000.0, eps=1e-6):
    s = model.summary()
    reps = [complex(b["sigma"]) for b in s["boundary_points"] if b["kind"] == "repelling"]
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    inside = np.abs(Z) < 1 - 1.0 / n
    lab = np.full(Z.shape, -9, dtype=int)
    tau = model.tau
    half_plane = model.kind != "elliptic"
    F = model.Gamma if half_plane else model.G
    targets = [(tau + r) / (tau - r) for r in reps] if half_plane else reps

    def rhs(_s, y):
        v = -F(np.array([y[0] + 1j * y[1]]))[0]
        return [v.real, v.imag]

    def leave(_s, y):
        return y[0] if half_plane else 1.0 - np.hypot(y[0], y[1])
    leave.terminal, leave.direction = True, -1
    events = [leave]
    for wt in targets:
        def hit(_s, y, wt=wt):
            return abs(complex(y[0], y[1]) - wt) - eps * max(1.0, abs(wt))
        hit.terminal, hit.direction = True, -1
        events.append(hit)

    for idx in zip(*np.nonzero(inside)):
        z0 = Z[idx]
        if not half_plane and abs(z0 - tau) < 1e-12:
            lab[idx] = UND
            continue
        w0 = complex(model.to_H(np.array([z0]))[0]) if half_plane else z0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(rhs, (0, T), [w0.real, w0.imag], method="DOP853", rtol=1e-11, atol=1e-13,
                            events=events, t_eval=[T / 4, T / 2, T])
        if sol.status == 1:
            k = [i for i, e in enumerate(sol.t_events) if len(e)][0]
            lab[idx] = ESC if k == 0 else k - 1
        elif sol.status == -1:
            lab[idx] = ESC                      # integration broke down at a pole on the boundary
        elif half_plane:
            x4, x2, x1 = sol.y[0]
            lim, lim_c = 2 * x1 - x2, 2 * x2 - x4
            big = abs(complex(sol.y[0][-1], sol.y[1][-1])) > 10
            lab[idx] = TAU if (big and x1 > 0 and lim > 1e-3 and abs(lim - lim_c) < 0.5 * lim) else UND
        else:
            lab[idx] = UND
    return g, inside, lab
