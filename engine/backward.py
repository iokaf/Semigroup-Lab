"""Time-free classification of backward orbits.

A point z lies in W = ∩ φ_t(D) exactly when the backward equation dw/ds = −F(w) has a solution
for all s ≥ 0. Every backward orbit either

* leaves the domain in finite time                        -> ESCAPE (z is not in W),
* converges to a repelling boundary fixed point σ          -> petal of σ,
* converges to the Denjoy–Wolff point τ (parabolic case)   -> parabolic petal,

or, for a group of rotations, never settles.

All decisions are geometric, so the result does not depend on how fast the flow runs
(replacing G by cG with c > 0 changes nothing):

Chart
    Non-elliptic semigroups are integrated in H = {Re w > 0}, w = (τ + z)/(τ − z), with
    generator Γ. τ sits at infinity, so nothing near τ is lost to rounding. Elliptic ones are
    integrated in the disc.

Escape
    The domain margin (Re w in H, 1 − |z| in D) becomes ≤ 0 (located by interpolation within the
    step) or, away from the capture balls, smaller than 10⁻¹²(1 + |w|); or the step size collapses
    next to the boundary. The last two catch orbits that hit ∂D in finite time while the field
    blows up or loses smoothness there (poles, square-root points).

Capture
    Near a repelling point w_σ the backward field is −β(w − w_σ) + O(c|w − w_σ|²) with β > 0
    real: a contraction towards w_σ. If r = |w − w_σ| < r_cap ≤ 0.005 β / c and the margin is at
    least 0.01 r, the orbit provably stays in the domain and converges to σ (the margin can only
    shrink by O(c r²/β) < 0.005 r along the way). Orbits creeping along the boundary inside the
    ball (margin < 0.01 r) are integrated further and captured once r < 10⁻⁹ (scale), where the
    residual uncertainty is below 10⁻¹⁸. c is measured by sampling Γ around w_σ.

Parabolic petal (parabolic semigroups only)
    Orbits that neither escape nor get captured run off to τ = ∞. They belong to the parabolic
    petal iff Re w(s) tends to a positive limit. With Re w(s) = x∞ + a/s + …, the limit is
    Richardson-extrapolated from the horizons s, 2s, 4s (x∞ ≈ 2x(4s) − x(2s)) and accepted
    when it is at least a quarter of the current Re w and consistent with the estimate from s, 2s. Theorem: hyperbolic semigroups have
    no backward orbit converging to τ, so this label is never used for them.

Horizon
    Integration runs in rounds with horizons T_k = T₀ 2^k, T₀ = 20/κ, where κ is the typical
    speed of the flow (median of |F|/(1 + |w|) on a fixed sample of D). Rounds stop when every orbit is
    classified. Orbits still unclassified at the end are reported as UNDECIDED, never guessed.
"""
from __future__ import annotations

import numpy as np

from .integrate import A, B5, E

ESCAPE, UNDECIDED, TAU, FIXED, IN_W, FAILED = -1, -2, -3, -4, -5, -6
_RUNNING = -100


class Chart:
    """Coordinates, vector field and capture balls for the backward flow."""

    def __init__(self, model, boundary_points):
        self.model = model
        self.half_plane = model.kind != "elliptic"
        if self.half_plane:
            self.F = model.Gamma
            self.to_chart = lambda z: model.to_H(np.asarray(z, dtype=complex))  # noqa: E731
            self.to_disc = lambda w: model.from_H(np.asarray(w, dtype=complex))  # noqa: E731
        else:
            self.F = model.G
            self.to_chart = lambda z: np.asarray(z, dtype=complex)  # noqa: E731
            self.to_disc = self.to_chart
        self.sigmas, self.betas = [], []
        for b in boundary_points:
            if b["kind"] != "repelling":
                continue
            beta = b["beta"]
            beta = float(np.real(beta)) if not isinstance(beta, str) else np.inf
            if np.isfinite(beta) and beta > 0:
                self.sigmas.append(complex(b["sigma"]))
                self.betas.append(beta)
        self.targets = [complex(self.to_chart(np.array([s]))[0]) for s in self.sigmas]
        self.scales = [max(1.0, abs(w)) for w in self.targets]
        self.r_cap = [self._capture_radius(k) for k in range(len(self.targets))]
        self.r_deep = [1e-9 * s for s in self.scales]

    # -- geometry ------------------------------------------------------------
    def margin(self, w):
        return w.real if self.half_plane else 1.0 - np.abs(w)

    def margin_rate(self, w, v):
        """d(margin)/ds along velocity v."""
        if self.half_plane:
            return v.real
        a = np.abs(w)
        with np.errstate(all="ignore"):
            return np.where(a > 0, -np.real(np.conj(w) * v) / a, -np.abs(v))

    def inward(self, k):
        return 1.0 + 0j if self.half_plane else -self.sigmas[k]

    def _capture_radius(self, k):
        """Largest r with r ≤ 0.005 β / c, where |F(w) − β(w − w_σ)| ≤ c |w − w_σ|² near w_σ."""
        wk, beta, scale = self.targets[k], self.betas[k], self.scales[k]
        rho = scale * np.array([1e-2, 3e-3, 1e-3, 3e-4])
        phi = np.linspace(-1.3, 1.3, 11)
        pts = wk + (rho[:, None] * np.exp(1j * phi)[None, :] * self.inward(k)).ravel()
        rr = np.abs(pts - wk)
        with np.errstate(all="ignore"):
            dev = np.abs(self.F(pts) - beta * (pts - wk)) / rr ** 2
        dev = dev[np.isfinite(dev)]
        c = float(dev.max()) if dev.size else np.inf
        r = 1e-2 * scale if c == 0 else min(1e-2 * scale, 0.005 * beta / c)
        others = [abs(wk - w) for j, w in enumerate(self.targets) if j != k]
        if others:
            r = min(r, 0.25 * min(others))
        return max(r, 1e-7 * scale)

    def typical_speed(self):
        """Typical speed κ of the flow: the median of |F|/(1 + |w|) over a fixed sample of the disc.

        It sets the unit of time for the horizon; replacing G by cG multiplies it by c.
        """
        r = np.array([0.3, 0.5, 0.7, 0.85])
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False) + 0.05
        w = self.to_chart((r[:, None] * np.exp(1j * a)[None, :]).ravel())
        with np.errstate(all="ignore"):
            v = np.abs(self.F(w)) / (1.0 + np.abs(w))
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.median(v)) if v.size else 1.0


def classify(chart, z, *, rounds=10, rtol=1e-9, atol=1e-12, max_steps=12000,
             parabolic_allowed=False, record_path=False, kappa=None):
    """Classify the backward orbits of the points z (disc coordinates). Returns a dict of arrays."""
    z = np.asarray(z, dtype=complex).ravel()
    n = z.size
    y = chart.to_chart(z).astype(complex)
    t = np.zeros(n)
    kappa = kappa or chart.typical_speed()
    T0 = 20.0 / kappa
    h = np.full(n, 0.05 / kappa)
    label = np.full(n, _RUNNING, dtype=int)
    exit_time = np.full(n, np.nan)
    exit_point = np.full(n, np.nan + 1j * np.nan)
    beta_est = np.full(n, np.nan)
    t_near = np.full(n, np.nan)
    r_near = np.full(n, np.nan)
    steps = np.zeros(n, dtype=int)
    K = len(chart.targets)
    tg = np.array(chart.targets, dtype=complex)
    rcap = np.array(chart.r_cap)
    rdeep = np.array(chart.r_deep)
    paths = [[(0.0, y[i])] for i in range(n)] if record_path else None

    def capture_test(idx, yy):
        """Return per-point target index or -1, and update the near-ball bookkeeping."""
        if K == 0:
            return np.full(idx.size, -1)
        r = np.abs(yy[:, None] - tg[None, :])
        m = chart.margin(yy)
        ok = (r < rcap[None, :]) & ((m[:, None] >= 0.01 * r) | (r < rdeep[None, :]))
        hit = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
        near = (r < 10 * rcap[None, :]).any(axis=1) & np.isnan(t_near[idx])
        if near.any():
            j = idx[near]
            t_near[j] = t[j]
            r_near[j] = r[near].min(axis=1)
        if (hit >= 0).any():
            sel = hit >= 0
            j = idx[sel]
            r_hit = r[sel, hit[sel]]
            dt = t[j] - t_near[j]
            with np.errstate(all="ignore"):
                beta_est[j] = np.where(dt > 0, np.log(r_near[j] / r_hit) / dt, np.nan)
        return hit

    def in_ball(yy):
        if K == 0:
            return np.zeros(yy.shape, dtype=bool)
        return (np.abs(yy[:, None] - tg[None, :]) < rcap[None, :]).any(axis=1)

    # points that start inside a capture ball
    idx0 = np.arange(n)
    hit0 = capture_test(idx0, y)
    label[hit0 >= 0] = hit0[hit0 >= 0]

    snapshots = []
    for rnd in range(rounds):
        Tk = T0 * 2 ** rnd
        while True:
            run = np.nonzero((label == _RUNNING) & (t < Tk * (1 - 1e-12)))[0]
            if run.size == 0:
                break
            over = steps[run] >= max_steps
            if over.any():
                label[run[over]] = FAILED
                run = run[~over]
                if run.size == 0:
                    break
            steps[run] += 1
            yi, ti = y[run], t[run]
            hi = np.minimum(h[run], Tk - ti)
            k0 = -chart.F(yi)
            # never step across the boundary or past a capture ball
            m0 = chart.margin(yi)
            approach = np.maximum(-chart.margin_rate(yi, k0), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                hi = np.minimum(hi, np.where(approach > 0, 0.5 * m0 / approach, np.inf))
                if K:
                    rmin = np.abs(yi[:, None] - tg[None, :]).min(axis=1)
                    hi = np.minimum(hi, np.where(np.abs(k0) > 0, 0.5 * rmin / np.abs(k0), np.inf))
            hi = np.maximum(hi, 1e-300)
            ks = [k0]
            for s in range(1, 7):
                acc = np.zeros_like(yi)
                for j, a in enumerate(A[s]):
                    if a:
                        acc = acc + a * ks[j]
                ks.append(-chart.F(yi + hi * acc))
            y5 = yi + hi * sum(B5[j] * ks[j] for j in range(7) if B5[j])
            err_vec = hi * sum(E[j] * ks[j] for j in range(7) if E[j])
            with np.errstate(all="ignore"):
                err = np.abs(err_vec) / (atol + rtol * np.maximum(np.abs(yi), np.abs(y5)))
            err = np.where(np.isfinite(err) & np.isfinite(y5), err, np.inf)
            accept = err <= 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = np.clip(np.nan_to_num(0.9 * err ** -0.2, nan=0.2, posinf=5.0), 0.2, 5.0)
            new_h = hi * fac
            # rejected steps
            rej = run[~accept]
            if rej.size:
                h[rej] = new_h[~accept]
                tiny = h[rej] < 1e-13 * np.maximum(1.0, t[rej] * kappa) / kappa
                if tiny.any():
                    j = rej[tiny]
                    near_bd = (chart.margin(y[j]) < 1e-4 * (1.0 + np.abs(y[j]))) & ~in_ball(y[j])
                    esc = j[near_bd]
                    label[esc] = ESCAPE                      # runs into a singularity on ∂D
                    exit_time[esc] = t[esc]
                    ed = chart.to_disc(y[esc])
                    exit_point[esc] = ed / np.abs(ed)
                    label[j[~near_bd]] = FAILED
            if not accept.any():
                continue
            acc_i = run[accept]
            y_old, y_new = y[acc_i], y5[accept]
            t_old, t_new = t[acc_i], t[acc_i] + hi[accept]
            m_old, m_new = chart.margin(y_old), chart.margin(y_new)
            y[acc_i], t[acc_i], h[acc_i] = y_new, t_new, new_h[accept]
            if record_path:
                for jj, i in enumerate(acc_i):
                    paths[i].append((t_new[jj], y_new[jj]))
            # reached the boundary: crossed it, or (away from every capture ball) come within
            # 1e-12 of it, which the step cap would otherwise approach in ever smaller steps when
            # the orbit hits ∂D in finite time (poles, square-root points)
            out = (m_new <= 0) | ((m_new < 1e-12 * (1.0 + np.abs(y_new))) & ~in_ball(y_new))
            if out.any():
                j = acc_i[out]
                with np.errstate(all="ignore"):
                    frac = np.where(m_new[out] <= 0, m_old[out] / (m_old[out] - m_new[out]), 1.0)
                frac = np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0)
                yx = y_old[out] + frac * (y_new[out] - y_old[out])
                label[j] = ESCAPE
                exit_time[j] = t_old[out] + frac * (t_new[out] - t_old[out])
                ed = chart.to_disc(yx)
                with np.errstate(all="ignore"):
                    exit_point[j] = ed / np.abs(ed)
                if record_path:
                    for jj, i in enumerate(j):
                        paths[i][-1] = (exit_time[i], yx[jj])
            inside = ~out
            if inside.any():
                j = acc_i[inside]
                hit = capture_test(j, y_new[inside])
                label[j[hit >= 0]] = hit[hit >= 0]
        if chart.half_plane:
            snapshots.append(y.real.copy())
            if parabolic_allowed and len(snapshots) >= 3:
                x1, x2, x4 = snapshots[-3], snapshots[-2], snapshots[-1]
                lim, lim_coarse = 2 * x4 - x2, 2 * x2 - x1
                with np.errstate(all="ignore"):
                    # positive limit, a fair share of the current value, and consistent across horizons
                    tau_ok = ((label == _RUNNING) & (x4 > 0) & (lim > 0.25 * x4)
                              & (np.abs(lim - lim_coarse) < 0.5 * lim) & (np.abs(y) > 10))
                label[tau_ok] = TAU
        if not (label == _RUNNING).any():
            break
    label[label == _RUNNING] = UNDECIDED
    out = {"label": label, "exit_time": exit_time, "exit_point": exit_point, "beta_est": beta_est,
           "t_max": float(T0 * 2 ** rnd), "kappa": kappa, "steps": steps}
    if record_path:
        out["path_t"] = [np.array([p[0] for p in pp]) for pp in paths]
        out["path_z"] = [chart.to_disc(np.array([p[1] for p in pp])) for pp in paths]
    return out
