"""Data for the pictures: vector field, phase portrait, flow lines, phi_t(D), grids."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .integrate import disk_margin, disk_margin_rate, integrate
from .orbits import default_horizon


def vector_field(model, n=21):
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = (X + 1j * Y).ravel()
    Z = Z[np.abs(Z) < 0.97]
    V = model.G(Z)
    ok = np.isfinite(V) & (np.abs(V) > 0)
    Z, V = Z[ok], V[ok]
    mag = np.abs(V)
    spacing = 2 / (n - 1)
    ref = np.quantile(mag, 0.9) if mag.size else 1.0
    length = 0.85 * spacing * np.clip((mag / max(ref, 1e-300)) ** 0.5, 0.15, 1.0)
    d = V / mag
    tip = Z + d * length
    xs, ys = [], []
    hx, hy = [], []
    for a, b, dd, L in zip(Z, tip, d, length):
        xs += [a.real, b.real, None]
        ys += [a.imag, b.imag, None]
        left = b - L * 0.35 * dd * np.exp(1j * 0.45)
        right = b - L * 0.35 * dd * np.exp(-1j * 0.45)
        hx += [left.real, b.real, right.real, None]
        hy += [left.imag, b.imag, right.imag, None]
    return {"shaft_x": xs, "shaft_y": ys, "head_x": hx, "head_y": hy,
            "max_abs": float(mag.max()) if mag.size else 0.0}


def phase_portrait_png(model, n=360):
    """Domain colouring of G: hue = arg G, lightness bands = log|G|."""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    Z = X + 1j * Y
    inside = np.abs(Z) < 1
    V = np.full(Z.shape, np.nan + 1j * np.nan)
    V[inside] = model.G(Z[inside])
    arg = np.angle(V)
    hue = (arg / (2 * np.pi)) % 1.0
    with np.errstate(all="ignore"):
        lm = np.log2(np.abs(V))
    band = lm - np.floor(lm)
    light = 0.42 + 0.16 * band
    sat = np.full(Z.shape, 0.75)
    rgb = _hsl_to_rgb(hue, sat, light)
    alpha = np.where(inside & np.isfinite(V), 255, 0).astype(np.uint8)
    img = np.dstack([(np.nan_to_num(rgb) * 255).astype(np.uint8), alpha])
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)  # uint8 H×W×4 is read as RGBA
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _hsl_to_rgb(h, s, l):  # noqa: E741
    c = (1 - np.abs(2 * l - 1)) * s
    hp = h * 6
    x = c * (1 - np.abs(hp % 2 - 1))
    zeros = np.zeros_like(h)
    conds = [(hp < 1), (hp < 2), (hp < 3), (hp < 4), (hp < 5), (hp <= 6)]
    r = np.select(conds, [c, x, zeros, zeros, x, c])
    g = np.select(conds, [x, c, c, x, zeros, zeros])
    b = np.select(conds, [zeros, zeros, x, c, c, x])
    m = l - c / 2
    return np.dstack([r + m, g + m, b + m])


def _polylines(Y):
    """(M, N) array of states -> list of [x[], y[]] without NaNs."""
    lines = []
    for j in range(Y.shape[1]):
        col = Y[:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            c = col[ok]
            lines.append([c.real.tolist(), c.imag.tolist()])
        else:
            lines.append([[], []])
    return lines


def flow_lines(model, T=None, n_out=160):
    T = T or default_horizon(model)
    seeds = []
    for r, k in ((0.3, 8), (0.6, 12), (0.88, 16)):
        a = np.linspace(0, 2 * np.pi, k, endpoint=False) + 0.1
        seeds.append(r * np.exp(1j * a))
    seeds = np.concatenate(seeds)
    tt = np.unique(np.concatenate([np.linspace(0, T, n_out // 2), np.geomspace(min(1e-3, T / 100), T, n_out // 2)]))
    fw = integrate(model.G, seeds, tt, margin=disk_margin, eps_levels=(1e-9,), rtol=1e-8,
                   atol=1e-12, max_steps_per_point=5000, margin_rate=disk_margin_rate)
    bw = integrate(lambda zz: -model.G(zz), seeds, tt, margin=disk_margin, eps_levels=(1e-9,),
                   rtol=1e-8, atol=1e-12, max_steps_per_point=5000, margin_rate=disk_margin_rate)
    fwd = _polylines(fw.y_out)
    bwd = _polylines(bw.y_out)
    # append the exit point of backward orbits so lines reach the circle
    for j in range(seeds.size):
        if not bw.alive[j] and np.isfinite(bw.last[j]):
            bwd[j][0].append(float(bw.last[j].real))
            bwd[j][1].append(float(bw.last[j].imag))
    return {"T": float(T), "seeds": seeds, "forward": fwd, "backward": bwd}


def images_of_disc(model, T=None, times=None):
    """phi_t(D) (image of |z| = 1 - 1e-4) and the push-forward of a polar grid."""
    T = T or default_horizon(model)
    if times is None:
        times = np.array([0.0, T / 16, T / 8, T / 4, T / 2, T])
    times = np.unique(np.asarray(times, dtype=float))
    th = np.linspace(0, 2 * np.pi, 721)
    bnd = (1 - 1e-4) * np.exp(1j * th)
    grid_r = np.array([0.3, 0.6, 0.8, 0.9])
    circ = [r * np.exp(1j * th[::3]) for r in grid_r]
    rr = np.linspace(0, 1 - 1e-3, 60)
    rays = [rr * np.exp(1j * a) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    pieces = [bnd] + circ + rays
    sizes = [p.size for p in pieces]
    allp = np.concatenate(pieces)
    exact = getattr(model, "phi_expr", None) is not None
    if exact:
        import sympy as sp

        from .expr import t as tsym
        from .expr import z as zsym
        f = sp.lambdify((zsym, tsym), model.phi_expr, "numpy")
        Y = np.empty((times.size, allp.size), dtype=complex)
        with np.errstate(all="ignore"):
            for i, tv in enumerate(times):
                Y[i] = np.asarray(f(allp, tv), dtype=complex) * np.ones_like(allp)
        method = "closed form phi_t(z)"
    else:
        res = integrate(model.G, allp, times, rtol=1e-9, atol=1e-13, margin=disk_margin,
                        eps_levels=(1e-14,), max_steps_per_point=8000, margin_rate=disk_margin_rate)
        Y = res.y_out
        method = "numerical flow of G"
    out = []
    for i, tv in enumerate(times):
        row = Y[i]
        chunks = np.split(row, np.cumsum(sizes)[:-1])
        def pl(c):
            ok = np.isfinite(c)
            return [c[ok].real.tolist(), c[ok].imag.tolist()]
        out.append({"t": float(tv), "boundary": pl(chunks[0]),
                    "circles": [pl(c) for c in chunks[1:1 + len(circ)]],
                    "rays": [pl(c) for c in chunks[1 + len(circ):]]})
    return {"method": method, "frames": out, "T": float(T)}


def halfplane_view(model, flows):
    """Flow lines and special points transported to H = {Re w > 0} (non-elliptic only)."""
    if model.kind == "elliptic":
        return None
    def tr(lines):
        res = []
        for x, y in lines:
            zz = np.array(x) + 1j * np.array(y)
            with np.errstate(all="ignore"):
                ww = model.to_H(zz) if zz.size else zz
            ok = np.isfinite(ww) & (np.abs(ww) < 1e6)
            res.append([ww[ok].real.tolist(), ww[ok].imag.tolist()])
        return res
    return {"forward": tr(flows["forward"]), "backward": tr(flows["backward"])}
