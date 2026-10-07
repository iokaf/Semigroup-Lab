"""Koenigs function h and Koenigs domain Omega = h(D).

Non-elliptic:  h'(z) G(z) = i,  h(0) = 0,   h ∘ phi_t = h + i t.
  Omega + it ⊂ Omega (starlike at infinity, vertical direction).
  hyperbolic        <=> Omega lies in a vertical strip; minimal width pi/lambda
  parabolic, step>0 <=> Omega lies in a vertical half-plane but in no vertical strip
  parabolic, step=0 <=> Omega lies in no vertical half-plane
Elliptic:  h'(z) G(z) = -lambda h(z), h(tau) = 0, h ∘ phi_t = e^{-lambda t} h;
  Omega is lambda-spirallike (starlike when lambda is real).
  Computed as h = h_0 ∘ M with M(z) = (z - tau)/(1 - conj(tau) z) and
  h_0(u) = u exp(∫_0^u (-lambda/G_0(s) - 1/s) ds),  so h_0'(0) = 1.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from .expr import compile_expr, is_rational_in, u, z

# composite Gauss–Legendre nodes on [0, 1], refined geometrically toward s = 1
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)


def _nodes(levels=22):
    edges = np.concatenate([[0.0], 1 - 0.5 ** np.arange(1, levels + 1), [1.0]])
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        xs.append(a + (b - a) * (_GL_X + 1) / 2)
        ws.append((b - a) / 2 * _GL_W)
    return np.concatenate(xs), np.concatenate(ws)


_S, _WS = _nodes()


def _segment_integral(f, endpoints, chunk=4000):
    """∫_0^z f(ζ) dζ along the segment [0, z] for each z in ``endpoints``."""
    zz = np.asarray(endpoints, dtype=complex).ravel()
    out = np.empty(zz.size, dtype=complex)
    for start in range(0, zz.size, chunk):
        zc = zz[start:start + chunk]
        pts = zc[:, None] * _S[None, :]
        with np.errstate(all="ignore"):
            vals = f(pts.ravel()).reshape(pts.shape)
        out[start:start + chunk] = zc * (vals @ _WS)
    return out


class Koenigs:
    def __init__(self, model):
        self.model = model
        self.elliptic = model.kind == "elliptic"
        if self.elliptic:
            lam = model.lam_exact if model.lam_exact is not None else sp.Float(model.lam.real, 30) + sp.I * sp.Float(model.lam.imag, 30)
            g0 = model.G0_expr
            integrand = -lam / g0 - 1 / u
            if is_rational_in(integrand, u):
                integrand = sp.cancel(sp.together(integrand))
                self.method = "h_0(u) = u·exp(∫ (-λ/G_0 - 1/u) du), integrand simplified symbolically"
            else:
                self.method = "h_0(u) = u·exp(∫ (-λ/G_0 - 1/u) du), numerical integrand"
            self._f = compile_expr(integrand, u)
            self.formula = None
        else:
            G = model.G
            self._f = lambda s: 1j / G(s)  # noqa: E731
            self.method = "h(z) = i ∫_0^z dζ / G(ζ) (composite Gauss–Legendre, refined toward the endpoint)"
            self.formula = None
            if model.rational and sp.Poly(model.num_expr, z).degree() <= 4 and sp.Poly(model.den_expr, z).degree() <= 4:
                try:
                    anti = sp.integrate(sp.apart(sp.I / model.G_expr, z), z)
                    anti = anti - anti.subs(z, 0)
                    if len(str(anti)) < 300:
                        self.formula = sp.latex(anti)
                except Exception:  # noqa: BLE001
                    self.formula = None

    def __call__(self, zz):
        zz = np.asarray(zz, dtype=complex)
        shape = zz.shape
        flat = zz.ravel()
        out = np.full(flat.shape, np.nan + 1j * np.nan)
        ok = np.isfinite(flat) & (np.abs(flat) < 1)
        if self.elliptic:
            uu = self.model.to_U(flat[ok])
            I = _segment_integral(self._f, uu)
            with np.errstate(all="ignore"):
                out[ok] = uu * np.exp(I)
        else:
            out[ok] = _segment_integral(self._f, flat[ok])
        return out.reshape(shape)


def koenigs_data(model, n_circles=None, n_radii=24):
    K = Koenigs(model)
    radii = np.array([0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]) if n_circles is None else np.asarray(n_circles)
    th = np.linspace(0, 2 * np.pi, 721)
    circles = []
    for r in radii:
        hz = K(r * np.exp(1j * th))
        circles.append({"r": float(r), "h": hz})
    rr = np.concatenate([np.linspace(0, 0.9, 40), 1 - np.geomspace(0.1, 1e-4, 40)])
    rays = []
    for a in np.linspace(0, 2 * np.pi, n_radii, endpoint=False):
        rays.append({"angle": float(a), "h": K(rr * np.exp(1j * a))})
    geom = omega_geometry(model, K)
    return {"method": K.method, "formula": K.formula, "circles": circles, "rays": rays,
            "geometry": geom, "K": K}


def omega_geometry(model, K):
    """Estimate inf/sup of Re h on circles |z| = 1 - 2^-k (non-elliptic case)."""
    if model.kind == "elliptic":
        lam = model.lam
        if model.group:
            txt = "Group of rotations: h is a Möbius normalisation and Omega is a disc."
        elif abs(lam.imag) < 1e-12:
            txt = "lambda is real: Omega is starlike with respect to 0."
        else:
            txt = f"Omega is lambda-spirallike (lambda = {lam.real:.4g} {'+' if lam.imag >= 0 else '-'} {abs(lam.imag):.4g}i)."
        return {"kind": "elliptic", "text": txt}
    ks = np.arange(2, 19)
    th0 = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    special = [complex(p["sigma"]) for p in getattr(model, "_boundary_pts", [])]
    if model.tau is not None and model.tau_on_boundary:
        special.append(complex(model.tau))
    sups, infs = [], []
    for k in ks:
        r = 1 - 0.5 ** k
        extra = [np.angle(sg) + sgn * (1 - r) * np.geomspace(1e-2, 1e3, 80)
                 for sg in special for sgn in (-1, 1)]
        th = np.concatenate([th0] + extra) if extra else th0
        hv = K(r * np.exp(1j * th)).real
        hv = hv[np.isfinite(hv)]
        if hv.size == 0:
            sups.append(np.nan)
            infs.append(np.nan)
            continue
        sups.append(float(hv.max()))
        infs.append(float(hv.min()))
    sups, infs = np.array(sups), np.array(infs)

    def bounded(seq):
        d = np.diff(seq)
        d = d[np.isfinite(d)]
        if d.size < 7:
            return None
        tail = np.abs(d[-6:])
        ref = 1 + abs(seq[0]) + abs(np.nanmax(seq) - np.nanmin(seq))
        if tail[-1] < 1e-5 * ref:
            return True
        ratio = (max(tail[-1], 1e-300) / max(tail[0], 1e-300)) ** (1 / 5)
        if ratio < 0.8:
            return True
        if ratio > 0.93 and tail[-1] > 1e-4 * ref:
            return False
        return None

    b_sup, b_inf = bounded(sups), bounded(infs)
    out = {"kind": "non-elliptic", "k": ks, "special_angles": [float(np.angle(sg)) for sg in special], "radius": 1 - 0.5 ** ks, "sup_re": sups, "inf_re": infs,
           "sup_bounded": b_sup, "inf_bounded": b_inf}
    def aitken(seq):
        s0, s1, s2 = seq[-3:]
        den = (s2 - s1) - (s1 - s0)
        return s2 - (s2 - s1) ** 2 / den if abs(den) > 1e-15 else s2

    if b_sup and b_inf:
        width = aitken(sups) - aitken(infs)
        out["width_raw"] = float(sups[-1] - infs[-1])
        out["shape"] = "vertical strip"
        out["width"] = float(width)
        out["lambda_from_width"] = float(np.pi / width) if width > 0 else None
        out["text"] = (f"Omega appears to lie in a vertical strip of width ≈ {width:.5g} (Aitken-extrapolated), "
                       f"i.e. lambda ≈ pi/width = {np.pi / width:.5g} (hyperbolic).")
    elif b_sup or b_inf:
        out["shape"] = "vertical half-plane"
        out["text"] = ("Re h is bounded on one side only: Omega appears to lie in a vertical half-plane "
                       "but not in a strip (parabolic, positive hyperbolic step).")
    elif b_sup is False and b_inf is False:
        out["shape"] = "no half-plane"
        out["text"] = ("Re h appears unbounded on both sides: Omega is in no vertical half-plane "
                       "(parabolic, zero hyperbolic step).")
    else:
        out["shape"] = "inconclusive"
        out["text"] = "The growth of Re h near the circle is inconclusive at the radii sampled."
    out["note"] = ("Heuristic: decided from |z| = 1 - 2^-k, k <= 18 with 2048 angles; "
                   "boundedness of an unbounded domain cannot be certified numerically.")
    return out
