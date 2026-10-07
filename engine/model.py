"""Core model of a continuous semigroup given by its infinitesimal generator.

Conventions (Bracci–Contreras–Díaz-Madrigal):
* d/dt phi_t(z) = G(phi_t(z)),  G(z) = (z - tau)(conj(tau) z - 1) p(z), Re p >= 0.
* Spectral value lambda:  phi_t'(tau) = e^{-lambda t}.
  Elliptic (tau in D): lambda = -G'(tau), Re lambda >= 0 (= 0 iff group of rotations).
  Non-elliptic (tau in dD): lambda = -angular-lim G(z)/(z - tau) >= 0;
  hyperbolic iff lambda > 0, parabolic iff lambda = 0.
* Half-plane model (non-elliptic): w = (tau + z)/(tau - z) maps D onto
  H = {Re w > 0}, tau to infinity, 0 to 1.  The conjugated generator is
  Gamma(w) = (w+1)^2 G(z)/(2 tau) = 2 p(z) with Re Gamma >= 0, and
  lambda = lim_{x -> +inf} Gamma(x)/x.
* Elliptic normalised model: u = (z - tau)/(1 - conj(tau) z), generator G_0(u).
"""
from __future__ import annotations

import copy

import mpmath as mp
import numpy as np
import sympy as sp

from . import boundary
from .precision import MP_LOCK
from .expr import (
    ExpressionError, cancel_rational_parts, compile_expr, is_rational_in, latex,
    parse, t, u, w, z,
)


class ModelError(ValueError):
    pass


def _num_mp(expr, dps=50):
    v = sp.N(expr, dps)
    return mp.mpc(sp.N(sp.re(v), dps), sp.N(sp.im(v), dps))


def _try_exact_number(value: complex, verify, extra=(sp.sqrt(2), sp.sqrt(3), sp.sqrt(5))):
    """Try to recognise a numeric value as a closed form and verify it exactly."""
    try:
        re_ = sp.nsimplify(sp.Float(repr(value.real), 30), list(extra), tolerance=1e-18, rational=False)
        im_ = sp.nsimplify(sp.Float(repr(value.imag), 30), list(extra), tolerance=1e-18, rational=False)
    except Exception:  # noqa: BLE001
        return None
    cand = sp.nsimplify(re_ + sp.I * im_)
    if len(str(cand)) > 80:
        return None
    try:
        if verify(cand):
            return cand
    except Exception:  # noqa: BLE001
        return None
    return None


def _is_zero(expr) -> bool:
    try:
        val = sp.simplify(sp.expand(expr))
        if val == 0:
            return True
        return bool(abs(complex(sp.N(val, 50))) < 1e-40)
    except Exception:  # noqa: BLE001
        return False


class SemigroupModel:
    def __init__(self, spec: dict):
        with MP_LOCK:  # see precision.py
            self._build(spec)

    def _build(self, spec: dict):
        self.spec = dict(spec)
        self.warnings: list[str] = []
        self.notes: list[str] = []
        mode = spec.get("mode", "generator")
        self.mode = mode
        self.tau_given = None
        self.p_given = None
        self.phi_expr = None
        if mode == "generator":
            G_expr = parse(spec.get("G"), ("z",))
        elif mode == "bp":
            tau_e = parse(spec.get("tau"), ())
            if tau_e.free_symbols:
                raise ExpressionError("tau must be a constant.")
            tv = complex(sp.N(tau_e, 30))
            if abs(tv) > 1 + 1e-12:
                raise ModelError(f"|tau| = {abs(tv):.6g} > 1: tau must lie in the closed unit disc.")
            p_e = parse(spec.get("p"), ("z",))
            self.tau_given = tau_e
            self.p_given = p_e
            G_expr = (z - tau_e) * (sp.conjugate(tau_e) * z - 1) * p_e
        elif mode == "semigroup":
            phi = parse(spec.get("phi"), ("z", "t"))
            self.phi_expr = phi
            G_expr = sp.diff(phi, t).subs(t, 0)
            phi0 = phi.subs(t, 0)
            # identity check at t=0
            f0 = sp.lambdify(z, phi0, "numpy")
            pts = np.array([0.1, 0.3 + 0.2j, -0.5j, 0.7 * np.exp(2j)])
            with np.errstate(all="ignore"):
                val = np.asarray(f0(pts), dtype=complex)
            if np.max(np.abs(val - pts)) > 1e-9:
                self.warnings.append("phi_0 is not the identity; the input is not a semigroup.")
        else:
            raise ModelError(f"Unknown mode '{mode}'.")
        self.G_expr = sp.simplify(G_expr) if sp.count_ops(G_expr) < 60 else G_expr
        if self.G_expr == 0:
            raise ModelError("G is identically zero (trivial semigroup).")
        self.rational = is_rational_in(self.G_expr, z)
        if self.rational:
            self.G_expr = sp.cancel(sp.together(self.G_expr))
            num, den = sp.fraction(self.G_expr)
            self.num_expr, self.den_expr = sp.expand(num), sp.expand(den)
            deg_n = sp.Poly(self.num_expr, z).degree()
            deg_d = sp.Poly(self.den_expr, z).degree()
            if max(deg_n, deg_d) > 40:
                raise ModelError("Rational generators of degree > 40 are not supported.")
            self.numerator_roots = boundary.poly_roots(self.num_expr)
            self.denominator_roots = boundary.poly_roots(self.den_expr)
            inside_poles = [r for r in self.denominator_roots if abs(r) < 1 - 1e-12]
            if inside_poles:
                raise ModelError(
                    "G has a pole inside the disc at z = "
                    f"{complex(inside_poles[0]):.6g}; it is not holomorphic in D.")
        self.dG_expr = sp.diff(self.G_expr, z)
        self.d2G_expr = sp.diff(self.dG_expr, z)
        self.G = compile_expr(self.G_expr, z)
        self.dG = compile_expr(self.dG_expr, z)
        self.d2G = compile_expr(self.d2G_expr, z)
        self._detect_group()
        self._find_tau()
        self._spectral_value()
        self._build_normal_model()
        self._classify()

    # ------------------------------------------------------------------ helpers
    def denominator_vanishes(self, r) -> bool:
        if not self.rational:
            return False
        return any(abs(mp.mpc(r) - d) < 1e-15 for d in self.denominator_roots)

    def exact_value_at(self, value: complex, kind="dG"):
        """Exact closed form of G'(sigma) when sigma is recognisable exactly."""
        if not self.rational:
            return None
        sig = _try_exact_number(value, lambda c: _is_zero(self.num_expr.subs(z, c)))
        if sig is None:
            return None
        expr = self.dG_expr if kind == "dG" else self.G_expr
        try:
            val = sp.nsimplify(sp.simplify(expr.subs(z, sig)))
            return {"sigma": latex(sig), "value": latex(val)}
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------ group check
    def _detect_group(self):
        """Groups of automorphisms: G(z) = a - conj(a) z^2 + i b z with b real."""
        self.group = False
        self.group_exact = False
        if self.rational and sp.Poly(self.num_expr, z).degree() <= 2 and sp.Poly(self.den_expr, z).degree() == 0:
            P = sp.Poly(sp.expand(self.G_expr), z)
            c = [P.coeff_monomial(z ** k) for k in range(3)]
            cond1 = _is_zero(c[2] + sp.conjugate(c[0]))
            cond2 = _is_zero(sp.re(c[1]))
            self.group = bool(cond1 and cond2)
            self.group_exact = True
            return
        rng = np.random.default_rng(1)
        pts = 0.9 * np.sqrt(rng.random(40)) * np.exp(2j * np.pi * rng.random(40))
        vals = self.G(pts)
        if not np.all(np.isfinite(vals)):
            return
        V = np.vstack([np.ones_like(pts), pts, pts ** 2]).T
        coef, *_ = np.linalg.lstsq(V, vals, rcond=None)
        resid = np.max(np.abs(V @ coef - vals)) / max(1.0, np.max(np.abs(vals)))
        if resid < 1e-10:
            c0, c1, c2 = coef
            scale = max(1.0, abs(c0), abs(c1))
            self.group = abs(c2 + np.conj(c0)) < 1e-9 * scale and abs(c1.real) < 1e-9 * scale

    # -------------------------------------------------------- Denjoy–Wolff
    def _interior_zeros(self):
        if self.rational:
            roots = [r for r in self.numerator_roots
                     if abs(r) < 1 - 1e-12 and not self.denominator_vanishes(r)]
            return roots, "root of the numerator of G (40 digits)"
        rs = np.array([0.0, 0.35, 0.65, 0.85, 0.95])
        th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        seeds = np.unique((rs[:, None] * np.exp(1j * th[None, :])).ravel())
        zz = seeds.copy()
        with np.errstate(all="ignore"):
            for _ in range(80):
                g, dg = self.G(zz), self.dG(zz)
                step = np.where(np.abs(dg) > 0, g / dg, 0)
                step = np.where(np.abs(step) > 0.5, 0.5 * step / np.abs(step), step)
                zz = zz - step
        ok = np.isfinite(zz) & (np.abs(zz) < 1 - 1e-9)
        ok &= np.abs(self.G(np.where(ok, zz, 0))) < 1e-10
        roots = []
        for r in zz[ok]:
            if all(abs(r - q) > 1e-7 for q in roots):
                roots.append(complex(r))
        refined = []
        for r in roots:
            with mp.workdps(40):
                try:
                    rr = mp.findroot(self.G.mp_fn, mp.mpc(r), tol=mp.mpf(10) ** -32)
                except Exception:  # noqa: BLE001
                    rr = mp.mpc(r)
            if all(abs(rr - q) > 1e-12 for q in refined):
                refined.append(rr)
        return refined, "Newton multistart + high-precision refinement"

    def _find_tau(self):
        self.tau_exact = None
        self.tau_method = ""
        if self.tau_given is not None:
            self.tau_exact = sp.nsimplify(self.tau_given) if not self.tau_given.has(sp.Float) else self.tau_given
            self.tau_mp = _num_mp(self.tau_given)
            self.tau = complex(self.tau_mp)
            self.tau_on_boundary = abs(abs(self.tau) - 1) < 1e-14
            self.tau_method = "given (Berkson–Porta data)"
            interior, _ = self._interior_zeros()
            others = [r for r in interior if abs(complex(r) - self.tau) > 1e-8]
            if others:
                self.warnings.append("G has zeros in D other than tau — check that Re p >= 0.")
            self._boundary_pts = boundary.find_boundary_null_points(self)
            return
        interior, how = self._interior_zeros()
        if len(interior) > 1:
            raise ModelError(
                "G has more than one zero in D (" + ", ".join(f"{complex(r):.4g}" for r in interior[:4])
                + "); a non-trivial generator has at most one.")
        if len(interior) == 1:
            self.tau_mp = mp.mpc(interior[0])
            self.tau = complex(self.tau_mp)
            self.tau_on_boundary = False
            self.tau_method = "interior zero of G: " + how
            if self.rational:
                ex = _try_exact_number(self.tau, lambda c: _is_zero(self.num_expr.subs(z, c)))
                if ex is not None:
                    self.tau_exact = ex
                    self.tau_method += "; closed form verified symbolically"
            self._boundary_pts = boundary.find_boundary_null_points(self)
            return
        # boundary Denjoy–Wolff point
        self.tau_on_boundary = True
        self.tau = None
        self._boundary_pts = boundary.find_boundary_null_points(self)
        betas = []
        for p in self._boundary_pts:
            if self.rational:
                with mp.workdps(50):
                    b = complex(self.dG.mp(p["sigma_mp"], dps=50))
                betas.append((b.real, p))
            else:
                info = boundary.beta_via_halfplane(self.G_expr, p["sigma_mp"])
                if not info["ok"]:
                    info = boundary.radial_beta(self.G.mp_fn, p["sigma_mp"])
                if info["ok"] and not info["infinite"]:
                    betas.append((info["beta"].real, p))
        dw = [bp for bp in betas if bp[0] <= 1e-7]
        if len(dw) >= 1:
            dw.sort(key=lambda x: x[0])
            if len(dw) > 1:
                self.warnings.append("Several boundary null points with beta <= 0 were found; "
                                     "choosing the one with the smallest beta.")
            p = dw[0][1]
            self.tau_mp = p["sigma_mp"]
            self.tau = complex(p["sigma"])
            self.tau_method = "boundary null point with beta <= 0: " + p["method"]
        else:
            # fall back to the limit of the forward orbit of 0
            from .integrate import disk_margin, integrate
            res = integrate(self.G, np.array([0j]), [1e4], margin=disk_margin, eps_levels=(1e-13,),
                            rtol=1e-10, atol=1e-14, max_iter=200000)
            end = res.last[0]
            self.tau = complex(end / abs(end))
            self.tau_mp = mp.mpc(self.tau)
            self.tau_method = ("limit of the forward orbit of 0 (no boundary null point with "
                               "beta <= 0 was located; accuracy limited)")
            self.warnings.append("Denjoy–Wolff point located only from the forward orbit; "
                                 "boundary quantities near tau are less reliable.")
        if self.rational:
            ex = _try_exact_number(self.tau, lambda c: _is_zero(self.num_expr.subs(z, c))
                                   and _is_zero(sp.Abs(c) ** 2 - 1))
            if ex is not None:
                self.tau_exact = ex
                self.tau_method += "; closed form verified symbolically"

    # ------------------------------------------------------ spectral value
    def _spectral_value(self):
        self.lam_exact = None
        if not self.tau_on_boundary:
            with mp.workdps(50):
                lam = -self.dG.mp(self.tau_mp, dps=50)
            self.lam = complex(lam)
            self.lam_err = 1e-25 if self.rational else 1e-20
            self.lam_method = "lambda = -G'(tau)"
            if self.tau_exact is not None:
                try:
                    self.lam_exact = sp.nsimplify(sp.simplify(-self.dG_expr.subs(z, self.tau_exact)))
                except Exception:  # noqa: BLE001
                    pass
            return
        if self.rational and not self.denominator_vanishes(self.tau_mp):
            with mp.workdps(50):
                lam = -self.dG.mp(self.tau_mp, dps=50)
            self.lam = complex(lam)
            self.lam_err = 1e-25
            self.lam_method = "lambda = -G'(tau) (G analytic at tau)"
            if self.tau_exact is not None:
                try:
                    self.lam_exact = sp.nsimplify(sp.simplify(-self.dG_expr.subs(z, self.tau_exact)))
                except Exception:  # noqa: BLE001
                    pass
        else:
            info = self._lambda_from_gamma()
            if not info["ok"] or info["infinite"]:
                self.lam = complex("nan")
                self.lam_err = None
                self.warnings.append("Could not evaluate the angular derivative at tau.")
            else:
                self.lam = -info["beta"]
                self.lam_err = max(info["err"], 1e-12)
            self.lam_method = info["method"]
            self.lam_sequence = [-v for v in info.get("sequence", [])]
        if abs(self.lam.imag) < 1e-12 * max(1, abs(self.lam)):
            self.lam = complex(self.lam.real, 0.0)
        else:
            self.warnings.append(f"Im(lambda) = {self.lam.imag:.3g} should vanish at a boundary DW point.")

    def _lambda_from_gamma(self):
        """lambda = lim Gamma(x)/x, Gamma the half-plane generator (exact chart if p is given)."""
        if self.p_given is not None:
            tau_s = self._tau_sym()
            expr = cancel_rational_parts(2 * self.p_given.subs(z, tau_s * (w - 1) / (w + 1)), w)
            f = sp.lambdify(w, expr, "mpmath")
            vals = []
            with mp.workdps(60):
                for k in range(2, 61, 3):
                    x = mp.mpf(10) ** k
                    try:
                        vals.append(complex(-f(x) / x))
                    except Exception:  # noqa: BLE001
                        vals.append(complex("nan"))
            v = np.array(vals)
            v = v[np.isfinite(v)]
            if v.size >= 3:
                return {"beta": v[-1], "err": float(abs(v[-1] - v[-2])), "infinite": False, "ok": True,
                        "sequence": v.tolist(),
                        "method": "lambda = lim Gamma(x)/x with Gamma = 2 p(C^-1(w)), x = 10^2..10^59"}
        info = boundary.beta_via_halfplane(self.G_expr, self.tau_mp)
        info["method"] = "lambda = lim Gamma(x)/x in the half-plane chart, x = 10^2..10^32"
        if not info["ok"]:
            info = boundary.radial_beta(self.G.mp_fn, self.tau_mp, ks=range(3, 13))
            info["method"] = "lambda = -radial-lim G(z)/(z - tau), z = (1-10^-k) tau, k = 3..12"
        return info

    def summary_tau_exact(self):
        return latex(self.tau_exact) if self.tau_exact is not None else None

    def summary_lam_exact(self):
        return latex(-self.lam_exact) if self.lam_exact is not None else None

    # ------------------------------------------------ normalised models
    def _tau_sym(self):
        if self.tau_exact is not None:
            return self.tau_exact
        with mp.workdps(40):
            return sp.Float(mp.nstr(self.tau_mp.real, 35), 35) + sp.I * sp.Float(mp.nstr(self.tau_mp.imag, 35), 35)

    def _build_normal_model(self):
        tau_s = self._tau_sym()
        self.Gamma = None
        self.G0 = None
        if self.tau_on_boundary:
            if self.p_given is not None:
                expr = 2 * self.p_given.subs(z, tau_s * (w - 1) / (w + 1))
            else:
                expr = (w + 1) ** 2 / (2 * tau_s) * self.G_expr.subs(z, tau_s * (w - 1) / (w + 1))
            expr = cancel_rational_parts(expr, w)
            self.Gamma_expr = expr
            self.Gamma = compile_expr(expr, w)
        else:
            if abs(self.tau) < 1e-300:
                expr = self.G_expr.subs(z, u)
            else:
                zt = (u + tau_s) / (1 + sp.conjugate(tau_s) * u)
                expr = (1 - tau_s * sp.conjugate(tau_s)) / (1 - sp.conjugate(tau_s) * zt) ** 2 * self.G_expr.subs(z, zt)
                expr = cancel_rational_parts(expr, u)
            self.G0_expr = expr
            self.G0 = compile_expr(expr, u)

    # Möbius maps -------------------------------------------------------
    def to_H(self, zz):
        return (self.tau + zz) / (self.tau - zz)

    def from_H(self, ww):
        return self.tau * (ww - 1) / (ww + 1)

    def to_U(self, zz):
        return (zz - self.tau) / (1 - np.conj(self.tau) * zz)

    def from_U(self, uu):
        return (uu + self.tau) / (1 + np.conj(self.tau) * uu)

    # ---------------------------------------------------- classification
    def _classify(self):
        lam = self.lam
        err = self.lam_err if self.lam_err is not None else np.inf
        exact = self.lam_exact is not None
        if not self.tau_on_boundary:
            self.kind = "elliptic"
            if exact:
                re_zero = _is_zero(sp.re(self.lam_exact))
            else:
                re_zero = abs(lam.real) <= max(1e-10, 10 * err)
            if re_zero and not self.group:
                self.group = True
            self.kind_confidence = "exact" if exact or self.group_exact else "numerical"
            self.kind_reason = ("tau is an interior zero of G" +
                                ("; Re lambda = 0, so every phi_t is an automorphism (rotation group)" if re_zero else ""))
            self.step = None
            return
        if exact:
            parabolic = _is_zero(self.lam_exact)
            conf = "exact"
        else:
            tol = max(1e-9, 10 * err)
            parabolic = abs(lam.real) <= tol
            conf = "numerical"
            if parabolic:
                self.notes.append(
                    f"lambda = {lam.real:.3e} ± {err:.1e}: numerically zero. A hyperbolic semigroup "
                    f"with 0 < lambda < {tol:.1e} cannot be excluded numerically.")
        self.kind = "parabolic" if parabolic else "hyperbolic"
        self.kind_confidence = conf
        self.kind_reason = ("boundary Denjoy–Wolff point with " +
                            ("lambda = 0" if parabolic else f"lambda = {lam.real:.6g} > 0"))
        self.step = self._step_exact()

    def _step_exact(self):
        if self.kind == "hyperbolic":
            return {"kind": "positive", "confidence": "theorem",
                    "reason": "hyperbolic semigroups always have positive hyperbolic step"}
        if self.group:
            return {"kind": "positive", "confidence": "exact",
                    "reason": "group of parabolic automorphisms: k(phi_t z, phi_{t+1} z) is constant > 0"}
        if self.rational and self.tau_exact is not None and not self.denominator_vanishes(self.tau_mp):
            c0 = sp.nsimplify(sp.simplify(self.tau_exact * self.d2G_expr.subs(z, self.tau_exact)))
            if not _is_zero(c0):
                if _is_zero(sp.re(c0)):
                    return {"kind": "positive", "confidence": "exact (asymptotic analysis)",
                            "reason": f"G analytic at tau with tau·G''(tau) = {c0} purely imaginary: "
                                      "Gamma(w) -> i·kappa with Gamma - i kappa = O(1/w) and real 1/w-coefficient, "
                                      "so Re w(t) stays bounded and the orbits are asymptotically parallel to dH.",
                            "c0": latex(c0)}
                return {"kind": "zero", "confidence": "exact (asymptotic analysis)",
                        "reason": f"G analytic at tau with Re(tau·G''(tau)) = Re({c0}) > 0: "
                                  "Gamma(w) -> c0 with Re c0 > 0, so Re w(t) ~ Re(c0)·t and "
                                  "k(w(t), w(t+1)) -> 0.",
                        "c0": latex(c0)}
            return {"kind": None, "confidence": "undecided",
                    "reason": "G''(tau) = 0 (zero of order >= 3 at tau); only the numerical estimate is available."}
        return {"kind": None, "confidence": "undecided",
                "reason": "no closed-form criterion applies; see the numerical estimate."}

    # ---------------------------------------------------------- summary
    def summary(self) -> dict:
        """Headline results; computed once per model, under the precision lock."""
        with MP_LOCK:
            if getattr(self, "_summary_cache", None) is None:
                self._summary_cache = self._compute_summary()
            return copy.deepcopy(self._summary_cache)

    def _compute_summary(self) -> dict:
        p_expr = None
        if self.tau_exact is not None or self.tau is not None:
            if self.p_given is not None:
                p_expr = self.p_given
            elif self.tau_exact is not None:
                try:
                    p_expr = sp.cancel(self.G_expr / ((z - self.tau_exact) * (sp.conjugate(self.tau_exact) * z - 1)))
                except Exception:  # noqa: BLE001
                    p_expr = None
        bpts = boundary.classify_boundary_points(self, self._boundary_pts)
        return {
            "mode": self.mode,
            "G_latex": latex(self.G_expr),
            "G_text": str(self.G_expr),
            "p_latex": latex(p_expr) if p_expr is not None else None,
            "Gamma_latex": latex(self.Gamma_expr) if self.Gamma is not None and len(str(self.Gamma_expr)) < 400 else None,
            "G0_latex": latex(self.G0_expr) if self.G0 is not None and len(str(self.G0_expr)) < 400 else None,
            "rational": self.rational,
            "tau": self.tau,
            "tau_abs": abs(self.tau),
            "tau_exact": latex(self.tau_exact) if self.tau_exact is not None else None,
            "tau_location": "boundary" if self.tau_on_boundary else "interior",
            "tau_method": self.tau_method,
            "lambda": self.lam,
            "lambda_err": self.lam_err,
            "lambda_exact": latex(self.lam_exact) if self.lam_exact is not None else None,
            "lambda_method": self.lam_method,
            "kind": self.kind,
            "kind_confidence": self.kind_confidence,
            "kind_reason": self.kind_reason,
            "group": bool(self.group),
            "step": self.step,
            "boundary_points": bpts,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }
