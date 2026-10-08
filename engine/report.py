"""Orchestration of the analyses and conversion to JSON-safe structures."""
from __future__ import annotations

import json
import math
import threading
import time
from collections import OrderedDict

import numpy as np

from . import koenigs, orbits, petals, validate, visuals
from .model import SemigroupModel

_CACHE: "OrderedDict[str, SemigroupModel]" = OrderedDict()
_LOCK = threading.Lock()
_MAX = 32


def spec_key(spec: dict) -> str:
    keep = {k: spec.get(k) for k in ("mode", "G", "tau", "p", "phi") if spec.get(k) not in (None, "")}
    return json.dumps(keep, sort_keys=True)


def get_model(spec: dict) -> SemigroupModel:
    key = spec_key(spec)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    model = SemigroupModel(spec)
    with _LOCK:
        _CACHE[key] = model
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return model


def _round_sig(arr, sig):
    arr = np.asarray(arr, dtype=float)
    with np.errstate(all="ignore"):
        mag = np.where(np.isfinite(arr) & (arr != 0), np.floor(np.log10(np.abs(arr))), 0)
        fac = 10.0 ** (sig - 1 - mag)
        return np.where(np.isfinite(arr), np.round(arr * fac) / fac, arr)


def compact(obj, sig=6):
    """Round every float in a nested structure to ``sig`` significant digits (for plots)."""
    if isinstance(obj, dict):
        return {k: compact(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, float) or x is None for x in obj):
            arr = np.array([np.nan if x is None else x for x in obj])
            return _nan_to_none(_round_sig(arr, sig).tolist())
        return [compact(v, sig) for v in obj]
    if isinstance(obj, float):
        return float(_round_sig(np.array([obj]), sig)[0])
    return obj


def jsonable(obj):
    """Recursively convert numpy / complex / non-finite values to JSON-safe objects.

    Complex numbers become [re, im]; NaN and +-inf become None ("inf" strings are
    used for documented infinite quantities such as beta = +inf).
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not callable(v) and k != "K"}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [jsonable(obj.real), jsonable(obj.imag)] if obj.ndim == 0 else \
                {"re": jsonable(obj.real), "im": jsonable(obj.imag)}
        if obj.dtype == bool:
            return obj.tolist()
        arr = obj.astype(float) if obj.dtype.kind in "fiu" else obj
        if arr.dtype.kind == "f":
            out = np.where(np.isfinite(arr), arr, np.nan).tolist()
            return _nan_to_none(out)
        return arr.tolist()
    if isinstance(obj, (complex, np.complexfloating)):
        return [_f(obj.real), _f(obj.imag)]
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "_mpf_") or hasattr(obj, "_mpc_"):
        return jsonable(complex(obj))
    return obj


def _f(x):
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _nan_to_none(v):
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# ------------------------------------------------------------------ analyses
def summary(spec):
    t0 = time.time()
    m = get_model(spec)
    s = m.summary()
    s["validation"] = validate.run_all(m)
    s["tiers"] = tiers(m, s)
    s["elapsed"] = time.time() - t0
    return jsonable(s)


def tiers(m, s):
    """Reliability label of each headline quantity (see the README)."""
    exact = "exact (symbolic)"
    t = {}
    t["tau"] = exact if s["tau_exact"] else ("given" if m.mode == "bp" else "numerical (tier 1)")
    if s["lambda_exact"]:
        t["lambda"] = exact
    elif m.tau_on_boundary:
        t["lambda"] = "numerical angular limit (tier 2)"
    else:
        t["lambda"] = "numerical (tier 1)"
    t["kind"] = s["kind_confidence"]
    if m.kind == "parabolic":
        st = s["step"] or {}
        t["step"] = st.get("confidence") if st.get("kind") else "numerical estimate only (tier 3)"
    elif m.kind == "hyperbolic":
        t["step"] = "theorem"
    t["backward_set"] = "outer approximation from finite-time backward integration (tier 2)"
    t["koenigs_geometry"] = "heuristic from finite radii (tier 3)"
    t["speeds_fits"] = "fits on a finite time window (tier 3)"
    return t


def dynamics(spec, T=None, field_n=21):
    m = get_model(spec)
    T = T or orbits.default_horizon(m)
    flows = visuals.flow_lines(m, T=T)
    out = {
        "T": T,
        "vector_field": visuals.vector_field(m, n=field_n),
        "phase_portrait": visuals.phase_portrait_png(m),
        "flows": flows,
        "images": visuals.images_of_disc(m, T=T),
        "halfplane": visuals.halfplane_view(m, flows),
    }
    return compact(jsonable(out), 5)


def step_kind(m):
    """Hyperbolic step for the petal cross-check: exact if available, else the numerical estimate."""
    if m.kind != "parabolic":
        return None
    s = m.summary()
    kind = (s.get("step") or {}).get("kind")
    if kind is None:
        try:
            kind = orbits.nonelliptic_asymptotics(m)["step_numeric"]["kind"]
        except Exception:  # noqa: BLE001 - the check is then reported as not applied
            kind = None
    return kind


def backward(spec, n=101):
    m = get_model(spec)
    bpts = m.summary()["boundary_points"]
    r = petals.backward_map(m, bpts, n=n, step_kind=step_kind(m))
    return compact(jsonable(r), 5)


def koenigs_view(spec, T=None):
    m = get_model(spec)
    d = koenigs.koenigs_data(m)
    K = d["K"]
    flows = visuals.flow_lines(m, T=T or orbits.default_horizon(m), n_out=80)
    def img(lines):
        res = []
        for x, y in lines:
            zz = np.array(x) + 1j * np.array(y)
            hz = K(zz) if zz.size else zz
            ok = np.isfinite(hz)
            res.append([hz[ok].real.tolist(), hz[ok].imag.tolist()])
        return res
    d["orbits"] = img(flows["forward"])
    d["backward_orbits"] = img(flows["backward"])
    # robust plotting window
    pts = np.concatenate([c["h"] for c in d["circles"][:-3]])
    pts = pts[np.isfinite(pts)]
    if pts.size:
        lo = np.percentile(pts.real, 1), np.percentile(pts.imag, 1)
        hi = np.percentile(pts.real, 99), np.percentile(pts.imag, 99)
        pad = 0.15 * max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
        d["window"] = [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad]
    d["kind"] = m.kind
    return compact(jsonable(d), 6)


def asymptotics(spec, z0=0j, T=None):
    m = get_model(spec)
    if m.kind == "elliptic":
        z_start = z0 if abs(z0 - m.tau) > 1e-9 else None
        r = orbits.elliptic_asymptotics(m, z0=z_start if z_start is not None else 0.5, T=T)
        r["kind"] = "elliptic"
    else:
        r = orbits.nonelliptic_asymptotics(m, z0=z0, T=T)
        r["kind"] = m.kind
        r["step_exact"] = m.step
        r["lambda"] = m.lam
        # downsample the half-plane orbit for transport
    return jsonable(r)


def point_orbits(spec, z0, T=None):
    m = get_model(spec)
    if abs(z0) >= 1:
        raise ValueError("The point must lie in the open unit disc.")
    T = T or orbits.default_horizon(m)
    bpts = m.summary()["boundary_points"]
    tf, zf = orbits.forward_orbit_disk(m, z0, T)
    fwd = {"t": tf, "z": zf, "k_from_start": orbits.k_D(z0, zf)}
    if m.kind != "elliptic":
        fwd["dist_tau"] = np.abs(zf - m.tau)
    else:
        fwd["k_to_tau"] = orbits.k_D(m.tau, zf)
    bwd = petals.backward_orbit(m, z0, bpts)
    # thin out long paths
    for d in (fwd, bwd):
        n = len(d["t"])
        if n > 1500:
            idx = np.unique(np.linspace(0, n - 1, 1500).astype(int))
            for k in ("t", "z", "k_from_start", "dist_tau", "k_to_tau"):
                if k in d:
                    d[k] = np.asarray(d[k])[idx]
    return compact(jsonable({"z0": z0, "T": T, "forward": fwd, "backward": bwd}), 8)
