"""The library examples have known answers; every analysis is checked against them."""
import json

import numpy as np
import pytest

from engine import koenigs, orbits, petals, report
from engine.expr import ExpressionError, parse
from engine.library import LIBRARY, by_slug
from engine.model import ModelError, SemigroupModel

IDS = [e["slug"] for e in LIBRARY]


@pytest.mark.parametrize("ex", LIBRARY, ids=IDS)
def test_summary_matches_known_answers(ex):
    s = report.summary(ex["spec"])
    exp = ex["expected"]
    assert s["kind"] == exp["kind"]
    assert s["group"] == exp["group"]
    assert np.allclose(s["tau"], exp["tau"], atol=1e-9)
    assert np.allclose(s["lambda"], exp["lambda"], atol=1e-6)
    if "step" in exp:
        num = report.asymptotics(ex["spec"])
        exact_kind = (s["step"] or {}).get("kind")
        numeric_kind = num["step_numeric"]["kind"]
        assert exp["step"] in (exact_kind, numeric_kind)
        if exact_kind is not None:
            assert exact_kind == exp["step"]
        assert numeric_kind == exp["step"]
    if "repelling" in exp:
        rep = [b for b in s["boundary_points"] if b["kind"] == "repelling"]
        assert len(rep) == len(exp["repelling"])
        for want in exp["repelling"]:
            match = [b for b in rep if np.allclose(b["sigma"], want["sigma"], atol=1e-8)]
            assert match, f"missing repelling point {want}"
            assert abs(match[0]["beta"][0] - want["beta"]) < 1e-6
    if "super_repelling" in exp:
        sup = [b for b in s["boundary_points"] if b["kind"].startswith("super")]
        assert len(sup) == len(exp["super_repelling"])
    assert s["validation"]["ok"], s["validation"]
    json.dumps(s, allow_nan=False)


def test_speeds_hyperbolic_rate_is_half_lambda():
    a = report.asymptotics(by_slug("hyperbolic-petal")["spec"])
    assert abs(a["fits"]["v_orth_slope"]["value"] - 0.5) < 1e-3
    assert abs(a["fits"]["rate_dist"]["value"] - 1.0) < 1e-3


def test_parabolic_rates():
    zero = report.asymptotics(by_slug("parabolic-zero-step")["spec"])
    assert abs(zero["fits"]["dist_power"]["value"] - 1) < 0.02
    assert not zero["slope_tail"]["tangential"]
    pos = report.asymptotics(by_slug("parabolic-positive-step")["spec"])
    assert pos["slope_tail"]["tangential"]


def test_koebe_koenigs_function():
    m = SemigroupModel(by_slug("koebe-elliptic")["spec"])
    K = koenigs.Koenigs(m)
    zz = np.array([0.3 + 0.2j, -0.5j, 0.9, -0.8 + 0.1j])
    assert np.max(np.abs(K(zz) - zz / (1 - zz) ** 2)) < 1e-10


def test_hyperbolic_strip_width():
    m = SemigroupModel(by_slug("hyperbolic-group")["spec"])
    g = koenigs.koenigs_data(m)["geometry"]
    assert g["shape"] == "vertical strip"
    assert abs(g["lambda_from_width"] - 2) < 1e-3


@pytest.mark.parametrize("slug,shape", [("parabolic-zero-step", "no half-plane"),
                                        ("parabolic-positive-step", "vertical half-plane"),
                                        ("parabolic-sqrt", "no half-plane")])
def test_parabolic_koenigs_shapes(slug, shape):
    m = SemigroupModel(by_slug(slug)["spec"])
    assert koenigs.koenigs_data(m)["geometry"]["shape"] == shape


def test_koebe_petal_is_disc_minus_slit():
    m = SemigroupModel(by_slug("koebe-elliptic")["spec"])
    r = petals.backward_map(m, m.summary()["boundary_points"], n=61, T=60)
    assert len(r["petals"]) == 1
    p = r["petals"][0]
    assert p["kind"] == "hyperbolic petal" and abs(p["alpha_point"] - 1) < 1e-9
    assert abs(p["beta_estimate_median"] - 0.5) < 1e-2
    g = r["grid"]
    codes = r["codes"]
    row = np.argmin(np.abs(g))  # the real axis
    neg = [codes[row, j] for j in range(len(g)) if -0.95 < g[j] < -0.05]
    assert all(c == 0 for c in neg), "points of (-1, 0) must escape"


def test_parabolic_petal_and_empty_W():
    m = SemigroupModel(by_slug("parabolic-positive-step")["spec"])
    r = petals.backward_map(m, m.summary()["boundary_points"], n=61, T=60)
    assert [p["kind"] for p in r["petals"]] == ["parabolic petal"]
    m = SemigroupModel(by_slug("parabolic-zero-step")["spec"])
    r = petals.backward_map(m, m.summary()["boundary_points"], n=61, T=60)
    assert r["petals"] == []


def test_backward_orbit_of_point_in_petal():
    m = SemigroupModel(by_slug("hyperbolic-petal")["spec"])
    b = orbits.backward_orbit(m, -0.3 + 0j, 60, m.summary()["boundary_points"])
    assert b["code"] == 1 and abs(b["sigma"] + 1) < 1e-9
    assert abs(b["beta_est"] - 2) < 0.05


@pytest.mark.parametrize("spec,msg", [
    ({"mode": "generator", "G": "z"}, None),                      # repelling at 0: not a generator
    ({"mode": "generator", "G": "-z*exp(3*z)"}, None),         # p = exp(3z) has Re p < 0 somewhere
    ({"mode": "bp", "tau": "1", "p": "-1"}, None),                 # Re p < 0
])
def test_invalid_generators_are_flagged(spec, msg):
    s = report.summary(spec)
    assert not s["validation"]["ok"]


def test_model_errors():
    with pytest.raises(ModelError):
        SemigroupModel({"mode": "generator", "G": "1/(z-1/2)"})   # pole in D
    with pytest.raises(ModelError):
        SemigroupModel({"mode": "generator", "G": "z^2-1/4"})     # two zeros in D
    with pytest.raises(ModelError):
        SemigroupModel({"mode": "bp", "tau": "2", "p": "1"})


@pytest.mark.parametrize("text", ["__import__('os')", "z.real", "lambda: 1", "open(1)", "z; 1", "x+1"])
def test_parser_rejects_unsafe_input(text):
    with pytest.raises(ExpressionError):
        parse(text)


def test_parser_niceties():
    assert parse("2z^2 + 0.5") == parse("2*z**2 + 1/2")
    assert parse("i z") == parse("I*z")


def test_branch_cut_detected():
    s = report.summary({"mode": "generator", "G": "-z*sqrt(z+1/4)"})
    assert not s["validation"]["holomorphy"]["ok"]


def test_semigroup_mode_residuals():
    s = report.summary(by_slug("lft-elliptic")["spec"])
    sg = s["validation"]["semigroup_law"]
    assert sg["closed_form_semigroup_residual"] < 1e-12
    assert sg["closed_form_vs_flow"] < 1e-8


def test_visual_endpoints_are_json():
    spec = by_slug("hyperbolic-petal")["spec"]
    for fn in (report.dynamics, report.koenigs_view):
        json.dumps(fn(spec), allow_nan=False)
    json.dumps(report.backward(spec, n=41), allow_nan=False)
    json.dumps(report.point_orbits(spec, 0.2 + 0.3j), allow_nan=False)
