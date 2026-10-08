"""Petals: exact answers, invariance under time rescaling, and an independent reference."""
import numpy as np
import pytest

from engine import petals
from engine.library import by_slug
from engine.model import SemigroupModel

from reference_petals import reference_labels


def _map(spec, n=61, step=None):
    m = SemigroupModel(spec)
    return m, petals.backward_map(m, m.summary()["boundary_points"], n=n, step_kind=step)


def test_two_petals_split_by_a_slanted_diameter_are_not_merged():
    """h(z) = z/(1 − (cz)²), c = e^{−iπ/8}: the petals are the half-discs Re(c z) > 0 and < 0."""
    m, r = _map(by_slug("elliptic-two-petals")["spec"], n=61)
    real = [p for p in r["petals"] if p["cells"]]
    assert len(real) == 2
    g = r["grid"]
    Z = g[None, :] + 1j * g[:, None]
    c = np.exp(-1j * np.pi / 8)
    lab = r["labels"]
    sigmas = [complex(p["alpha_point"]) for p in real]
    k_plus = int(np.argmin([abs(s - 1 / c) for s in sigmas]))     # σ = e^{iπ/8}
    side = (c * Z).real
    spacing = g[1] - g[0]
    clear = (lab != -9) & (np.abs(side) > spacing)                # away from the dividing diameter
    want = np.where(side > 0, real[k_plus]["key"], real[1 - k_plus]["key"])
    assert np.mean(lab[clear] == want[clear]) > 0.999
    assert all(c_["ok"] for c_ in r["checks"]), r["checks"]


@pytest.mark.parametrize("c", ["1/100", "100"])
def test_petals_do_not_depend_on_the_speed_of_time(c):
    """G and cG (c > 0) have the same orbits, hence the same W and petals."""
    _, base = _map({"mode": "generator", "G": "(1-z^2)*(3-z)/4"}, n=41)
    _, fast = _map({"mode": "generator", "G": f"({c})*(1-z^2)*(3-z)/4"}, n=41)
    same = base["labels"] == fast["labels"]
    assert np.mean(same[base["labels"] != -9]) > 0.998
    rate = [p for p in fast["petals"] if p["cells"]][0]["beta_estimate_median"]
    beta = [p for p in fast["petals"] if p["cells"]][0]["beta"]
    assert abs(rate - beta) / beta < 0.05


@pytest.mark.parametrize("slug,step", [("hyperbolic-petal", None), ("parabolic-positive-step", "positive")])
def test_agrees_with_the_independent_reference(slug, step):
    m, r = _map(by_slug(slug)["spec"], n=31, step=step)
    g, inside, ref = reference_labels(m, n=31)
    lab = r["labels"].copy()
    lab[(lab <= -4) & (lab != -9)] = -2
    assert np.mean(lab[inside] == ref[inside]) > 0.995


def test_strongly_repelling_point_keeps_its_petal():
    """β = 200: the previous version lost this petal entirely."""
    m, r = _map({"mode": "generator", "G": "(1-z^2)*(3/2-z)/(1/40)"}, n=41)
    real = [p for p in r["petals"] if p["cells"]]
    assert len(real) == 1 and real[0]["area_fraction"] > 0.3
    assert abs(real[0]["beta_estimate_median"] - 200) / 200 < 0.05


def test_rotation_group_has_W_equal_D():
    _, r = _map(by_slug("rotation-group")["spec"], n=31)
    assert r["petals"][0]["kind"].startswith("whole disc") and r["petals"][0]["area_fraction"] == 1.0


@pytest.mark.parametrize("slug", ["koebe-elliptic", "elliptic-off-centre", "elliptic-spiral", "hyperbolic-group",
                                  "hyperbolic-affine", "parabolic-group", "parabolic-sqrt"])
def test_no_theorem_check_fails_on_library_examples(slug):
    m = SemigroupModel(by_slug(slug)["spec"])
    step = (m.summary()["step"] or {}).get("kind")
    r = petals.backward_map(m, m.summary()["boundary_points"], n=41, step_kind=step)
    assert all(c["ok"] for c in r["checks"]), [c["text"] for c in r["checks"] if not c["ok"]]


def test_checks_flag_a_missing_parabolic_petal():
    """Positive step but the parabolic petal removed: the cross-check must complain."""
    m = SemigroupModel(by_slug("parabolic-positive-step")["spec"])
    checks = petals.petal_checks(m, [], {}, 100, step_kind="positive")
    assert any(not c["ok"] and "must exist" in c["text"] for c in checks)
