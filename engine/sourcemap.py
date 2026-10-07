"""Named references from the documentation to the functions that implement each step.

The documentation page shows the live source of these functions, so the
explanation and the implementation cannot drift apart silently.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

# key -> (module, qualified name)
SOURCES = {
    "parse": (".expr", "parse"),
    "cancel_rational_parts": (".expr", "cancel_rational_parts"),
    "model_init": (".model", "SemigroupModel._build"),
    "detect_group": (".model", "SemigroupModel._detect_group"),
    "interior_zeros": (".model", "SemigroupModel._interior_zeros"),
    "find_tau": (".model", "SemigroupModel._find_tau"),
    "spectral_value": (".model", "SemigroupModel._spectral_value"),
    "lambda_from_gamma": (".model", "SemigroupModel._lambda_from_gamma"),
    "normal_model": (".model", "SemigroupModel._build_normal_model"),
    "classify": (".model", "SemigroupModel._classify"),
    "step_exact": (".model", "SemigroupModel._step_exact"),
    "boundary_find": (".boundary", "find_boundary_null_points"),
    "beta_halfplane": (".boundary", "beta_via_halfplane"),
    "radial_beta": (".boundary", "radial_beta"),
    "classify_boundary": (".boundary", "classify_boundary_points"),
    "integrate": (".integrate", "integrate"),
    "classify_backward": (".orbits", "classify_backward"),
    "backward_map": (".petals", "backward_map"),
    "backward_orbit": (".orbits", "backward_orbit"),
    "koenigs": (".koenigs", "Koenigs"),
    "koenigs_geometry": (".koenigs", "omega_geometry"),
    "k_H": (".orbits", "k_H"),
    "asym_nonelliptic": (".orbits", "nonelliptic_asymptotics"),
    "step_numeric": (".orbits", "numeric_step_verdict"),
    "asym_elliptic": (".orbits", "elliptic_asymptotics"),
    "generator_inequality": (".validate", "generator_inequality"),
    "berkson_porta_check": (".validate", "berkson_porta"),
    "holomorphy": (".validate", "holomorphy"),
    "branch_cut_scan": (".validate", "branch_cut_scan"),
    "semigroup_law": (".validate", "semigroup_law"),
    "vector_field": (".visuals", "vector_field"),
    "phase_portrait": (".visuals", "phase_portrait_png"),
    "flow_lines": (".visuals", "flow_lines"),
    "images_of_disc": (".visuals", "images_of_disc"),
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(key: str) -> dict:
    if key not in SOURCES:
        raise KeyError(key)
    module_name, qualname = SOURCES[key]
    obj = importlib.import_module(module_name, package=__package__)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    lines, start = inspect.getsourcelines(obj)
    path = Path(inspect.getsourcefile(obj)).resolve()
    try:
        rel = str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        rel = path.name
    return {"key": key, "file": rel, "qualname": qualname, "start_line": start,
            "end_line": start + len(lines) - 1, "source": "".join(lines)}
