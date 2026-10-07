"""Built-in examples with known answers (also used as regression tests)."""

LIBRARY = [
    {
        "slug": "rotation-group",
        "name": "Rotation group",
        "spec": {"mode": "generator", "G": "I*z"},
        "description": "phi_t(z) = e^{it} z. Elliptic group: every point has a backward orbit.",
        "expected": {"kind": "elliptic", "group": True, "tau": [0, 0], "lambda": [0, -1]},
    },
    {
        "slug": "elliptic-contraction",
        "name": "Elliptic contraction",
        "spec": {"mode": "semigroup", "phi": "exp(-t)*z"},
        "description": "phi_t(z) = e^{-t} z. Backward invariant set W = {0}.",
        "expected": {"kind": "elliptic", "group": False, "tau": [0, 0], "lambda": [1, 0],
                     "repelling": []},
    },
    {
        "slug": "koebe-elliptic",
        "name": "Elliptic, Koebe Koenigs function",
        "spec": {"mode": "generator", "G": "-z*(1-z)/(1+z)"},
        "description": "Koenigs function h(z) = z/(1-z)^2, Koenigs domain C \\ (-inf, -1/4]. "
                       "Repelling boundary fixed point 1 with spectral value 1/2; one hyperbolic "
                       "petal D \\ (-1, 0].",
        "expected": {"kind": "elliptic", "group": False, "tau": [0, 0], "lambda": [1, 0],
                     "repelling": [{"sigma": [1, 0], "beta": 0.5}]},
    },
    {
        "slug": "elliptic-spiral",
        "name": "Elliptic, spiralling",
        "spec": {"mode": "bp", "tau": "0", "p": "(1-z)/(1+z) + I"},
        "description": "lambda = 1 + i: orbits spiral into 0. Boundary fixed point at z = i.",
        "expected": {"kind": "elliptic", "group": False, "tau": [0, 0], "lambda": [1, 1]},
    },
    {
        "slug": "elliptic-off-centre",
        "name": "Elliptic, tau = 1/2",
        "spec": {"mode": "bp", "tau": "1/2", "p": "(1+z)/(1-z)"},
        "description": "Interior Denjoy–Wolff point 1/2, lambda = 9/4, repelling point -1 with beta = 9/8.",
        "expected": {"kind": "elliptic", "group": False, "tau": [0.5, 0], "lambda": [2.25, 0],
                     "repelling": [{"sigma": [-1, 0], "beta": 1.125}]},
    },
    {
        "slug": "lft-elliptic",
        "name": "Linear fractional elliptic (phi_t given)",
        "spec": {"mode": "semigroup", "phi": "exp(-t)*z/(1-(1-exp(-t))*z)"},
        "description": "G(z) = -z(1-z); repelling boundary fixed point 1 with beta = 1.",
        "expected": {"kind": "elliptic", "group": False, "tau": [0, 0], "lambda": [1, 0],
                     "repelling": [{"sigma": [1, 0], "beta": 1.0}]},
    },
    {
        "slug": "hyperbolic-affine",
        "name": "Hyperbolic, no repelling points",
        "spec": {"mode": "semigroup", "phi": "1-(1-z)*exp(-2t)"},
        "description": "G(z) = 2(1-z), lambda = 2. No backward orbits in D.",
        "expected": {"kind": "hyperbolic", "group": False, "tau": [1, 0], "lambda": [2, 0],
                     "repelling": [], "step": "positive"},
    },
    {
        "slug": "hyperbolic-group",
        "name": "Hyperbolic group",
        "spec": {"mode": "generator", "G": "1-z^2"},
        "description": "phi_t(z) = tanh(t + artanh z): hyperbolic automorphisms, lambda = 2.",
        "expected": {"kind": "hyperbolic", "group": True, "tau": [1, 0], "lambda": [2, 0],
                     "repelling": [{"sigma": [-1, 0], "beta": 2.0}], "step": "positive"},
    },
    {
        "slug": "hyperbolic-petal",
        "name": "Hyperbolic with a petal",
        "spec": {"mode": "generator", "G": "(1-z^2)*(3-z)/4"},
        "description": "Gamma(w) = w(w+2)/(w+1) in the half-plane. lambda = 1, repelling point -1 "
                       "with beta = 2 and a hyperbolic petal.",
        "expected": {"kind": "hyperbolic", "group": False, "tau": [1, 0], "lambda": [1, 0],
                     "repelling": [{"sigma": [-1, 0], "beta": 2.0}], "step": "positive"},
    },
    {
        "slug": "hyperbolic-bp-rotated",
        "name": "Hyperbolic, tau = i (Berkson–Porta)",
        "spec": {"mode": "bp", "tau": "I", "p": "1 + (I+z)/(I-z)"},
        "description": "Gamma(w) = 2 + 2w, lambda = 2, no repelling points.",
        "expected": {"kind": "hyperbolic", "group": False, "tau": [0, 1], "lambda": [2, 0],
                     "repelling": [], "step": "positive"},
    },
    {
        "slug": "parabolic-zero-step",
        "name": "Parabolic, zero hyperbolic step",
        "spec": {"mode": "generator", "G": "(z-1)^2/2"},
        "description": "Gamma(w) = 1: phi_t(w) = w + t in H. Orbits converge orthogonally to the boundary.",
        "expected": {"kind": "parabolic", "group": False, "tau": [1, 0], "lambda": [0, 0],
                     "repelling": [], "step": "zero"},
    },
    {
        "slug": "parabolic-positive-step",
        "name": "Parabolic, positive hyperbolic step",
        "spec": {"mode": "generator", "G": "(z-1)^2*(I/2 + (1-z)/4)"},
        "description": "Gamma(w) = i + 1/(w+1): orbits are asymptotically parallel to dH; "
                       "there is a parabolic petal.",
        "expected": {"kind": "parabolic", "group": False, "tau": [1, 0], "lambda": [0, 0],
                     "repelling": [], "step": "positive"},
    },
    {
        "slug": "parabolic-group",
        "name": "Parabolic group",
        "spec": {"mode": "generator", "G": "I*(z-1)^2"},
        "description": "Parabolic automorphisms (Gamma(w) = 2i).",
        "expected": {"kind": "parabolic", "group": True, "tau": [1, 0], "lambda": [0, 0],
                     "step": "positive"},
    },
    {
        "slug": "parabolic-sqrt",
        "name": "Parabolic, non-rational generator",
        "spec": {"mode": "bp", "tau": "1", "p": "sqrt((1+z)/(1-z))"},
        "description": "Gamma(w) = 2 sqrt(w): zero step; -1 is a non-regular (super-repelling) "
                       "boundary fixed point.",
        "expected": {"kind": "parabolic", "group": False, "tau": [1, 0], "lambda": [0, 0],
                     "step": "zero", "super_repelling": [[-1, 0]]},
    },
]


def by_slug(slug):
    for e in LIBRARY:
        if e["slug"] == slug:
            return e
    return None
