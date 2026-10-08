# Semigroup Lab

An analysis workbench for continuous semigroups (φₜ) of holomorphic self-maps of the unit disc, built
with Streamlit.

Give the infinitesimal generator G, Berkson–Porta data (τ, p), or a closed form φₜ(z), and the app
reports:

* the Denjoy–Wolff point and the spectral value;
* the type (elliptic, hyperbolic or parabolic) and the hyperbolic step;
* boundary fixed points, the backward invariant set and petals, cross-checked against the theorems;
* the Koenigs domain, and speeds and rates of convergence;
* interactive pictures.

Every result says how far it can be trusted (exact, numerical, numerical limit or numerical estimate).
A "How it works" page explains the theory, algorithm and thresholds behind each result, and shows the
code that computes it.

There are no accounts or database. The input is kept in the address bar, so a bookmark or a shared link
reopens an analysis. The results can be downloaded as JSON.

## Run it locally

Requires Python 3.12 or 3.13 (the pinned versions in `requirements.txt` were tested on both). The app also
runs without warnings on any Streamlit from 1.46 on; `ui/compat.py` adapts to the differences between
versions. Older versions use fallback fonts.

```sh
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py       # opens http://localhost:8501
```

## Put it on GitHub

Create an empty repository on GitHub (for example `semigroup-lab`), then from this folder:

```sh
git init -b main
git add .
git commit -m "Semigroup Lab"
git remote add origin https://github.com/<your-user>/semigroup-lab.git
git push -u origin main
```

If you use GitHub's "upload files" page instead, include the hidden `.streamlit` folder (it holds the
colours and fonts). On macOS, press Cmd+Shift+. in Finder to show hidden files. The app also runs
without that folder, in Streamlit's default theme.

## Host it on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Choose **Create app**, then **Deploy a public app from GitHub**.
3. Pick the repository and the `main` branch, and set the main file path to `streamlit_app.py`.
4. Under **Advanced settings**, choose **Python 3.12**. The pinned versions in `requirements.txt` were
   tested on it. No secrets are needed.
5. Optionally choose a custom subdomain, then **Deploy**. The first build installs the requirements and
   takes a few minutes; later pushes to `main` redeploy automatically.

Once it is live, you can add a badge to this README:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-subdomain>.streamlit.app)
```

### Hosting notes

* **Memory:** about 250 MB with several users. The caches are bounded (`max_entries` in `ui/compute.py`
  and the model cache in `engine/report.py`), well inside Community Cloud's limits.
* **Speed:** a first analysis takes about 1–4 s, and repeating it is instant because each step is cached
  on its input and its own parameters. Many users analysing new inputs at the same moment share one CPU,
  so they queue: six simultaneous cold analyses finished within about 45 s in testing.
* **Precision across users:** mpmath keeps its working precision in one global setting. Each Streamlit
  session runs in its own thread, so engine code that uses it holds a lock (`engine/precision.py`).
  Without the lock, one user's analysis could change the precision of another's.
* **Hostile input:** expressions are checked against a whitelist of names before SymPy parses them.
  Numeric exponents above 100 and integers longer than 50 digits are rejected before SymPy evaluates
  anything, since inputs like `9^9^9^9` would otherwise hang a worker.
* **Sleeping:** Community Cloud puts apps to sleep after a period without visitors. The first visitor
  after that wakes the app up, which takes a short while.

## Using the app

* **Input** (sidebar): choose Generator, Berkson–Porta or Semigroup mode, type the expression and press
  **Analyse**. Or pick one of the 15 library examples, which all have known answers.
* **Parameters** (sidebar): the picture horizon, the resolution of the backward map, the asymptotics
  horizon, and the base point z₀ for the speeds. The backward map needs no horizon: every orbit is
  followed until it escapes or provably converges.
* **Disc:** toggle the layers:
  * flow lines and the vector field;
  * the phase portrait of G;
  * φₜ(D), with a time slider;
  * backward escape time, petals and fixed points.

  Click any point to see its forward and backward orbit, or type it into the point inspector.
* **Tabs:** Koenigs domain, half-plane model, speeds and rates, boundary fixed points and petals, and
  method details (including the JSON download).
* **How it works:** every result has a "How?" link to the matching section of the documentation.

**Input language.** The variable is `z` (plus `t` for φₜ). The functions are `exp`, `log`, `sqrt`,
`sin`, `cos`, `tan` and their hyperbolic and inverse versions, all on principal branches. The constants
are `I` (or `i`), `pi` and `E`. Both `^` and implicit multiplication (`2z^2`) work, and decimals are read
as exact rationals.

## Project layout

```
streamlit_app.py        entry point: page config and navigation
views/analyse.py        the analysis page
views/docs.py           the documentation page (renders docs.md)
docs.md                 theory, algorithms, thresholds, expected accuracy
ui/compute.py           cached wrappers around the engine
ui/plots.py             Plotly figures
ui/fmt.py               number formatting, reliability badges, complex-number input
ui/compat.py            passes each widget only the arguments the installed Streamlit supports
engine/                 the mathematics (SymPy, mpmath, NumPy, SciPy); no Streamlit inside
tests/                  local test suite (not run on GitHub)
.streamlit/config.toml  theme
requirements.txt        runtime dependencies, pinned (used by Community Cloud)
requirements-dev.txt    adds pytest for the local tests
```

## Tests (local, optional)

```sh
pip install -r requirements-dev.txt
pytest
```

There are 78 tests, taking about 80 s:

* the engine against library examples with known answers;
* petals against exact answers (two half-disc petals, the Koebe slit), under rescaling of time, and
  against an independent reference implementation (`tests/reference_petals.py`, SciPy DOP853 with
  event location);
* both pages end to end with Streamlit's AppTest;
* consistency between the documentation and the code;
* the protections described under hosting notes;
* a check that no Streamlit deprecation warning is raised.

Nothing runs automatically on GitHub.

## Known limits

Anything about behaviour at ∂D or as t → ∞ is sampled. That covers the validity checks, petals, the shape
of Ω, the numerical hyperbolic step and all fits, so treat these as evidence rather than proof. Only exact
computations can tell λ = 0 apart from a tiny λ > 0. The documentation's "Known limits" section has the
full list.
