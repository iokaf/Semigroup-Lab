"""Analysis page: input in the sidebar, results in the main area."""
from __future__ import annotations

import hashlib
import html
import json

import streamlit as st

from engine.library import LIBRARY, by_slug
from ui import compat, compute, plots
from ui.fmt import badge, cnum, md_escape_cell, num, parse_complex

MODES = {"generator": "Generator G", "bp": "Berkson–Porta (τ, p)", "semigroup": "Semigroup φₜ"}
SPEC_FIELDS = ("G", "tau", "p", "phi")
DEFAULT_SLUG = "hyperbolic-petal"
STATE_DEFAULTS = {
    "in_mode": "generator", "in_G": "(1-z^2)*(3-z)/4", "in_tau": "1", "in_p": "(1+z)/(1-z)",
    "in_phi": "exp(-t)*z/(1-(1-exp(-t))*z)",
    "prm_Tvis": 0.0, "prm_bwn": 101, "prm_bwT": 60.0, "prm_asT": 0.0, "prm_z0": "0",
    "inspect": None, "last_sel": None, "insp_text": "",
}
KIND_COLOR = {"elliptic": "#0E8A6E", "hyperbolic": "#2747C7", "parabolic": "#C0266D"}

CSS = """
<style>
.verdict { font-family: Spectral, Georgia, serif; font-weight: 400; font-size: 2.05rem; line-height: 1.2;
           margin: 0 0 .4rem; padding: 0; }
.verdict b { font-weight: 600; }
.st-key-facts { border-top: 1px solid #D9DEE7; border-bottom: 1px solid #D9DEE7; padding: .6rem 0 .2rem; }
.st-key-facts [data-testid="stMarkdownContainer"] p { font-family: Spectral, Georgia, serif; font-size: 1.3rem; }
.st-key-facts [data-testid="stCaptionContainer"] p { font-family: inherit; font-size: .82rem; }
.st-key-checks [data-testid="stMarkdownContainer"] p { font-size: .88rem; line-height: 1.45; }
a.how { font-size: .78rem; font-weight: 400; margin-left: .35rem; text-decoration: underline; }
dl.insp { display: grid; grid-template-columns: auto 1fr; gap: .2rem .8rem; margin: .2rem 0 .6rem; font-size: .92rem; }
dl.insp dt { color: #7A8496; }
dl.insp dd { margin: 0; font-family: Spectral, Georgia, serif; }
</style>
"""


def how(anchor: str, text: str = "How?") -> str:
    """Link into the documentation page (opens in a new tab so the analysis stays put)."""
    return f'<a class="how" href="docs#{anchor}" target="_blank">{html.escape(text)}</a>'


# ------------------------------------------------------------------ state
def _persist_inputs():
    # widgets that are not drawn in a run lose their value; re-assigning keeps them
    for k in STATE_DEFAULTS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _fill(spec: dict):
    st.session_state.in_mode = spec.get("mode", "generator")
    for f in SPEC_FIELDS:
        if spec.get(f) is not None:
            st.session_state[f"in_{f}"] = spec[f]


def _read_spec() -> dict:
    mode = st.session_state.in_mode
    spec = {"mode": mode}
    if mode == "generator":
        spec["G"] = st.session_state.in_G.strip()
    elif mode == "bp":
        spec["tau"] = st.session_state.in_tau.strip()
        spec["p"] = st.session_state.in_p.strip()
    else:
        spec["phi"] = st.session_state.in_phi.strip()
    return spec


def _read_params() -> dict:
    ss = st.session_state
    return {"T_vis": float(ss.prm_Tvis) or None, "bw_n": int(ss.prm_bwn), "bw_T": float(ss.prm_bwT),
            "as_T": float(ss.prm_asT) or None, "z0": ss.prm_z0.strip() or "0"}


def _sync_url(spec: dict):
    st.query_params.from_dict({k: v for k, v in spec.items() if v})


def _init():
    for k, v in STATE_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if "spec" in st.session_state:
        return
    qp = st.query_params
    if qp.get("lib") and by_slug(qp.get("lib")):
        _fill(by_slug(qp.get("lib"))["spec"])
    elif qp.get("mode") in MODES:
        _fill({"mode": qp.get("mode"), **{f: qp.get(f) for f in SPEC_FIELDS if qp.get(f)}})
    else:
        _fill(by_slug(DEFAULT_SLUG)["spec"])
    st.session_state.spec = _read_spec()
    st.session_state.params = _read_params()


def _new_analysis():
    st.session_state.inspect = None
    st.session_state.last_sel = None
    st.session_state.insp_text = ""


def _on_library():
    e = by_slug(st.session_state.lib)
    if not e:
        return
    _fill(e["spec"])
    st.session_state.spec = _read_spec()
    _new_analysis()
    _sync_url(st.session_state.spec)


def _on_analyse():
    st.session_state.spec = _read_spec()
    st.session_state.params = _read_params()
    st.session_state.lib = ""
    _new_analysis()
    _sync_url(st.session_state.spec)


def _on_inspect_typed():
    try:
        z = parse_complex(st.session_state.insp_text)
    except ValueError as exc:
        st.session_state.insp_error = str(exc)
        return
    if abs(z) >= 1:
        st.session_state.insp_error = "The point must lie in the open unit disc."
        return
    st.session_state.insp_error = None
    st.session_state.inspect = (z.real, z.imag)


# ------------------------------------------------------------------ sidebar
def sidebar():
    with st.sidebar:
        st.selectbox("Load a library example", [""] + [e["slug"] for e in LIBRARY], key="lib",
                     format_func=lambda s: "Choose…" if not s else by_slug(s)["name"], on_change=_on_library)
        st.radio("Input", list(MODES), format_func=MODES.get, key="in_mode")
        with st.form("input", border=False):
            mode = st.session_state.in_mode
            if mode == "generator":
                st.text_input("G(z) =", key="in_G")
                st.caption("∂ₜφₜ = G(φₜ). Variable z; functions exp, log, sqrt, sin, … (principal branches); "
                           "constants I, pi, E.")
            elif mode == "bp":
                st.text_input("τ =", key="in_tau")
                st.text_input("p(z) =", key="in_p")
                st.caption("G(z) = (z − τ)(τ̄z − 1) p(z) with |τ| ≤ 1 and Re p ≥ 0.")
            else:
                st.text_input("φₜ(z) =", key="in_phi")
                st.caption("Variables z and t. The generator is ∂ₜφₜ at t = 0; the closed form is checked "
                           "against the flow.")
            with st.expander("Parameters"):
                st.number_input("Picture horizon T (0 = automatic)", min_value=0.0, max_value=500.0, step=1.0,
                                key="prm_Tvis")
                st.number_input("Backward map resolution", min_value=21, max_value=201, step=10, key="prm_bwn")
                st.number_input("Backward horizon", min_value=1.0, max_value=1000.0, step=10.0, key="prm_bwT")
                st.number_input("Asymptotics horizon (0 = automatic)", min_value=0.0, max_value=1e6, step=10.0,
                                key="prm_asT")
                st.text_input("Base point z₀ for asymptotics", key="prm_z0")
            compat.form_submit_button("Analyse", type="primary", on_click=_on_analyse)
        st.caption("The address bar holds the input, so a bookmark or shared link reopens this analysis.")
        return st.container(key="checks")


def render_checks(box, s):
    v = s["validation"]
    items = [
        (v["generator_inequality"]["ok"], "Generator criterion: " + v["generator_inequality"]["text"]
         + (f" (max relative excess {num(v['generator_inequality']['max_violation'], 2)})"
            if v["generator_inequality"].get("max_violation") is not None else "")),
        (v["berkson_porta"]["ok"], "Berkson–Porta: " + (v["berkson_porta"].get("text") or "")
         + (f" (min Re p/|p| = {num(v['berkson_porta']['min_re_p_relative'], 3)})"
            if v["berkson_porta"].get("min_re_p_relative") is not None else "")),
        (v["holomorphy"]["ok"], "Holomorphy: " + v["holomorphy"]["text"]),
    ]
    sg = v["semigroup_law"]
    if sg.get("closed_form_semigroup_residual") is not None:
        t = (f"Closed form: semigroup law residual {num(sg['closed_form_semigroup_residual'], 2)}, distance to the "
             f"flow of G {num(sg['closed_form_vs_flow'], 2)}, max |φₜ| {num(sg['closed_form_max_abs'], 6)}")
    else:
        t = f"Flow consistency |φₜ₊ₛ − φₜ∘φₛ| = {num(sg['numerical_flow_residual'], 2)}"
    items.append((sg["ok"], t))
    with box:
        st.markdown(f"**Checks** {how('validation')}", unsafe_allow_html=True)
        for ok, text in items:
            mark = ":green[**✓**]" if ok else ":red[**✕**]"
            st.markdown(f"{mark} {md_escape_cell(text).replace('*', '∗')}")


# ------------------------------------------------------------------ headline
def step_verdict(s, asym):
    if s["kind"] == "elliptic":
        return None, None
    ex = s.get("step") or {}
    if ex.get("kind"):
        return ex["kind"], ex.get("confidence")
    sn = (asym or {}).get("step_numeric") or {}
    if sn.get("kind"):
        return sn["kind"], "numerical estimate (tier 3)"
    return None, "undecided"


def render_headline(s, asym):
    kind = s["kind"]
    word = kind.capitalize() + (" group" if s["group"] else " semigroup")
    if kind == "elliptic":
        rest = (" of rotations about τ" if s["group"] else
                ", orbits spiral into τ" if abs(s["lambda"][1]) > 1e-12 else ", orbits converge to τ ∈ D")
    elif kind == "hyperbolic":
        rest = " of hyperbolic automorphisms" if s["group"] else ""
    else:
        k, _ = step_verdict(s, asym)
        rest = f" of {k} hyperbolic step" if k else ", hyperbolic step undecided"
    nrep = sum(1 for b in s["boundary_points"] if b["kind"] == "repelling")
    if nrep:
        rest += f", {nrep} repelling boundary fixed point{'s' if nrep > 1 else ''}"
    st.markdown(f'<div class="verdict"><b style="color:{KIND_COLOR[kind]}">{word}</b>{html.escape(rest)}</div>',
                unsafe_allow_html=True)

    with st.container(key="facts"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption(f"Denjoy–Wolff point τ {how('dw-point')}", unsafe_allow_html=True)
            where = r"\in\partial\mathbb{D}" if s["tau_location"] == "boundary" else r"\in\mathbb{D}"
            if s["tau_exact"]:
                st.markdown(f"${s['tau_exact']}{where}$")
            else:
                st.markdown(f"{cnum(s['tau'], 12)} (on ∂D)" if s["tau_location"] == "boundary"
                            else f"{cnum(s['tau'], 12)} (in D)")
            st.caption(badge(s["tiers"]["tau"]))
        with c2:
            st.caption(f"Spectral value λ {how('spectral')}", unsafe_allow_html=True)
            if s["lambda_exact"]:
                st.markdown(f"${s['lambda_exact']}$")
            else:
                err = f" ± {num(s['lambda_err'], 2)}" if s.get("lambda_err") else ""
                st.markdown(cnum(s["lambda"], 10) + err)
            st.caption(badge(s["tiers"]["lambda"]))
        with c3:
            st.caption(f"Hyperbolic step {how('step')}", unsafe_allow_html=True)
            k, conf = step_verdict(s, asym)
            st.markdown("not defined" if kind == "elliptic" else (k or "undecided"))
            st.caption(badge(conf) if kind != "elliptic" else " ")
        with c4:
            st.caption("Generator")
            st.markdown(f"$G(z)={s['G_latex']}$" if len(s["G_latex"]) < 300 else s["G_text"])
    lines = []
    if s.get("p_latex"):
        lines.append(f"Berkson–Porta: $p(z)={s['p_latex']}$")
    if s.get("Gamma_latex"):
        lines.append(f"Half-plane model: $\\Gamma(w)={s['Gamma_latex']}$")
    elif s.get("G0_latex") and s["tau_abs"] > 1e-14:
        lines.append(f"Normalised model (τ ↦ 0): $G_0(u)={s['G0_latex']}$")
    for ln in lines:
        st.markdown(ln)
    for w in s.get("warnings") or []:
        st.warning(w)
    for n in s.get("notes") or []:
        st.info(n)


# ------------------------------------------------------------------ disc + inspector
def _chart_key(key: str) -> str:
    return "disc-" + hashlib.md5(key.encode()).hexdigest()[:10]


def _consume_click(chart_key: str):
    state = st.session_state.get(chart_key)
    pts = []
    if state:
        try:
            pts = state["selection"]["points"]
        except (KeyError, TypeError):
            pts = []
    if not pts:
        return
    p = pts[0]
    sel = (round(p.get("x", 0), 6), round(p.get("y", 0), 6))
    if sel == st.session_state.last_sel:
        return
    st.session_state.last_sel = sel
    if sel[0] ** 2 + sel[1] ** 2 < 1:
        st.session_state.inspect = sel
        st.session_state.insp_text = cnum([sel[0], sel[1]], 4)
        st.session_state.insp_error = None


def render_inspector(key, orbit, err):
    st.markdown(f"#### Point inspector {how('backward')}", unsafe_allow_html=True)
    st.text_input("Point z₀", key="insp_text", placeholder="click the disc, or type 0.3+0.4i",
                  on_change=_on_inspect_typed)
    if st.session_state.get("insp_error"):
        st.caption(st.session_state.insp_error)
    if err:
        st.caption(err)
    if not orbit:
        st.caption("Click a point in the disc to follow it forwards and backwards.")
        return
    b = orbit["backward"]
    fwd = orbit["forward"]
    last = [fwd["z"]["re"][-1], fwd["z"]["im"][-1]]
    rows = [("z₀", cnum(orbit["z0"], 6)), (f"φₜ(z₀), t = {num(fwd['t'][-1], 4)}", cnum(last, 6))]
    if fwd.get("dist_tau"):
        rows.append(("|φₜ(z₀) − τ|", num(fwd["dist_tau"][-1], 4)))
    if fwd.get("k_to_tau"):
        rows.append(("k_D(φₜ(z₀), τ)", num(fwd["k_to_tau"][-1], 4)))
    rows.append(("Backward orbit", b["code_name"]))
    if b.get("reason") and b["code"] not in (1, 2):
        rows.append(("", b["reason"]))
    if b.get("exit_time") is not None:
        rows.append(("Escape time T*", num(b["exit_time"], 6)))
    if b.get("landing"):
        rows.append(("Hits ∂D at" if b["code"] == 0 else "Converges to", cnum(b["landing"], 6, 1e-7)))
    if b.get("beta_est"):
        rows.append(("Rate (β estimate)", num(b["beta_est"], 5)))
    if b.get("beta_fit_from_log_margin"):
        rows.append(("Rate from log(1−|z|)", num(b["beta_fit_from_log_margin"], 5)))
    body = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in rows)
    st.markdown(f'<dl class="insp">{body}</dl>', unsafe_allow_html=True)
    compat.plotly_chart(plots.inspector_figure(orbit), config=plots.CONFIG,
                    key=f"insp-{_chart_key(key)}")


# ------------------------------------------------------------------ tabs
def tab_koenigs(key, params):
    with st.spinner("Computing the Koenigs function…"):
        kg, err = compute.run(compute.koenigs, key, params["T_vis"])
    if err:
        st.warning(err)
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        compat.plotly_chart(plots.koenigs_figure(kg), config=plots.CONFIG)
    with right:
        g = kg.get("geometry") or {}
        st.markdown(f"##### {g.get('text', '')}")
        st.caption(kg["method"] + (f" {g['note']}" if g.get("note") else ""))
        if kg.get("formula"):
            st.markdown("Formal antiderivative of i/G (principal branches; may differ from h by branch choices):")
            st.latex(kg["formula"])
        st.markdown(how("koenigs", "How the Koenigs function and the shape of Ω are computed"),
                    unsafe_allow_html=True)


def tab_halfplane(s, dyn):
    left, right = st.columns([3, 2], gap="large")
    if s["kind"] == "elliptic":
        with left:
            st.info("The Denjoy–Wolff point is inside D, so there is no half-plane model. The normalised model "
                    "u = (z − τ)/(1 − τ̄z) is used for the asymptotics.")
        with right:
            if s.get("G0_latex"):
                st.latex(f"G_0(u)={s['G0_latex']}")
            st.markdown(how("charts", "Why the charts are used"), unsafe_allow_html=True)
        return
    with left:
        if dyn and dyn.get("halfplane"):
            compat.plotly_chart(plots.halfplane_figure(s, dyn), config=plots.CONFIG)
    with right:
        st.markdown("##### In w = (τ + z)/(τ − z) the Denjoy–Wolff point is at infinity and the generator becomes "
                    "Γ(w) = 2p(z), with Re Γ ≥ 0.")
        if s.get("Gamma_latex"):
            st.latex(f"\\Gamma(w)={s['Gamma_latex']}")
        st.caption("Point 0 of D corresponds to w = 1. Boundary fixed points other than τ sit on the imaginary axis.")
        st.markdown(how("charts", "Why the half-plane chart is used"), unsafe_allow_html=True)


def tab_asym(s, asym, err):
    if err:
        st.warning(err)
        return
    if asym.get("ok") is False:
        st.warning(asym.get("reason", "The asymptotics could not be computed."))
        return
    figs, rows = plots.asym_figures(asym, s)
    for i in range(0, 4, 2):
        a, b = st.columns(2)
        compat.plotly_chart(figs[i], container=a, config=plots.CONFIG)
        compat.plotly_chart(figs[i + 1], container=b, config=plots.CONFIG)
    table = "| Estimate | Value | Fit residual |\n|---|---|---|\n" + "\n".join(
        f"| {md_escape_cell(a)} | {md_escape_cell(b)} | {md_escape_cell(c)} |" for a, b, c in rows)
    st.markdown(table)
    if asym["kind"] == "elliptic":
        st.caption(f"Base point z₀ = {cnum(asym['z0'], 5)}. Expected: rate = Re λ = {num(s['lambda'][0], 6)}, "
                   f"rotation ω = Im λ = {num(s['lambda'][1], 6)}.")
    else:
        st.caption(f"Orbit of z₀ = {cnum(asym['z0'], 5)} integrated in the half-plane model up to "
                   f"t = {num(asym['T'], 4)}. Fits use the last 30% of the window and do not decide limits.")
    st.markdown(how("speeds", "How speeds, rates and the slope are computed"), unsafe_allow_html=True)


def _math_cell(latex: str) -> str:
    return "$" + latex.replace("|", r"\vert ") + "$"


def tab_boundary(s, bw, err):
    st.markdown(f"#### Boundary fixed points {how('boundary')}", unsafe_allow_html=True)
    pts = s["boundary_points"]
    if not pts:
        st.caption("No boundary null points of G were found.")
    else:
        lines = ["| σ | arg σ / π | Type | β(σ) | Exact | How it was found |", "|---|---|---|---|---|---|"]
        for b in pts:
            beta = "+∞" if b["beta"] == "inf" else cnum(b["beta"], 10)
            if b.get("beta_err") and b["beta"] != "inf" and not b.get("beta_exact"):
                beta += f" ± {num(b['beta_err'], 2)}"
            ex = b.get("beta_exact")
            exact = _math_cell(f"\\sigma={ex['sigma']},\\ \\beta={ex['value']}") if ex else "—"
            lines.append(f"| {cnum(b['sigma'], 10)} | {num(b['angle_over_pi'], 8)} | {b['kind']} | {beta} | {exact} | "
                         f"{md_escape_cell(b['locate_method'] + '; ' + b['beta_method'])} |")
        st.markdown("\n".join(lines))
    st.markdown(f"#### Backward invariant set and petals {how('backward')}", unsafe_allow_html=True)
    if err:
        st.warning(err)
        return
    if not bw:
        return
    st.markdown(" ".join(bw["summary"]) + f" Computed on a {bw['n']}×{bw['n']} grid with backward horizon "
                f"T = {num(bw['T'], 4)}.")
    if bw["petals"]:
        lines = ["| Petal | α-point | Area (share of D) | β from backward orbits |", "|---|---|---|---|"]
        for p in bw["petals"]:
            lines.append(f"| {p['kind']} | {cnum(p['alpha_point'], 6) if p.get('alpha_point') else '—'} | "
                         f"{num(100 * p['area_fraction'], 3)}% | "
                         f"{num(p['beta_estimate_median'], 5) if p.get('beta_estimate_median') else '—'} |")
        st.markdown("\n".join(lines))
    counts = [(k, v) for k, v in bw["counts"].items() if v]
    st.markdown("| Backward behaviour of grid points | Count |\n|---|---|\n"
                + "\n".join(f"| {k} | {v} |" for k, v in counts))


def tab_details(s, asym, kg, bw, spec):
    items = [
        ("Denjoy–Wolff point", s["tau_method"], "dw-point"),
        ("Spectral value", f"{s['lambda_method']}. Convention φₜ′(τ) = e^(−λt).", "spectral"),
        ("Type", f"{s['kind_reason']} ({s['kind_confidence']}).", "spectral"),
    ]
    if s.get("step"):
        items.append(("Hyperbolic step", f"{s['step']['reason']} ({s['step']['confidence']}).", "step"))
    sn = (asym or {}).get("step_numeric")
    if sn and s["kind"] != "elliptic":
        items.append(("Hyperbolic step, numerical", f"{sn.get('rule', '')}. {sn.get('confidence', '')}.", "step"))
    if kg:
        items.append(("Koenigs function", kg["method"], "koenigs"))
    items.append(("Backward invariant set",
                  "Each grid point is integrated backwards (dz/ds = −G). Reaching the circle in finite time means the "
                  "point is not in φₜ(D) for large t; an asymptotic approach to a boundary fixed point puts it in a "
                  "petal. Survivors up to T are undecided, so the result is an outer approximation.", "backward"))
    items.append(("Reliability", "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in s["tiers"].items()) + ".",
                  "reliability"))
    for title, text, anchor in items:
        st.markdown(f"**{title}**  \n{md_escape_cell(text)} {how(anchor, 'Theory and full method')}",
                    unsafe_allow_html=True)
    export = {"input": spec, "summary": s,
              "asymptotics": {k: (asym or {}).get(k) for k in ("kind", "z0", "T", "fits", "step_numeric", "slope_tail")},
              "petals": (bw or {}).get("petals"), "backward_counts": (bw or {}).get("counts"),
              "koenigs_geometry": {k: v for k, v in ((kg or {}).get("geometry") or {}).items()
                                   if k in ("shape", "text", "width", "lambda_from_width", "note")}}
    compat.download_button("Download the results (JSON)", data=json.dumps(export, indent=2),
                       file_name="semigroup-analysis.json", mime="application/json")


# ------------------------------------------------------------------ page
def main():
    st.html(CSS)
    _persist_inputs()
    _init()
    checks_box = sidebar()

    spec, params = st.session_state.spec, st.session_state.params
    key = compute.spec_key(spec)
    try:
        z0 = parse_complex(params["z0"])
    except ValueError as exc:
        st.sidebar.error(str(exc))
        z0 = 0j

    with st.spinner("Analysing…"):
        s, err = compute.run(compute.summary, key)
    if err:
        st.markdown('<div class="verdict">The input could not be analysed.</div>', unsafe_allow_html=True)
        st.error(err)
        st.stop()
    render_checks(checks_box, s)
    with st.spinner("Integrating the long-time orbit…"):
        asym, asym_err = compute.run(compute.asymptotics, key, z0.real, z0.imag, params["as_T"])
    render_headline(s, asym)

    layers = st.pills("Layers", plots.LAYERS, selection_mode="multi", default=plots.DEFAULT_LAYERS,
                      key="layers", label_visibility="collapsed")
    left, right = st.columns([3, 2], gap="large")
    chart_key = _chart_key(key)
    _consume_click(chart_key)
    with left:
        with st.spinner("Computing flow lines and the backward map…"):
            dyn, dyn_err = compute.run(compute.dynamics, key, params["T_vis"])
            bw, bw_err = compute.run(compute.backward, key, params["bw_n"], params["bw_T"])
        for e in (dyn_err, bw_err):
            if e:
                st.warning(e)
        frame = 3
        if dyn and "φₜ(D)" in (layers or []):
            frames = dyn["images"]["frames"]
            frame = st.select_slider("Time t for φₜ(D)", options=list(range(len(frames))), value=min(3, len(frames) - 1),
                                     format_func=lambda i: num(frames[i]["t"], 4), key="frame")
        orbit, orbit_err = None, None
        if st.session_state.inspect:
            x, y = st.session_state.inspect
            with st.spinner("Following the point…"):
                orbit, orbit_err = compute.run(compute.orbit, key, x, y, params["T_vis"])
        fig = plots.disc_figure(s, dyn, bw, layers, frame=frame, orbit=orbit)
        compat.plotly_chart(fig, key=chart_key, on_select="rerun", selection_mode="points",
                            config=plots.CONFIG)
        if dyn:
            st.caption(f"Flow lines for |t| ≤ {num(dyn['T'], 3)} (solid forward, dotted backward). "
                       f"φₜ(D) from {dyn['images']['method']}. Click inside the disc to follow a point. "
                       + how("pictures", "How the pictures are made"), unsafe_allow_html=True)
    with right:
        render_inspector(key, orbit, orbit_err)

    tabs = st.tabs(["Koenigs domain", "Half-plane model", "Speeds and rates", "Boundary and petals", "Method details"])
    with tabs[0]:
        tab_koenigs(key, params)
    with tabs[1]:
        tab_halfplane(s, dyn)
    with tabs[2]:
        tab_asym(s, asym, asym_err)
    with tabs[3]:
        tab_boundary(s, bw, bw_err)
    with tabs[4]:
        kg, _ = compute.run(compute.koenigs, key, params["T_vis"])
        tab_details(s, asym, kg, bw, spec)


main()
