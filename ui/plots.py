"""Plotly figures built from the engine's JSON results (no Streamlit calls here)."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .fmt import cnum, num

C = {
    "ink": "#18202E", "ink2": "#4A5568", "ink3": "#7A8496", "rule": "#D9DEE7",
    "cobalt": "#2747C7", "cobalt_light": "#9AAAE8", "magenta": "#C0266D", "teal": "#0E8A6E",
    "amber": "#C98A00", "grid": "#EEF1F5",
}
FONT = dict(family="Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif", color=C["ink2"], size=12)
CONFIG = {"displaylogo": False, "scrollZoom": True,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

LAYERS = ["Flow lines", "Vector field", "Phase portrait", "φₜ(D)", "Escape time", "Petals", "Fixed points"]
DEFAULT_LAYERS = ["Flow lines", "Petals", "Fixed points"]
CLICK_TRACE_NAME = "click-grid"


def _layout(**extra):
    base = dict(margin=dict(l=48, r=16, t=40, b=40), font=FONT, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", hovermode="closest")
    base.update(extra)
    return base


def _axis(title, **extra):
    a = dict(title=dict(text=title, standoff=6), gridcolor=C["grid"], zerolinecolor=C["rule"],
             linecolor=C["rule"], exponentformat="power")
    a.update(extra)
    return a


def join_lines(lines):
    """[[xs, ys], ...] -> one x list and one y list separated by None."""
    x, y = [], []
    for xs, ys in lines:
        if not xs or len(xs) < 2:
            continue
        x.extend(xs)
        y.extend(ys)
        x.append(None)
        y.append(None)
    return x, y


def _lines_trace(lines, color, width=1.3, dash=None, hover=None):
    x, y = join_lines(lines)
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash),
                      hoverinfo="skip" if hover is None else None, hovertemplate=hover)


# ------------------------------------------------------------------ petals
# Categorical colours for petals, in fixed order (validated: every pair separable, also with
# colour-vision deficiencies). Beyond three petals identity also rests on the white borders drawn
# between petals and on the hover labels, never on colour alone.
PETAL_COLORS = ["#1baf7a", "#eb6834", "#4a3aa7", "#eda100", "#e34948", "#008300"]
PARABOLIC_COLOR = "#e87ba4"
GROUP_COLOR = "#9AAAE8"


def petal_colors(bw):
    """Colour per petal key, following the entity (the α-point), not its rank on screen."""
    out, i = {}, 0
    for p in bw.get("petals", []):
        if not p.get("cells"):
            continue
        if p["key"] == "tau":
            out["tau"] = PARABOLIC_COLOR
        elif p["key"] == "W":
            out["W"] = GROUP_COLOR
        else:
            out[p["key"]] = PETAL_COLORS[min(i, len(PETAL_COLORS) - 1)]
            i += 1
    return out


def petal_name(p):
    if p["key"] == "W":
        return "W = D (group of rotations)"
    if p["key"] == "tau":
        return "parabolic petal (α-point τ)"
    return f"petal of σ = {cnum(p['alpha_point'], 4)}"


def sigma_color(sigma, bw, pcolor):
    for p in (bw or {}).get("petals", []):
        if p["key"] in pcolor and p.get("alpha_point") is not None and p["kind"] == "hyperbolic petal":
            a = p["alpha_point"]
            if abs(complex(a[0], a[1]) - complex(sigma[0], sigma[1])) < 1e-3:  # results are rounded for transport
                return pcolor[p["key"]]
    return C["teal"]


def petal_borders(bw):
    """Grid edges where two different petals meet (a surface-coloured gap between the fills)."""
    lab = np.array(bw["labels"], dtype=float)
    g = np.asarray(bw["grid"], dtype=float)
    d = (g[1] - g[0]) / 2
    is_petal = (lab >= 0) | (lab == -3)
    xs, ys = [], []
    # vertical edges between (i, j) and (i, j+1)
    diff = (lab[:, :-1] != lab[:, 1:]) & is_petal[:, :-1] & is_petal[:, 1:]
    for i, j in zip(*np.nonzero(diff)):
        x = g[j] + d
        xs += [x, x, None]
        ys += [g[i] - d, g[i] + d, None]
    diff = (lab[:-1, :] != lab[1:, :]) & is_petal[:-1, :] & is_petal[1:, :]
    for i, j in zip(*np.nonzero(diff)):
        y = g[i] + d
        xs += [g[j] - d, g[j] + d, None]
        ys += [y, y, None]
    return xs, ys


# ------------------------------------------------------------------ the disc
def disc_figure(summary, dyn, bw, layers, frame=3, orbit=None, height=580):
    layers = set(layers or [])
    fig = go.Figure()
    images = []
    if dyn and "Phase portrait" in layers:
        images.append(dict(source=dyn["phase_portrait"], xref="x", yref="y", x=-1, y=1, sizex=2, sizey=2,
                           sizing="stretch", layer="below", opacity=0.9))
    if bw and "Escape time" in layers:
        z = [[(math.log10(v) if (v is not None and v > 0) else None) for v in row] for row in bw["exit_time"]]
        fig.add_trace(go.Heatmap(
            x=bw["grid"], y=bw["grid"], z=z, opacity=0.8, zsmooth=False,
            colorscale=[[0, "#FFF7E0"], [0.5, "#E8B341"], [1, "#7A4E00"]],
            hovertemplate="escape time 10^%{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="log₁₀ T*", side="right"), thickness=10, len=0.55, x=1.0)))
    pcolor = petal_colors(bw) if bw else {}
    if bw and "Petals" in layers and pcolor:
        labels = bw["labels"]
        for p in bw["petals"]:
            key = p["key"]
            if key not in pcolor:
                continue
            code = -3 if key == "tau" else (-5 if key == "W" else key)
            z = [[(1 if l == code else None) for l in row] for row in labels]
            name = petal_name(p)
            fig.add_trace(go.Heatmap(
                x=bw["grid"], y=bw["grid"], z=z, zmin=0, zmax=1, showscale=False, opacity=0.32,
                colorscale=[[0, pcolor[key]], [1, pcolor[key]]], hovertemplate=f"{name}<extra></extra>"))
        bx, by = petal_borders(bw)
        if bx:
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color="#FFFFFF", width=1.6),
                                     hoverinfo="skip"))
    if dyn and "Vector field" in layers:
        v = dyn["vector_field"]
        for xs, ys in ((v["shaft_x"], v["shaft_y"]), (v["head_x"], v["head_y"])):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C["ink3"], width=1), hoverinfo="skip"))
    if dyn and "Flow lines" in layers:
        fig.add_trace(_lines_trace(dyn["flows"]["backward"], C["cobalt_light"], 1, "dot"))
        fig.add_trace(_lines_trace(dyn["flows"]["forward"], C["cobalt"], 1.3))
    if dyn and "φₜ(D)" in layers:
        frames = dyn["images"]["frames"]
        fr = frames[min(frame, len(frames) - 1)]
        fig.add_trace(_lines_trace(fr["circles"] + fr["rays"], "#B48AA3", 0.9))
        fig.add_trace(go.Scatter(x=fr["boundary"][0], y=fr["boundary"][1], mode="lines",
                                 line=dict(color=C["magenta"], width=2),
                                 hovertemplate=f"boundary of φ_t(D), t = {num(fr['t'], 4)}<extra></extra>"))
    th = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color=C["ink"], width=1.6),
                             hoverinfo="skip"))
    if summary and "Fixed points" in layers:
        bps = summary["boundary_points"]
        rep = [b for b in bps if b["kind"] == "repelling"]
        sup = [b for b in bps if b["kind"].startswith("super")]
        if rep:
            colors = [sigma_color(b["sigma"], bw, pcolor) for b in rep]
            fig.add_trace(go.Scatter(
                x=[b["sigma"][0] for b in rep], y=[b["sigma"][1] for b in rep], mode="markers",
                marker=dict(symbol="triangle-up", size=13, color=colors, line=dict(color=C["ink"], width=1)),
                text=[f"repelling, β = {cnum(b['beta'], 6)}" for b in rep], hovertemplate="%{text}<extra></extra>"))
        if sup:
            fig.add_trace(go.Scatter(
                x=[b["sigma"][0] for b in sup], y=[b["sigma"][1] for b in sup], mode="markers",
                marker=dict(symbol="triangle-up-open", size=12, color=C["teal"], line=dict(width=2)),
                hovertemplate="super-repelling (β = +∞)<extra></extra>"))
        tau = summary["tau"]
        fig.add_trace(go.Scatter(
            x=[tau[0]], y=[tau[1]], mode="markers",
            marker=dict(symbol="circle", size=13, color=C["magenta"], line=dict(color="#fff", width=2)),
            hovertemplate=f"Denjoy–Wolff point τ = {cnum(tau, 6)}<extra></extra>"))
    if orbit:
        fz, bz = orbit["forward"]["z"], orbit["backward"]["z"]
        fig.add_trace(go.Scatter(x=bz["re"], y=bz["im"], mode="lines", line=dict(color=C["amber"], width=3),
                                 hovertemplate="backward orbit<extra></extra>"))
        fig.add_trace(go.Scatter(x=fz["re"], y=fz["im"], mode="lines", line=dict(color=C["ink"], width=3),
                                 hovertemplate="forward orbit<extra></extra>"))
        z0 = orbit["z0"]
        fig.add_trace(go.Scatter(x=[z0[0]], y=[z0[1]], mode="markers",
                                 marker=dict(size=10, color=C["ink"], line=dict(color="#fff", width=2)),
                                 hovertemplate=f"z₀ = {cnum(z0, 5)}<extra></extra>"))
    # Invisible grid on top: clicking selects the nearest grid point, which the page turns into z0.
    g = np.linspace(-0.99, 0.99, 67)
    X, Y = np.meshgrid(g, g)
    inside = X ** 2 + Y ** 2 < 0.985
    fig.add_trace(go.Scatter(
        x=np.round(X[inside], 4), y=np.round(Y[inside], 4), mode="markers", name=CLICK_TRACE_NAME,
        marker=dict(size=9, color="rgba(0,0,0,0)"),
        hovertemplate="z = %{x:.3f} %{y:+.3f}i<extra>click to inspect</extra>"))
    fig.update_layout(**_layout(
        margin=dict(l=8, r=8, t=8, b=8), images=images, height=height, dragmode="pan",
        plot_bgcolor="rgba(0,0,0,0)", clickmode="event+select",
        xaxis=dict(range=[-1.06, 1.06], visible=False, constrain="domain"),
        yaxis=dict(range=[-1.06, 1.06], visible=False, scaleanchor="x", scaleratio=1, constrain="domain")))
    return fig


def inspector_figure(orbit):
    b = orbit["backward"]
    fig = go.Figure([
        go.Scatter(x=orbit["forward"]["t"], y=orbit["forward"]["k_from_start"], mode="lines",
                   line=dict(color=C["ink"], width=2), name="forward"),
        go.Scatter(x=b["t"], y=b["k_from_start"], mode="lines", line=dict(color=C["amber"], width=2),
                   name="backward"),
    ])
    fig.update_layout(**_layout(
        height=240, margin=dict(l=40, r=8, t=30, b=34), showlegend=True,
        legend=dict(orientation="h", y=-0.3), title=dict(text="k_D(z₀, orbit) against t", font=dict(size=12)),
        xaxis=_axis("t"), yaxis=_axis("")))
    return fig


# ------------------------------------------------------------------ Koenigs
def koenigs_figure(kg):
    fig = go.Figure()
    grid = [[c["h"]["re"], c["h"]["im"]] for c in kg["circles"]] + [[r["h"]["re"], r["h"]["im"]] for r in kg["rays"]]
    fig.add_trace(_lines_trace(grid, "#C6CCD8", 0.8))
    fig.add_trace(_lines_trace(kg["backward_orbits"], C["cobalt_light"], 1, "dot"))
    fig.add_trace(_lines_trace(kg["orbits"], C["cobalt"], 1.3))
    last = kg["circles"][-1]
    fig.add_trace(go.Scatter(x=last["h"]["re"], y=last["h"]["im"], mode="lines",
                             line=dict(color=C["magenta"], width=1.6),
                             hovertemplate=f"image of |z| = {last['r']}<extra></extra>"))
    g = kg.get("geometry") or {}
    shapes = []
    if g.get("kind") == "non-elliptic" and g.get("shape") == "vertical strip":
        for xv in (g["sup_re"][-1], g["inf_re"][-1]):
            shapes.append(dict(type="line", xref="x", yref="paper", x0=xv, x1=xv, y0=0, y1=1,
                               line=dict(color=C["teal"], dash="dash", width=1)))
    w = kg.get("window")
    title = "Ω = h(D), h∘φ_t = e^{−λt} h" if kg["kind"] == "elliptic" else "Ω = h(D), h∘φ_t = h + it"
    fig.update_layout(**_layout(
        height=520, title=dict(text=title, font=dict(size=13)), shapes=shapes, dragmode="pan",
        xaxis=_axis("Re h", **({"range": [w[0], w[1]]} if w else {})),
        yaxis=_axis("Im h", scaleanchor="x", scaleratio=1, **({"range": [w[2], w[3]]} if w else {}))))
    return fig


# ------------------------------------------------------------------ half-plane
def halfplane_figure(summary, dyn):
    hp = dyn["halfplane"]
    fig = go.Figure([
        _lines_trace(hp["backward"], C["cobalt_light"], 1, "dot"),
        _lines_trace(hp["forward"], C["cobalt"], 1.3),
    ])
    tau = complex(*summary["tau"])
    others = [b for b in summary["boundary_points"] if not b["is_dw"]]
    if others:
        ws = [(tau + complex(*b["sigma"])) / (tau - complex(*b["sigma"])) for b in others]
        fig.add_trace(go.Scatter(x=[w.real for w in ws], y=[w.imag for w in ws], mode="markers",
                                 marker=dict(symbol="triangle-right", size=12, color=C["teal"]),
                                 text=[b["kind"] for b in others], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(**_layout(
        height=520, title=dict(text="Orbits in H = {Re w > 0}; τ is at infinity", font=dict(size=13)),
        dragmode="pan", xaxis=_axis("Re w", range=[-0.3, 8]),
        yaxis=_axis("Im w", range=[-6, 6], scaleanchor="x", scaleratio=1),
        shapes=[dict(type="line", x0=0, x1=0, y0=-1e3, y1=1e3, line=dict(color=C["ink"], width=1.5))]))
    return fig


# ------------------------------------------------------------------ asymptotics
def _chart(traces, title, xt, yt, xtype="linear", ytype="linear", **extra):
    fig = go.Figure(traces)
    fig.update_layout(**_layout(
        height=320, title=dict(text=title, font=dict(size=13)), showlegend=len(traces) > 1,
        legend=dict(orientation="h", y=-0.28), xaxis=_axis(xt, type=xtype), yaxis=_axis(yt, type=ytype)))
    if extra:
        fig.update_layout(**extra)
    return fig


def _line(x, y, name, color, dash=None):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2, dash=dash))


def asym_figures(a, summary):
    """Four figures and the rows of the fits table."""
    rows = []
    if a["kind"] == "elliptic":
        figs = [
            _chart([_line(a["t"], a["dist_tau"], "|φ_t(z₀) − τ|", C["cobalt"])], "Distance to τ",
                   "t", "|φ_t(z₀) − τ|", ytype="log"),
            _chart([_line(a["t"], a["arg"], "arg", C["magenta"])], "Rotation: arg u(t), u = M(φ_t(z₀))",
                   "t", "arg (unwrapped)"),
            _chart([_line(a["t"], a["k_to_tau"], "k", C["teal"])], "Hyperbolic distance k_D(τ, φ_t(z₀))",
                   "t", "k", ytype="log"),
            _chart([go.Scatter(x=a["u"]["re"], y=a["u"]["im"], mode="lines", line=dict(color=C["cobalt"], width=1.5))],
                   "Orbit in the normalised chart (τ ↦ 0)", "Re u", "Im u",
                   yaxis=_axis("Im u", scaleanchor="x", scaleratio=1)),
        ]
    else:
        figs = [
            _chart([_line(a["t"], a["v_total"], "total v(t)", C["ink"]),
                    _line(a["t"], a["v_orth"], "orthogonal v_o(t)", C["cobalt"]),
                    _line(a["t"], a["v_tan"], "tangential v_T(t)", C["magenta"], "dot")],
                   "Speeds of convergence", "t", "k_D", xtype="log"),
            _chart([_line(a["t"], a["dist_tau"], "|φ_t − τ|", C["cobalt"]),
                    _line(a["t"], a["one_minus_abs2"], "1 − |φ_t|²", C["teal"], "dot")],
                   "Euclidean rates", "t", "", xtype="linear" if a["kind"] == "hyperbolic" else "log", ytype="log"),
            _chart([_line(a["step_t"], a["step_v"], "k(φ_t, φ_{t+1})", C["amber"])],
                   "Hyperbolic step k_D(φ_t(z₀), φ_{t+1}(z₀))", "t", "", xtype="log", ytype="log"),
            _chart([_line(a["t"], a["slope"], "slope", C["magenta"])], "Slope arg(1 − τ̄ φ_t(z₀))", "t", "radians",
                   xtype="log", yaxis=_axis("radians", range=[-1.7, 1.7]),
                   shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=v, y1=v,
                                line=dict(color=C["rule"], dash="dash")) for v in (-math.pi / 2, math.pi / 2)]),
        ]
        sn = a.get("step_numeric") or {}
        rows.append((f"Hyperbolic step (numerical): tail log-log slope of k(φ_t, φ_t+1); last value "
                     f"{num(sn.get('last_value'), 4)} at t = {num(sn.get('t_last'), 4)}",
                     sn.get("kind") or "undecided", f"slope {num(sn.get('loglog_tail_slope'), 3)}"))
        st_ = a.get("slope_tail") or {}
        rows.append(("Slope on the last 30% of the window",
                     f"[{num(st_.get('min'), 4)}, {num(st_.get('max'), 4)}]",
                     "tangential" if st_.get("tangential") else "non-tangential"))
    for f in (a.get("fits") or {}).values():
        rows.append((f["meaning"], num(f["value"], 6), num(f["resid"], 2)))
    return figs, rows
