# How everything is computed

This page explains, for each quantity the tool reports, the theory it rests on, the algorithm that computes it, and what accuracy to expect. Every section ends with the actual source of the functions involved, read live from the running code, so what you read here is what runs.

Conventions throughout follow Bracci, Contreras and Díaz-Madrigal. $\mathbb D$ is the unit disc, $(\varphi_t)_{t\ge 0}$ a continuous semigroup of holomorphic self-maps, $G$ its infinitesimal generator,

$$
\frac{\partial \varphi_t(z)}{\partial t} = G(\varphi_t(z)), \qquad \varphi_0 = \mathrm{id},
$$

and the hyperbolic distance is normalised as $k_{\mathbb D}(z,w) = \operatorname{artanh}\left|\frac{z-w}{1-\bar w z}\right|$, so $k_{\mathbb D}(0,z) = \operatorname{artanh}|z|$.

<!-- section: overview -->

## What happens when you press Analyse

Everything runs inside the Streamlit process, in five steps. The summary comes first because everything else depends on the Denjoy–Wolff point found there; the other four fill in the page as they finish.

| Step | Computes | Typical time |
|---|---|---|
| Summary | generator, τ, λ, type, group test, exact step test, boundary fixed points, all checks | 0.1–0.4 s |
| Dynamics | vector field, phase portrait, flow lines, $\varphi_t(\mathbb D)$, half-plane picture | 0.5 s |
| Backward map | backward integration of a grid of points, escape times, petals | 0.2–1.5 s |
| Koenigs | Koenigs function on a polar grid, orbits in $\Omega$, shape of $\Omega$ | 0.1–1.2 s |
| Asymptotics | long-time orbit, speeds, rates, slope, numerical hyperbolic step | 0.2–0.6 s |

Each step is cached for its input and its own parameters, and the model built from your input (parsed generator, τ, λ, the conjugated charts) is cached as well. Switching picture layers, moving the time slider, changing tabs or inspecting points therefore never repeats an analysis, and changing, say, the backward horizon recomputes only the backward map.

<!-- section: reliability -->

## Reading the reliability tags

Each headline number carries a tag. They correspond to how the number was obtained, not to how many digits are shown.

| Tag | Meaning | Trust it for |
|---|---|---|
| :green-badge[exact] | A closed form was found and verified symbolically (for example τ = 1 is checked to be a root of the numerator of G in exact arithmetic). | Statements of theorems. |
| :green-badge[theorem] | Follows from a theorem once the type is known (hyperbolic ⇒ positive step). | As reliable as the type. |
| :green-badge[given] | You supplied it (τ in Berkson–Porta mode). | Your input. |
| :blue-badge[numerical] | A finite computation with a controlled error, e.g. polynomial roots to 40 digits or Newton refinement to 32 digits. | All shown digits, up to the error given. |
| :blue-badge[numerical limit] | A limit (angular derivative at a boundary point) estimated from a sequence approaching it; the error shown is the difference of the last two terms. | Values; a reported 0 means "below the error", not exactly 0. |
| :orange-badge[numerical estimate] | An asymptotic property (t → ∞, or behaviour at ∂D) inferred from a finite window by a stated rule. | Evidence. These can be wrong for slowly converging examples. |

<!-- section: input -->

## Input and the generator

### Theory

A holomorphic $G\colon\mathbb D\to\mathbb C$ generates a semigroup exactly when it has the Berkson–Porta form

$$
G(z) = (z-\tau)(\bar\tau z - 1)\,p(z), \qquad \tau\in\overline{\mathbb D},\ \operatorname{Re}p \ge 0.
$$

The pair $(\tau, p)$ is unique unless $G\equiv 0$, and τ is then the Denjoy–Wolff point of the semigroup.

### What the tool does

- **Generator mode** uses your G directly.
- **Berkson–Porta mode** builds $G = (z-\tau)(\bar\tau z-1)p$ symbolically; τ must be a constant with $|\tau|\le 1$. Because τ is given, it is used as the Denjoy–Wolff point without searching, and the checks then test whether Re p ≥ 0 really holds.
- **Semigroup mode** takes $\varphi_t(z)$ in closed form and sets $G = \partial_t\varphi_t|_{t=0}$ by symbolic differentiation. It checks $\varphi_0 = \mathrm{id}$ at sample points, and later compares the closed form with the numerical flow of G.

The expression is first checked against a whitelist of names and characters and only then parsed by SymPy. Before SymPy evaluates anything, numeric exponents larger than 100 in absolute value and integers longer than 50 digits are rejected, because inputs such as `9^9^9^9` or `(1+z)^100000` would otherwise occupy the server for a very long time. Decimals become exact rationals (0.5 is 1/2), so the exact engine can use them. Functions are taken on their principal branches, which is why the checks look for branch cuts inside D.

If the simplified G is a rational function of z (degree at most 40), the *exact engine* is used: numerator and denominator are kept as polynomials and their roots are computed to 40 digits. A pole inside D is rejected immediately. Otherwise G is evaluated numerically through NumPy (fast, double precision) and mpmath (any precision).

<!-- code: parse -->

<!-- code: model_init -->

<!-- section: validation -->

## Checks on the input

The checks list in the sidebar reports four tests. All of them sample the disc, so a pass is evidence, not proof.

### Generator criterion (no τ needed)

A holomorphic G is an infinitesimal generator if and only if

$$
\operatorname{Re}\big(G(z)\,\bar z\big) \;\le\; (1-|z|^2)\,\operatorname{Re}\big(G(0)\,\bar z\big)\qquad (z\in\mathbb D).
$$

This characterisation needs no knowledge of τ, so it is the first test. The tool evaluates both sides on 70 radii (30 evenly spaced up to 0.9, then 40 approaching the circle geometrically down to $1-|z| = 10^{-7}$) times 240 angles, divides the excess by $|G(z)||z| + |G(0)|$, and passes if the largest relative excess is at most $10^{-9}$. It also reports the worst point. (Before relying on it, the criterion was confirmed numerically on 300 random Berkson–Porta generators and fails on non-generators such as G(z) = z.)

<!-- code: generator_inequality -->

### Berkson–Porta

With the τ found (or given), $p = G/((z-\tau)(\bar\tau z-1))$ is evaluated on the same sample (points within $10^{-6}$ of τ are skipped). The test passes if $\min \operatorname{Re}p/\max(1,|p|) \ge -10^{-9}$.

<!-- code: berkson_porta_check -->

### Holomorphy

Three tests. First, G must be finite at 6000 random points with $|z| < 1-10^{-4}$. Second, a Cauchy–Riemann test: for a random step $h$ of size $10^{-6}$, the relative mismatch $|G(z+h)-G(z)-G'(z)h| / |G'(z)h|$ must stay below $10^{-2}$. Third, a branch-cut scan, because random points almost never land on a cut: every subexpression with a principal branch (non-integer powers, log, inverse trigonometric and hyperbolic functions) has its argument evaluated on a polar grid of 220 radii by 481 angles. Where the argument crosses the cut (for sqrt and log, the negative real axis) between neighbouring grid points, G is evaluated on both sides and a jump larger than ten times the expected change $|G'|\,|\Delta z|$ is reported with its location. Example: $-z\sqrt{z+1/4}$ fails, with a cut along $(-1,-1/4]$.

<!-- code: holomorphy -->

<!-- code: branch_cut_scan -->

### Semigroup law

For 40 random points with $|z| < 0.95$ and $s = 0.37$, $t = 0.81$, the numerical flow is checked for $|\varphi_{t+s}(z) - \varphi_t(\varphi_s(z))|$ (pass below $10^{-6}$; typical values are $10^{-13}$). In semigroup mode the closed form is also checked: its own semigroup law (below $10^{-8}$), its distance to the flow of G (below $10^{-6}$), and $\max|\varphi_t| \le 1$ at t = 0.1, 1 and 5.

<!-- code: semigroup_law -->

<!-- section: dw-point -->

## Denjoy–Wolff point :green-badge[exact] :blue-badge[numerical]

### Theory

Unless the semigroup is a group of elliptic automorphisms, every orbit $\varphi_t(z)$ converges to a unique point $\tau\in\overline{\mathbb D}$ as $t\to\infty$. A non-trivial generator has at most one zero in D, and it is τ when it exists. If G has no zero in D, then τ lies on ∂D and it is the unique boundary point where G vanishes with $\beta(\tau) = \angle\lim G(z)/(z-\tau) \le 0$; every other boundary fixed point has $\beta > 0$ (see [boundary fixed points](#boundary)).

### What the tool does

1. **Interior zeros.** For rational G: roots of the numerator inside $|z| < 1 - 10^{-12}$ that are not roots of the denominator. Otherwise: Newton's method from 57 seeds (radii 0, 0.35, 0.65, 0.85, 0.95 times 14 angles, steps capped at 0.5, 80 iterations), keeping limits with $|G| < 10^{-10}$, then refining each with mpmath to 32 digits. More than one zero means the input is not a generator, and the analysis stops with that message.
1. **Boundary case.** If there is no interior zero, the boundary null points of G are located (next sections), β is computed for each, and τ is the one with $\beta \le 10^{-7}$.
1. **Fallback.** If no boundary null point with β ≤ 0 is found, τ is taken as the direction of $\varphi_t(0)$ at $t = 10^4$. This is flagged with a warning because the boundary quantities that depend on τ are then less accurate.
1. **Closed form.** For rational G the numerical τ is passed to SymPy's `nsimplify` (allowing √2, √3, √5); the candidate is accepted only if it is verified to be an exact root of the numerator (and, on the boundary, to have modulus exactly 1). That is what the :green-badge[exact] tag means.

### What to expect

Interior τ is accurate to 30+ digits. Boundary τ for rational G is accurate to 40 digits; for non-rational G it is refined by Newton's method in high precision when G is analytic there, or by golden-section minimisation of |G| on the circle $|z| = 1-10^{-30}$ when it is not (for instance at a square-root singularity).

<!-- code: interior_zeros -->

<!-- code: find_tau -->

<!-- section: spectral -->

## Spectral value and type :green-badge[exact] :blue-badge[numerical limit]

### Theory

The spectral value λ is defined by $\varphi_t'(\tau) = e^{-\lambda t}$ (an angular derivative when τ ∈ ∂D).

- **Elliptic** (τ ∈ D): $\lambda = -G'(\tau)$, $\operatorname{Re}\lambda \ge 0$, and $\operatorname{Re}\lambda = 0$ exactly when every $\varphi_t$ is an automorphism (a group of rotations about τ). $\operatorname{Im}\lambda \ne 0$ makes orbits spiral.
- **Hyperbolic** (τ ∈ ∂D, λ > 0) and **parabolic** (τ ∈ ∂D, λ = 0), where $\lambda = -\angle\lim_{z\to\tau} G(z)/(z-\tau) \ge 0$.
- **Groups.** The semigroup extends to a group of automorphisms exactly when $G(z) = a - \bar a z^2 + i b z$ with b real (the complete holomorphic vector fields of D).

### What the tool does

If G is analytic at τ, $\lambda = -G'(\tau)$ is evaluated at 50 digits, and symbolically when τ is exact. Otherwise λ is the limit $\lim_{x\to+\infty}\Gamma(x)/x$ in the half-plane chart (next section), evaluated at $x = 10^2, 10^4, \dots, 10^{32}$ (or up to $10^{59}$ when p is given), and the reported error is the difference of the last two terms.

The type is then read off. The decision "λ = 0" uses the tolerance $\max(10^{-9}, 10\cdot\text{error})$ unless λ is exact. When this happens numerically, a note says that a hyperbolic semigroup with λ below the tolerance cannot be excluded: that distinction can only be made exactly. The group test is exact for polynomial G of degree at most 2; otherwise G is fitted by a quadratic at 40 points and accepted as a group when the fit residual is below $10^{-10}$ and the coefficient conditions hold to $10^{-9}$.

<!-- code: spectral_value -->

<!-- code: lambda_from_gamma -->

<!-- code: classify -->

<!-- code: detect_group -->

<!-- section: charts -->

## Half-plane and normalised charts

### Theory

For τ ∈ ∂D the Cayley map $w = (\tau+z)/(\tau-z)$ sends D onto $\mathbb H = \{\operatorname{Re}w > 0\}$, τ to ∞ and 0 to 1. The conjugated generator is

$$
\Gamma(w) = \frac{(w+1)^2}{2\tau}\,G(z) = 2\,p(z), \qquad \operatorname{Re}\Gamma \ge 0, \qquad \lambda = \lim_{x\to+\infty}\frac{\Gamma(x)}{x}.
$$

For τ ∈ D the automorphism $u = (z-\tau)/(1-\bar\tau z)$ moves τ to 0, with conjugated generator $G_0(u)$ and $G_0'(0) = -\lambda$.

### Why the charts matter numerically

Orbits pile up at τ. In disc coordinates the distance $|z-\tau|$ is lost to rounding once it drops below about $10^{-16}$, which happens after a few dozen time units for hyperbolic semigroups. In the half-plane chart the same orbit simply grows ($|w|\sim e^{\lambda t}$ or $\sim t$) and stays exact in double precision.

The substitution $z = \tau(w-1)/(w+1)$ alone would not help: evaluating $1-\bar\tau z$ for z close to τ cancels catastrophically. So after substitution every rational subexpression in w is cancelled symbolically, bottom-up; then $1 - \bar\tau z$ becomes exactly $2/(w+1)$. For example $G = 1-z^2$ with τ = 1 becomes $\Gamma(w) = 2w$, and $p = \sqrt{(1+z)/(1-z)}$ becomes $\Gamma = 2\sqrt w$. When you give p (Berkson–Porta mode) the tool uses $\Gamma = 2p(z(w))$ directly.

<!-- code: normal_model -->

<!-- code: cancel_rational_parts -->

<!-- section: boundary -->

## Boundary fixed points :green-badge[exact] :blue-badge[numerical limit]

### Theory

A boundary point σ with radial limit $G(\sigma) = 0$ is a boundary fixed point of every $\varphi_t$. Its behaviour is measured by

$$
\beta(\sigma) = \angle\lim_{z\to\sigma}\frac{G(z)}{z-\sigma}\ \in\ [-\infty, +\infty], \qquad \varphi_t'(\sigma) = e^{\beta t}.
$$

β is real. β ≤ 0 happens only at the Denjoy–Wolff point (where β = −λ). $0 < \beta < \infty$ is a **repelling** (regular) boundary fixed point with repelling spectral value β. β = +∞ is a **non-regular**, or super-repelling, boundary fixed point.

### What the tool does

- **Rational G:** numerator roots with $\big||r|-1\big| < 10^{-15}$ that are not poles, then $\beta = G'(\sigma)$ at 50 digits, with a closed form when σ is recognised exactly.
- **Other G:** $|G|$ is sampled on the circle $|z| = 1-10^{-10}$ at 4096 angles. The 60 deepest local minima are refined by golden-section search at $|z| = 1-10^{-12}$ and kept if $|G|$ there is below $10^{-4}$ times its median on the circle and $|G|$ decreases along the radius (at $1-|z| = 10^{-4}, 10^{-8}, 10^{-12}$). Each point is refined in high precision as for τ.
- **β for non-rational G** is $-\lim_{x\to\infty}\Gamma_\sigma(x)/x$, where $\Gamma_\sigma$ is G in the half-plane chart centred at σ (same symbolic cancellation as above), at $x = 10^2,\dots,10^{32}$. If $|\Gamma_\sigma(x)/x|$ keeps growing by more than a factor 1.5 per step and exceeds 100, β is reported as +∞.

### What to expect

For rational G the list is complete. For other G, a boundary fixed point is found only if |G| has a detectable radial minimum there; super-repelling points where |G| decays very slowly can be missed.

<!-- code: boundary_find -->

<!-- code: beta_halfplane -->

<!-- code: classify_boundary -->

<!-- section: step -->

## Hyperbolic step :green-badge[theorem] :green-badge[exact] :orange-badge[numerical estimate]

### Theory

For a non-elliptic semigroup the limit $\lim_{t\to\infty} k_{\mathbb D}(\varphi_t(z), \varphi_{t+1}(z))$ exists. Whether it is positive does not depend on z: the semigroup has **positive** or **zero hyperbolic step**. Hyperbolic semigroups always have positive step, so the question is only about parabolic ones.

### Exact test (parabolic, G analytic at τ, G″(τ) ≠ 0)

Write $\Gamma(w) = c_0 + c_1/w + O(w^{-2})$ at ∞; then $c_0 = \tau G''(\tau)$ and $\operatorname{Re}c_0 \ge 0$.

- If $\operatorname{Re}c_0 > 0$, then $\operatorname{Re}w(t)\sim \operatorname{Re}(c_0)\,t$ and $k(w(t), w(t+1))\to 0$: **zero step**.
- If $c_0 = i\kappa$ with κ ≠ 0, then $q = \Gamma - i\kappa$ has $\operatorname{Re}q\ge0$ and vanishes at ∞. Because $f(u) = q(1/u)$ is analytic at the boundary point 0 with positive real part, $c_1 = f'(0)$ must be real. Hence $\operatorname{Re}\Gamma(w(t)) = O(t^{-2})$ is integrable, $\operatorname{Re}w(t)$ stays bounded while $|w(t)|\sim|\kappa|t$, and the step is **positive**.

This is a formal asymptotic argument written for this tool, not a result quoted from the literature. It agrees with every library example and with the numerical estimate. When G″(τ) = 0 (a zero of order three or more) or G is not analytic at τ, only the numerical estimate is available.

<!-- code: step_exact -->

### Numerical estimate (always computed)

The orbit of the base point is integrated in $\mathbb H$ up to $t = 10^4$ (parabolic) and $s(t) = k_{\mathbb H}(w(t), w(t+1))$ is evaluated at 320 log-spaced times. On the last 35% of them the slope of $\log s$ against $\log t$ is fitted:

- slope below −0.2, or a last value below $10^{-6}$: **zero**;
- slope within ±0.05 and a last value above $10^{-3}$: **positive**;
- anything else: undecided.

Expect this to work when $s(t)$ decays like a power of t (it does like $1/t$ in all zero-step library examples). If $s(t)$ decays like $1/\log t$, the slope sits near zero and the rule may wrongly say positive or stay undecided. That is why the result is tagged as an estimate.

<!-- code: step_numeric -->

<!-- section: integrator -->

## How orbits are integrated

All orbits come from one integrator: an adaptive Dormand–Prince 5(4) Runge–Kutta method written for complex arrays, in which every initial point carries its own time and step size. A whole grid of points (10,000 for the backward map) therefore advances together in NumPy.

- **Error control:** each step is accepted when $|\text{err}| \le \text{atol} + \text{rtol}\cdot|y|$. The tolerances are rtol $10^{-8}$ to $10^{-11}$ depending on the task (listed in each section).
- **Output times:** steps are shortened to land exactly on requested times; there is no interpolation.
- **Boundary handling:** a margin function ($1-|z|$ in the disc) is monitored and the first time it drops below each threshold is recorded by linear interpolation within the step. Steps are additionally capped so that the radial approach to the circle is at most half the current margin. Only the radial velocity counts, so orbits sliding along the circle are not slowed down.
- **Budgets:** each point has a step budget (5000–8000 steps for grids, up to 200,000 for single long orbits). A point that exhausts it, or whose step size underflows, is reported as a numerical failure, never as a result.

<!-- code: integrate -->

<!-- section: backward -->

## Backward orbits, W and petals :blue-badge[numerical]

### Theory

A point z lies in $\varphi_t(\mathbb D)$ exactly when the backward equation $\dot\zeta = -G(\zeta)$, $\zeta(0)=z$, has a solution in D up to time t (each $\varphi_t$ is univalent). The **backward invariant set** is $W = \bigcap_{t\ge0}\varphi_t(\mathbb D)$, the set of points with a backward orbit defined for all time. The connected components of its interior are the **petals**, on which every $\varphi_t$ acts as an automorphism.

- A **hyperbolic petal** has backward orbits converging to a repelling boundary fixed point, its α-point. Each repelling boundary fixed point is the α-point of exactly one hyperbolic petal.
- A **parabolic petal** has backward orbits converging to τ itself. It exists exactly for parabolic semigroups of positive hyperbolic step.

### What the tool does

Every point of an n × n grid on $[-1,1]^2$ with $|z| < 1-1/n$ (default n = 101) is integrated backwards up to time T (default 60) with rtol $10^{-8}$. For each point the tool records the times $t_1, t_2$ at which $1-|\zeta|$ first drops below $10^{-5}$ and $10^{-9}$. Their difference separates the two ways of reaching the circle:

$$
\begin{aligned} t_2 - t_1 &\approx \frac{\ln 10^4}{\beta} && \text{near a repelling point with value } \beta,\\ t_2 - t_1 &\approx \frac{10^{-5}}{v_\perp} \approx 0 && \text{for an exit crossing the circle with normal speed } v_\perp. \end{aligned}
$$

| Grid point classified as | Rule |
|---|---|
| escapes in finite time | reaches $1-\vert\zeta\vert = 10^{-9}$ with $t_2 - t_1 \le 0.05$; also when it runs into a pole of G on the circle (the step size collapses there, \|G\| > 10⁶). Its escape time $T^* = t_2$ is what the escape-time layer shows: the point lies in $\varphi_t(\mathbb D)$ exactly for $t < T^*$. |
| converges to a repelling boundary fixed point | $t_2 - t_1 > 0.05$ and the landing point is within 0.02 of a known boundary fixed point (or \|G\| < 10⁻⁴ there). The rate is estimated as $\beta \approx \ln 10^4/(t_2-t_1)$. |
| converges to τ (parabolic petal) | non-elliptic, still inside D at time T, and within 0.05 of τ with $1-\vert\zeta\vert < 10^{-2}$; or a slow approach landing at τ. |
| survives up to T (undecided) | still inside D at time T, elsewhere. |
| numerical failure | step budget (6000) exhausted or step size underflow away from a pole. |

Petals are the 8-connected components of the points with an asymptotic backward orbit, with at least 3 grid cells. The α-point is the most frequent landing angle (rounded to 0.01π), snapped to a listed boundary fixed point within 0.03. A component is a parabolic petal if most of its points converge to τ. The β estimate shown for a petal is the median over its points and should match the repelling spectral value of its α-point; in the library examples it agrees to 3–4 digits.

### What to expect

The result is an *outer approximation* of W at resolution $2/(n-1)$. Undecided survivors are not counted as W, so a larger T sharpens the picture. Petal edges are uncertain by one grid cell, and thin pieces of D \ W (such as the slit (−1, 0] in the Koebe example) show up only where grid points happen to lie on them. Clicking a point in the disc (or typing it in the point inspector) runs the same classification for that single point with rtol $10^{-10}$ and also fits β from the slope of $\log(1-|\zeta(t)|)$.

<!-- code: classify_backward -->

<!-- code: backward_map -->

<!-- code: backward_orbit -->

<!-- section: koenigs -->

## Koenigs function and domain :blue-badge[numerical] :orange-badge[numerical estimate]

### Theory

The semigroup is conjugate to a linear model through a univalent Koenigs function h:

$$
\begin{aligned} &\text{non-elliptic:} && h'(z)\,G(z) = i, && h(0)=0, && h\circ\varphi_t = h + it,\\ &\text{elliptic:} && h'(z)\,G(z) = -\lambda h(z), && h(\tau)=0, && h\circ\varphi_t = e^{-\lambda t}h. \end{aligned}
$$

The Koenigs domain $\Omega = h(\mathbb D)$ encodes the type. In the non-elliptic case $\Omega + it\subset\Omega$ for t ≥ 0, and:

- hyperbolic ⇔ Ω lies in a vertical strip, whose minimal width is π/λ;
- parabolic of positive step ⇔ Ω lies in a vertical half-plane but in no vertical strip;
- parabolic of zero step ⇔ Ω lies in no vertical half-plane.

In the elliptic case Ω is λ-spirallike, and starlike when λ is real.

### What the tool does

**Non-elliptic:** $h(z) = i\int_0^z d\zeta/G(\zeta)$ along the segment from 0, by 12-point Gauss–Legendre quadrature on 23 subintervals whose endpoints $1-2^{-k}$ (k ≤ 22) crowd toward z. This keeps the error small when z is close to a boundary singularity of 1/G.

**Elliptic:** in the normalised chart $h_0(u) = u\exp\int_0^u\big(-\lambda/G_0(s) - 1/s\big)\,ds$, so $h_0'(0)=1$, and $h = h_0\circ u$. The integrand is regular at 0 but suffers cancellation there, so when it is a rational function it is first cancelled symbolically.

**Shape of Ω** (non-elliptic): the supremum and infimum of $\operatorname{Re}h$ are computed on the circles $|z| = 1-2^{-k}$, k = 2…18, with 2048 angles plus 160 angles clustered around each boundary fixed point at distances $10^{-2}$ to $10^{3}$ times $1-|z|$, where h blows up. A sequence is called bounded when its last increment is below $10^{-5}$ of its scale, or when its last six increments shrink by a factor below 0.8 per step; it is called unbounded when the per-step ratio of its increments stays above 0.93 and the last increment is still above $10^{-4}$ of its scale. Anything in between is reported as inconclusive. For a strip, both limits are Aitken-extrapolated and $\lambda \approx \pi/\text{width}$ is shown as an independent check of λ.

### What to expect

Values of h inside D are accurate to near double precision (the Koebe example $h = z/(1-z)^2$ is reproduced to $10^{-14}$). The shape verdict is a heuristic about an unbounded set, decided from $|z| \le 1-2^{-18}$: it agrees with theory on all library examples, but treat it as evidence. The formal antiderivative shown under the picture comes from SymPy with principal branches and can differ from h by branch choices; the plotted h never uses it.

<!-- code: koenigs -->

<!-- code: koenigs_geometry -->

<!-- section: speeds -->

## Speeds, rates and slope :orange-badge[numerical estimate]

### Theory

For non-elliptic semigroups Bracci introduced three speeds of an orbit. Let π(z) be the hyperbolic projection of z onto the diameter from −τ to τ:

$$
\begin{aligned} v(t) &= k_{\mathbb D}\big(0,\varphi_t(z_0)\big), \\ v_o(t) &= k_{\mathbb D}\big(0,\pi(\varphi_t(z_0))\big), \\ v_T(t) &= k_{\mathbb D}\big(\varphi_t(z_0),\pi(\varphi_t(z_0))\big). \end{aligned}
$$

These are the total, orthogonal and tangential speeds; up to bounded terms $v \approx v_o + v_T$. For hyperbolic semigroups $v_o(t)$ grows like $\lambda t/2$; for parabolic ones the growth is logarithmic. The **slope** is the direction of approach, $\theta(t) = \arg(1-\bar\tau\varphi_t(z_0))\in(-\pi/2,\pi/2)$: convergence is non-tangential when θ stays away from ±π/2.

### What the tool does

The orbit is integrated in $\mathbb H$ with rtol $10^{-11}$, to $t = 35/\lambda$ (hyperbolic) or $t = 10^4$ (parabolic), at 320 log-spaced times. In H the diameter becomes the positive real axis and its perpendicular geodesics are the circles $|w| = r$, so everything has a closed, cancellation-free form:

$$
\begin{aligned} v_o &= \tfrac12\big|\log|w|\big|, & v_T &= -\tfrac12\log\tan\tfrac{\varepsilon}{2}, \quad \varepsilon = \arctan\frac{\operatorname{Re}w}{|\operatorname{Im}w|},\\ |\varphi_t-\tau| &= \frac{2}{|w+1|}, & 1-|\varphi_t|^2 &= \frac{4\operatorname{Re}w}{|w+1|^2}, \qquad \theta = -\arg(w+1). \end{aligned}
$$

The hyperbolic distance in H is computed as $k_{\mathbb H}(a,b) = \tfrac12\log\frac{1+\rho}{1-\rho}$ with $\rho = |a-b|/|a+\bar b|$ and $1-\rho = 4\operatorname{Re}a\operatorname{Re}b/(|a+\bar b|(|a+\bar b|+|a-b|))$, so it stays accurate when $\rho$ is close to 1.

The fits table fits the last 30% of the window. For hyperbolic semigroups it fits a linear rate of $-\log|\varphi_t-\tau|$ (expected λ) and slopes of $v$ and $v_o$ (expected λ/2). For parabolic semigroups it fits a power α in $|\varphi_t-\tau|\sim Ct^{-\alpha}$ and coefficients c in $v\sim c\log t$.

**Elliptic:** the orbit is integrated in the normalised chart u, to $t = 35/\operatorname{Re}\lambda$. It reports $|\varphi_t-\tau|$, $k_{\mathbb D}(\tau,\varphi_t)$, the unwrapped argument, and fits of the rate (expected $\operatorname{Re}\lambda$) and rotation (expected $\operatorname{Im}\lambda$).

### What to expect

The time series are accurate; the fitted constants describe the chosen window only. Comparing the fit with λ is a useful self-test: in the library examples the hyperbolic slopes match λ/2 to $10^{-4}$.

<!-- code: asym_nonelliptic -->

<!-- code: k_H -->

<!-- code: asym_elliptic -->

<!-- section: pictures -->

## The pictures

- **Flow lines.** 36 seeds on the circles of radius 0.3, 0.6 and 0.88 are integrated forwards (solid) and backwards (dotted) for $|t|\le T$. The default T is $5/\operatorname{Re}\lambda$ or $5/\lambda$, clipped to [1, 60], and 40 for parabolic semigroups. Backward lines end where they hit the circle.
- **Vector field.** Arrows on a 21 × 21 grid point along G. Their length grows like $|G|^{1/2}$, normalised by the 90th percentile, so that the direction stays readable where |G| varies over orders of magnitude.
- **Phase portrait.** Domain colouring of G: hue is arg G, and the lightness bands repeat each time |G| doubles. Zeros of G are points where all colours meet.
- **$\varphi_t(\mathbb D)$.** The slider shows the image of the circle $|z| = 1-10^{-4}$ (magenta) and of a polar grid at t = 0, T/16, T/8, T/4, T/2 and T. These come from the closed form in semigroup mode and from the flow of G otherwise.
- **Backward escape time** shows $\log_{10}T^*$ for the points that escape (see [backward orbits](#backward)). **Petals** are shaded teal (hyperbolic) or magenta (parabolic).
- **Koenigs domain.** The images of the circles $|z| = 0.2,\dots,0.999$ and of 24 radii, with the flow lines mapped by h: forward orbits become vertical lines (non-elliptic) or spirals $e^{-\lambda t}h(z)$ (elliptic). For a strip, dashed lines mark the estimated edges.

<!-- code: flow_lines -->

<!-- code: vector_field -->

<!-- code: phase_portrait -->

<!-- code: images_of_disc -->

<!-- section: limits -->

## Known limits

- Anything that depends on behaviour at ∂D or as t → ∞ is sampled. Validity, petals, the shape of Ω, the numerical step and all fits are evidence, not proof.
- A tiny positive λ cannot be told apart from λ = 0 numerically; only the exact engine decides it.
- For non-rational G, boundary fixed points are found only where |G| has a detectable minimum on the circle.
- Backward orbits that slide tangentially along the circle can exhaust the step budget and appear as numerical failures.
- Principal branches are used for all multivalued functions. If the branch you mean differs, rewrite the expression so that the principal branch is the right one; the holomorphy check reports cuts inside D.

<!-- section: references -->

## References

- F. Bracci, M. D. Contreras, S. Díaz-Madrigal, *Continuous Semigroups of Holomorphic Self-maps of the Unit Disc*, Springer Monographs in Mathematics, 2020. The source of the conventions, the Berkson–Porta formula, the classification, petals and the Koenigs model.
- E. Berkson, H. Porta, Semigroups of analytic functions and composition operators, *Michigan Math. J.* 25 (1978).
- F. Bracci's papers on the speeds of convergence of orbits of non-elliptic semigroups, for v, v_o and v_T.
- J. R. Dormand, P. J. Prince, A family of embedded Runge–Kutta formulae, *J. Comput. Appl. Math.* 6 (1980), for the integrator.
