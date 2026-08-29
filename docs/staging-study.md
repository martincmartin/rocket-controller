# Two-Stage Staging Study

> Historical fixed-schedule baseline. The current staged formulation is in
> [Multi-Stage Primer Study](multi-stage-primer-study.md). This document keeps
> the earlier fixed full-stage-1 experiment for comparison.

## Scope

This document extends the normalized polar research model to one hardware
staging event. The experiment uses the supplied Kerbin-like vehicle data and a
fixed two-second ballistic interval between the stages. It does not change
`sim.py`, `sim_experiment.py`, or the flight controller.

The schedule studied here is

$$
\text{stage 1 at full thrust}
\;\longrightarrow\;
\text{2 s coast and jettison}
\;\longrightarrow\;
\text{stage 2 at full thrust}.
$$

The first-stage burn is fixed at its full-burn duration. The second-stage burn
duration is a solved variable. This is a fixed hardware schedule, not a proof
that unrestricted throttle control has the same sequence.

The executable implementation is
`experiments/staged_two_arc_research.py`.

## Physical Fixture

The body and target are

$$
\mu=3.5316\times10^{12}\ \mathrm{m^3/s^2},
\qquad
R=600000\ \mathrm{m},
\qquad
r_\star=680000\ \mathrm{m}.
$$

The initial inertial vectors are the `kerbin-example` vectors used by the
existing research experiments. The local normalized state is

$$
x_0=(\rho_0,u_{r,0},u_{t,0},\eta_0)
=(0.932317760533369,
0.252573095512292,
0.199946283806988,
1).
$$

The normalization is

$$
v_\star=\sqrt{\frac{\mu}{r_\star}}
=2278.931638238564\ \mathrm{m/s},
\qquad
t_\star=\frac{r_\star}{v_\star}
=298.385431396962\ \mathrm{s}.
$$

The stage data are:

| Stage | Start mass (kg) | Thrust (N) | $v_e$ (m/s) | Full burn (s) | End mass (kg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swivel | $13885.650391$ | $215000$ | $3138.128$ | $59.050096$ | $9839.999404$ |
| Terrier | $4449.999408$ | $60000$ | $3383.29425$ | $112.776473$ | $2449.999451$ |

The first-stage dry-stage jettison is therefore

$$
m_{\mathrm{jettison}}
=9839.999404-4449.999408
=5389.999996\ \mathrm{kg}.
$$

The jettison is $38.817051\%$ of the original mass. The second stage starts
with $32.047468\%$ of the original mass. This mass transition must not be
represented as continuous propellant flow through the staging interval.

## Normalized Stage Data

Use one reference mass, the initial first-stage mass, for both stages:

$$
m_{\mathrm{ref}}=13885.650391\ \mathrm{kg}.
$$

For each stage $i$,

$$
\gamma_i=\frac{T_i r_\star}{m_{\mathrm{ref}}v_\star^2},
\qquad
\kappa_i=\frac{v_{e,i}}{v_\star}.
$$

The experiment obtains:

| Stage | $\gamma_i$ | $\kappa_i$ | Full burn $\tau_i$ | Start $\eta$ | End $\eta$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swivel | $2.027302475465$ | $1.377017172145$ | $0.197898723556$ | $1.000000000000$ | $0.708645193196$ |
| Terrier | $0.565758830362$ | $1.484596638719$ | $0.377955693171$ | $0.320474683025$ | $0.176441101566$ |

The fixed staging interval is

$$
\tau_g=\frac{2\ \mathrm{s}}{t_\star}
=0.006702740112466.
$$

The normalized equations use the active stage's $\gamma_i$ and
$\kappa_i$. The values must not be copied from stage 1 into stage 2.

## State Propagation

The state is

$$
x=(\rho,u_r,u_t,\eta).
$$

For stage $i$ and throttle $q$,

$$
\begin{aligned}
\rho'&=u_r,\\
u_r'&=\frac{u_t^2}{\rho}-\frac{1}{\rho^2}
      +\frac{q\gamma_i}{\eta}\cos\alpha,\\
u_t'&=-\frac{u_ru_t}{\rho}
      +\frac{q\gamma_i}{\eta}\sin\alpha,\\
\eta'&=-\frac{q\gamma_i}{\kappa_i}.
\end{aligned}
$$

The propagation order is explicit:

1. Integrate stage 1 for its fixed full-burn duration with $q=1$.
2. Subtract the fixed normalized jettison mass from $\eta$.
3. Verify that the result equals the configured stage-2 start mass.
4. Integrate the coast for $\tau_g$ with $q=0$.
5. Integrate stage 2 for the solved duration with $q=1$.

The implementation subtracts a fixed jettison amount instead of assigning an
unrelated absolute mass at the boundary. At the nominal trajectory both forms
give the same mass. The subtraction states the required derivative of the
hybrid mass map:

$$
\eta^+=\eta^- - \Delta\eta_{\mathrm{jettison}},
\qquad
\frac{\partial\eta^+}{\partial\eta^-}=1.
$$

## Costates And Staging

Use the existing primer convention

$$
(p_r,p_t)=-(\lambda_{u_r},\lambda_{u_t}),
\qquad
P=\sqrt{p_r^2+p_t^2}.
$$

The costate equations are unchanged in form. During an active stage, substitute
that stage's $\gamma_i$ and $\kappa_i$:

$$
\lambda_\eta'=-\frac{q\gamma_iP}{\eta^2}.
$$

At the fixed jettison,

$$
\rho^+=\rho^- ,\quad
u_r^+=u_r^- ,\quad
u_t^+=u_t^- ,\quad
\eta^+=\eta^- - \Delta\eta,
$$

and the costates pass through continuously:

$$
\lambda_\rho^+=\lambda_\rho^- ,\quad
p_r^+=p_r^- ,\quad
p_t^+=p_t^- ,\quad
\lambda_\eta^+=\lambda_\eta^-.
$$

This continuity rule requires a fixed jettison mass. If the discarded mass is
state-dependent, derive a new hybrid junction condition before using this
rule.

No switching residual is imposed at the stage boundary. The boundary is a
hardware schedule boundary, not a zero of the switching function. The two
second coast is also a schedule interval, so its throttle is set to zero even
when the unconstrained switching function does not equal zero there.

## Hamiltonian Condition

For stage $i$,

$$
S_i=\frac{P}{\eta}+\frac{\lambda_\eta}{\kappa_i},
$$

and the minimized Hamiltonian is

$$
H_i=\lambda_\rho u_r
-p_r\left(\frac{u_t^2}{\rho}-\frac{1}{\rho^2}\right)
+p_t\frac{u_ru_t}{\rho}
-q\gamma_iS_i.
$$

The Hamiltonian is conserved within each constant-stage interval. It is not
globally conserved across this schedule. A throttle change, a changed stage
parameter set, and the mass jump all occur at known boundaries.

The final time is the end of the stage-2 burn and is free in this experiment.
At the exact circular target,

$$
H_2(t_f)=-\gamma_2S_2(t_f),
$$

so the final residual can use either $H_2(t_f)=0$ or $S_2(t_f)=0$. The
experiment uses $S_2(t_f)=0$. It does not use $H(0)=0$.

If a future solution reaches the stage-2 full-burn limit, the burn-cap
condition replaces the free-final-time condition. Do not impose both the cap
condition and $H_2(t_f)=0$.

## Solver Structure

### Direct Reference

The direct reference uses four, eight, sixteen, or thirty-two constant-angle
intervals in each powered stage. Its parameters are

$$
z_D=(\tau_{b2},
\alpha_{1,1},\ldots,\alpha_{1,N},
\alpha_{2,1},\ldots,\alpha_{2,N}).
$$

Stage 1 has a fixed duration and stage 2 has the only variable duration. The
three terminal orbit equations are equality constraints. The objective is the
stage-2 propellant fraction; stage-1 propellant use is fixed by the schedule.
The total propellant fraction is also reported.

The direct reference is independent of costates. Mesh refinement tests whether
the piecewise-angle result approaches the primer result.

### Fixed-Schedule Primer Solve

The primer parameters are

$$
z_P=(\alpha_0,\lambda_{\rho,0},\lambda_{\eta,0},\tau_{b2}).
$$

One residual evaluation integrates the three known intervals in order and
returns

$$
\mathcal R_P=
\begin{bmatrix}
\rho_f-1\\
u_{r,f}\\
u_{t,f}-1\\
S_2(t_f)
\end{bmatrix}.
$$

The costate vector is carried through the mass jump. It is not reinitialized
from the stage-2 mass or from stage 2's $\kappa$.

The experiment uses bounded `scipy.optimize.least_squares` with deterministic
initial guesses and accepts a result only when the solver reports success and
the residual norm is below $2\times10^{-6}$.

## Numerical Results

Command used for the final mesh result:

```text
PYTHONPATH=experiments python3 experiments/staged_two_arc_research.py \
  --angles 32 --direct-attempts 4 --primer-attempts 4
```

The integration tolerances were `rtol=2e-11` and `atol=2e-13`, using DOP853.

### Direct Mesh Refinement

| Angle intervals per stage | Stage-2 burn $\tau$ | Stage-2 burn (s) | Total propellant fraction | Final mass fraction | Residual norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | $0.200904757795$ | $59.947053$ | $0.367916775098$ | $0.243912714731$ | $1.06\times10^{-11}$ |
| 8 | $0.200734284015$ | $59.896186$ | $0.367851809946$ | $0.243977679883$ | $9.94\times10^{-13}$ |
| 16 | $0.200692071950$ | $59.883590$ | $0.367835723524$ | $0.243993766305$ | $4.22\times10^{-12}$ |
| 32 | $0.200681544080$ | $59.880449$ | $0.367831711501$ | $0.243997778328$ | $1.18\times10^{-12}$ |

The stage-2 burn and propellant fraction decrease as the direct mesh is
refined. The change from 16 to 32 intervals is
$4.01\times10^{-6}$ in total propellant fraction.

### Primer Result

The fixed-schedule primer solve returned:

| Quantity | Value |
| --- | ---: |
| $\alpha_0$ | $1.009579096315$ rad |
| $\lambda_{\rho,0}$ | $-3.215859496659$ |
| $\lambda_{\eta,0}$ | $-4.775775123151$ |
| Stage-2 burn | $0.200678036950$ normalized, $59.879403$ s |
| Total elapsed schedule | $120.929499$ s |
| Total propellant fraction | $0.367830374984$ |
| Final mass fraction | $0.243999114845$ |
| Terminal residual norm | $2.68\times10^{-15}$ |

The primer result differs from the 32-interval direct result by

$$
\Delta f=-1.34\times10^{-6},
\qquad
\Delta\eta_f=+1.34\times10^{-6},
$$

which is $0.0186$ kg at the reference mass. The direct result is still mesh
discretized; the small lower primer propellant value is not a global-optimality
claim.

### Boundary Diagnostics

The primer trajectory reported these local Hamiltonian drifts:

| Interval | Hamiltonian start | Hamiltonian end | Drift |
| --- | ---: | ---: | ---: |
| Stage-1 burn | $4.826886304669$ | $4.826886304668$ | $-3.66\times10^{-13}$ |
| Staging coast | $-0.499425471935$ | $-0.499425471935$ | $0$ |
| Stage-2 burn | $-1.18\times10^{-12}$ | $-2.47\times10^{-15}$ | $1.18\times10^{-12}$ |

At staging,

$$
\eta^- =0.708645193196,
\qquad
\eta^+=0.320474683025,
$$

and the normalized mass-costate jump was zero to machine precision:

$$
\lambda_{\eta}^+-\lambda_{\eta}^-=0.
$$

The measured value was exactly $0$ in the stored result. The Hamiltonian changed from
$4.826886304668$ before the scheduled transition to
$-0.499425471935$ at the start of the coast. This confirms that a global
$H(0)$ conservation residual is not valid for the staged schedule.

The final switching residual and final Hamiltonian were approximately

$$
S_2(t_f)=3\times10^{-15},
\qquad
H_2(t_f)=-2\times10^{-15}.
$$

The independent physical-unit check was run with

```text
PYTHONPATH=experiments python3 experiments/staging_unit_checks.py
```

It propagated the same stage schedule once in normalized units and once in
physical polar units. The maximum normalized trajectory difference was
$1.01\times10^{-13}$. The stage-1 and stage-2 mass-budget errors were
$1.11\times10^{-16}$ and $8.33\times10^{-17}$. During the two-second coast,
the normalized specific-energy drift and angular-momentum drift were both
reported as $0$ at the displayed precision.

The powered-arc switching function was negative over much of both fixed
full-thrust arcs. This is an important scope result: the run validates the
fixed schedule and the terminal primer equations, but it does not validate a
free-throttle bang-bang solution. A future free-throttle solver must allow
additional coast arcs and enforce the correct switching-function sign on each
active interval.

## Problems Found In The Supplied Staged Solver

The supplied `experiments/staged_solver.py` was reviewed before this
implementation. Its returned `least_squares` status can be successful while
the terminal residual remains nonzero for the supplied physical case. One run
returned residual components approximately

$$
(-0.0380,-0.00406,-0.7997,0.0714),
$$

so solver status alone cannot be used as acceptance.

The new experiment avoids these problems:

- It uses the same radial/tangential primer component order as the existing
  single-stage equations.
- It carries a fixed mass subtraction through the staging junction and checks
  the configured stage-2 start mass.
- It integrates the known stage, gap, and stage-2 intervals explicitly. It does
  not infer a window from a floating-point boundary time.
- It uses $S_2(t_f)=0$ instead of an initial Hamiltonian residual.
- It reports local Hamiltonian drift and costate continuity at the junction.
- It rejects nonzero terminal residuals even when SciPy reports optimizer
  success.

## Implementation Path

The production planner should adopt the following data and control boundaries
after this research result is accepted:

1. Store each stage's thrust, exhaust velocity, full-burn capacity, and start
   mass as stage data. Do not store only one global thrust or exhaust velocity.
2. Convert every stage to common normalized units using the original vehicle
   mass as the reference mass.
3. Represent staging as an explicit phase list: active stage 1, fixed coast and
   jettison, active stage 2.
4. Propagate mass continuously only during powered phases. Apply the fixed dry
   mass subtraction once at separation.
5. Carry position, velocity, and costates through the junction. Verify the
   mass map and costate continuity numerically.
6. Use the active stage's $\gamma$ and $\kappa$ in every state, costate, primer,
   and Hamiltonian evaluation.
7. Use the final Hamiltonian or final-stage switching residual for an interior
   final time. Use only the active burn-cap condition when a duration bound is
   active.
8. Keep the direct staged transcription as an offline reference. Do not place
   it in the runtime path.

## Falsifiers And Open Work

This result does not establish a globally optimal staged trajectory. The next
experiments should test:

- A free-throttle staged solver with event detection inside each active stage.
- A first-stage coast or early shutdown before jettison.
- A stage-2 coast followed by a restart.
- A variable or state-dependent jettison mass.
- Several staging gaps, including zero and longer gaps.
- Direct meshes beyond 32 intervals and independent multistart sets.
- A Cartesian physical-unit propagation check across the mass jump.
- A lower-propellant valid trajectory with a different event sequence.

Any lower-propellant valid trajectory, or any failed costate junction check,
requires changing the provisional staged formulation.
