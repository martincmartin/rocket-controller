# Offline Oracle Finder

## Purpose

This document records the standalone offline single-stage trajectory oracle.
The implementation is in `offline/oracle_finder.py`.

The oracle will accept a 3D position, a 3D velocity, and a segment list. It
will use the first segment only. It will use a configurable effective
powered-time limit, with 150 seconds as the default. It will not model staging.

The default target radius is $680000\ \mathrm{m}$. This is 80 km above the
Kerbin body radius.

If the input state already satisfies the target orbit within tolerance, return
an empty plan immediately. Its fuel and delta-v are zero, and its mass is
unchanged.

Define `MIN_PHASE_DURATION_SECONDS = 0.02`. Convert it to normalized time for
each problem with
$\texttt{min\_phase\_duration\_tau}=\texttt{MIN\_PHASE\_DURATION\_SECONDS}/t_*`.
Accepted plans must end at a burn endpoint.
Terminal coast phases are trimmed because they do not change fuel or the target
circular orbit.

## Selected Design

Add `offline/oracle_finder.py`. It will contain a small standalone
implementation. It will not import code from `experiments/` or `examples/`.

The public function will accept physical SI inputs and optional `mu`, target
radius, and powered-time limit values. The default gravitational parameter will
be Kerbin's $3.5316\times10^{12}\ \mathrm{m^3/s^2}$.

The module will convert the input vectors into a local 2D orbital plane. It will
store the radial, tangential, and normal basis vectors in each result.

Use these normalized scales:

$$
v_* = \sqrt{\frac{\mu}{r_*}},
\qquad
t_* = \frac{r_*}{v_*}.
$$

The normalized powered-time limit is

$$
\tau_{\max}=\frac{t_{\max}}{t_*},
\qquad t_{\max}=\texttt{max\_burn\_time}.
$$

The solver state will use polar radius, radial velocity, tangential velocity,
and normalized mass. Tangential position will remain outside the solver state.

The output will derive tangential angle after integration. It will include both
polar data and 2D Cartesian position and velocity data.

## Solver Candidates

Run and compare these candidates:

1. Pure Lawden primer shooting with sigmoid continuation and hard-switch
   validation.
2. Explicit burn/coast/burn shooting.
3. Explicit coast/burn shooting.
4. Explicit burn-only shooting.

The explicit-duration candidates will use Prussing's implicit-mass convention.
They will calculate mass from cumulative powered time. They will not use a mass
costate.

Their controls will be thrust acceleration and thrust direction. A burn uses

$$
\mathbf a_T=\frac{T}{m}\hat u_T,
\qquad
\hat u_T=\cos(\alpha)e_r+\sin(\alpha)e_t.
$$

A coast uses zero thrust acceleration.

The Lawden candidate will explicitly integrate mass and its mass costate. The
Lawden switching function will be

$$
\Phi=\frac{P}{\eta}+\frac{\lambda_\eta}{\kappa},
\qquad
P=\sqrt{p_r^2+p_t^2}.
$$

The sigmoid throttle will be

$$
q_\varepsilon=\operatorname{expit}(\Phi/\varepsilon).
$$

A smooth result will not be accepted as the final oracle unless its hardened
trajectory reaches the target and passes all arc-sign checks.

## Domain Rule

Use the current first-outbound time domain. Set its limit to the next initial
apoapsis plus three quarters of the initial orbital period:

$$
\tau_{\mathrm{limit}}
=\tau_{\mathrm{apo},0}+\frac{3}{4}\tau_{\mathrm{period},0}.
$$

This rule prevents later-revolution coast duplicates. It also matches the
existing Kerbin reference data. The task review should confirm this assumption.

If a candidate reaches the powered-time limit, emit a runtime warning and mark
the result. This identifies a trajectory constrained by the configured budget.

Hard Lawden propagation must canonicalize terminal phases. It must remove a
terminal coast, including a zero-duration event artifact, and report the end of
the last burn as the terminal time. Compute residuals and objectives at that
endpoint. Short internal phases below 0.02 seconds are not implementable and
must be rejected or re-solved. Use the named constant for this check.

Supported Lawden sequences are `burn/coast/burn`, `coast/burn`, `burn`, and the
empty sequence for an already-circular input. Any other sequence is rejected,
reported in diagnostics, emitted as a runtime warning, and shown by the CLI.

Explicit-duration candidates will use a zero switching function at the end of
each segment, including the final burn. They will not use a final Hamiltonian
residual. At the target circular state, the non-thrust Hamiltonian is zero.
Therefore $H(t_f)=\Gamma(t_f)(1-P(t_f))$, so the final switch residual is
equivalent to $H(t_f)=0$ when $\Gamma(t_f)>0$.

The time limit remains an elapsed-time bound. It is not an additional residual.

## Initial Values

Build a deterministic impulse-based seed. Compute the initial osculating
apoapsis. If it is below the target, use vis-viva and angular momentum to find
the prograde impulse in the initial velocity direction that raises the
apoapsis to the target. Do not use a numerical root finder.

With $k=r v_t/|v|$ and target apoapsis $r_a^*$, calculate

$$
v_1^+=
\sqrt{\frac{2\mu(1/r-1/r_a^*)}
{1-k^2/(r_a^*)^2}},
\qquad
\Delta v_1=v_1^+-|v|.
$$

Convert each impulse to finite burn time with

$$
t_b(\Delta v,m)=\frac{m v_e}{T}
\left(1-\exp\left(-\frac{\Delta v}{v_e}\right)\right).
$$

After the first impulse, estimate the second impulse at the new apoapsis. Use
the circular velocity difference, convert it with the remaining mass, and set

$$
t_c=\max\left(0,
t_{\mathrm{apo},1}-t_{b1}-\frac{t_{b2}}{2}\right).
$$

If the initial apoapsis is already above the target, skip the first burn. Use
the current apoapsis circularization estimate as the seed. The explicit solve
will still enforce the exact target state.

For explicit candidates, set the initial primer magnitude to $P_0=1$. Set its
direction to the initial velocity direction. Estimate $\lambda_{\rho,0}$ from
$H(0)=0$ only as a seed. Near apoapsis, use bounded fallback values instead of
dividing by a small radial velocity.

Also use the existing deterministic direction sweep: initial velocity,
tangential, and predicted-apoapsis-prograde directions. Do not copy the old
mass-costate sweep because explicit candidates have no mass costate.

## Result Data

Return a typed result and a JSON form. The result will contain:

- input vectors, target, normalization, and first-segment values;
- the 2D orbital-plane basis and its 3D mapping;
- one summary for every candidate;
- the selected candidate and its initial costate;
- final mass, fuel consumed, and delta-v consumed;
- phase boundaries and full burn/coast trajectory arrays;
- state, costate, thrust acceleration, thrust direction, and switching data;
- Lawden throttle data where the sigmoid candidate is reported;
- solver messages, epsilon history, residuals, and elapsed time.

Each phase will use a `solve_ivp`-style time array and parallel value arrays.
The output will identify whether its costates use the Lawden or Prussing model.

## Kerbin Examples

Include these fixtures without importing experimental modules:

- `kerbin-example`;
- `kerbin-first-example`;
- `kerbin-coast-first-example`.

The explicit-duration comparison is expected to find burn/coast/burn,
burn/coast/burn, and coast/burn for the three fixtures. The Lawden candidate
may find and select a supported lower-fuel sequence after hard polishing.

Every selected result must end with a burn phase unless it is the empty plan. A
burn/coast/burn result with a first burn below
`MIN_PHASE_DURATION_SECONDS` is treated as coast/burn instead.

## Options Considered

| Option | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Import `single_stage_research.py` | Reuses equations and fixtures | Couples the oracle to experimental code | Reject |
| Wrap `production_solver.py` | Reuses sigmoid offline settings | Can accept a fractional or wrong event structure | Reject |
| Copy all experiment modules | Maximum code reuse | Preserves the maintenance baggage | Reject |
| Build a small module in `offline/` | Clear API and no experimental dependency | Repeats a small equation implementation | Choose |
| Use only explicit arc families | Reliable event selection | Omits the requested Lawden comparison | Reject |
| Use only pure Lawden | General mode discovery | Known stiffness and event-loss failures | Reject |
| Run both solver types | Supports comparison and fallback | Requires more runtime and diagnostics | Choose |
| Add a direct angle-mesh optimizer | Independent falsification | Adds scope and another result model | Defer |

## Validation Plan

The implementation must test 3D-to-2D conversion, polar dynamics, positive
mass, the 150-second limit, objective consistency, output shape, and the
equivalence of final switching and Hamiltonian residuals at the target.
It must also test terminal-coast trimming and the 0.02-second phase threshold.

Run all three examples. Check target residuals, switching signs, expected family
selection, and agreement with the documented normalized references.

Run the full fixture check with `python3 -m offline.validate_examples`.

Run `./validate.sh`. Verify the limit warning. Measure the complete example
run. Keep it below ten minutes.

## Validation Result

The three Kerbin examples complete in about 27 seconds on the development
machine. The selected families and normalized fuel fractions are:

| Example | Family | Fuel fraction | Residual norm |
| --- | --- | ---: | ---: |
| `kerbin-example` | burn/coast/burn | $0.458673471$ | $8.23\times10^{-12}$ |
| `kerbin-first-example` | Lawden burn/coast/burn | $0.417870055$ | $7.81\times10^{-13}$ |
| `kerbin-coast-first-example` | Lawden burn/coast/burn | $0.339812664$ | $4.25\times10^{-14}$ |

The full repository validation passed. It reported 118 passing tests. The normal
unit suite uses one shared end-to-end fixture result; the three-fixture check is
explicit so routine validation does not repeat the expensive solves.
