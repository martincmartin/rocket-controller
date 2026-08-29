# Single-Stage Solver Comparison

## Scope

This document compares numerical solvers for the single-stage primer problem.
The state and costate equations are in [Primer Equations](primer-equations.md).
The selected mode sequence is in [Burn Sequence Study](burn-sequence-study.md).

The results in this document use the previous final-time coefficient $1/4$.
The current coefficient is $3/4$. Rerun the comparison after the current
cap-dependent oracle values are regenerated.

## Event-Based Single Shooting

The event-based solver integrates one initial state-costate value problem. It
uses $\Phi=0$ events to change throttle. The event order is not a solver input.
The residual path changes when the event order changes.

Initial parameter policy:

- $\alpha_0$, $t_f$, and $\lambda_{\rho,0}$ use analytical estimates.
- $\lambda_{\eta,0}$ uses the factors

  $$
  0.25,\;0.5,\;0.75,\;1,\;1.25,\;1.5,\;2,\;3,\;4.
  $$

- The rocket configuration and initial state change between test cases. The
  parameter vector does not receive random values.

The corrected residual uses the actual initial throttle when it evaluates
$H_0$. A trial is accepted only when the residual norm is below the configured
threshold.

| Solver | Accepted trials | Trials | Mean time per trial |
| --- | ---: | ---: | ---: |
| `root(method="hybr")` | $2$ | $9$ | $0.263\ \textrm{s}$ |
| `root(method="lm")` | $0$ | $9$ | $0.960\ \textrm{s}$ |
| Bounded `least_squares` | $0$ | $9$ | $0.655\ \textrm{s}$ |

The accepted `hybr` trials were two Kerbin trials. The synthetic trials did not
produce an accepted event-based single-shooting root. `lm` returned a solver
success status for trials with residual norms between $10^{-2}$ and $10^{-1}$.
The solver status is therefore not an acceptance condition.

## Fixed-Sequence Segmented Shooting

The segmented solver receives the mode sequence

$$
q=1\;\longrightarrow\;q=0\;\longrightarrow\;q=1.
$$

It performs one bounded `least_squares` solve. One residual evaluation performs
three sequential integrations and evaluates two internal switching conditions.
It does not run separate solvers for the burns or the coast.

The fixed sequence removes event-order changes from the local residual. The
solver accepts all eleven core and randomized cases and all five edge cases in
the previous-cap study. The maximum relative fuel difference from the direct
two-burn reference was approximately $8.3\times10^{-6}$.

When the final-time cap is active, the solver uses

$$
t_f-t_{\max}=0
$$

instead of $H(t_f)=0$. It does not impose both equations.

## Runtime Selection

The provisional runtime solver is fixed-sequence segmented shooting with
bounded `least_squares`. It receives analytical initial values. It does not
run the direct optimizer and it does not scan $\lambda_{\eta,0}$.

The direct optimizer remains an offline reference. It can find a lower-fuel
trajectory and falsify the primer result. It is not part of the runtime
calculation.

The event-based solver remains an event-sequence diagnostic. It is not the
runtime solver because the corrected deterministic scan produced zero accepted
trials across ninety-nine cases.

## Falsifiers

- A direct solution with lower fuel and valid terminal residuals.
- A case that requires an initial coast before the first burn.
- A case that requires more than two powered arcs.
- A singular interval where $\Phi=0$ does not converge to an on/off control.
- A current-cap rerun that changes the selected solver statistics or oracle
  event sequence.
