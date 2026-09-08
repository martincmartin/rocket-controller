# Plan: Independent Closed-Form Seeding For The Lawden Solver

## Status

Approved and implemented. See Results.

## Results

- `_lawden_seeds` builds eight closed-form seeds. `_solve_lawden(problem, timing)`
  runs the full epsilon ladder and polish for every seed. `find_oracle` no
  longer passes any explicit result into the Lawden solver.
- Independence test `test_lawden_succeeds_without_explicit_results` forces
  every explicit mode to fail; the Lawden candidate still succeeds on
  `kerbin-first-example`. `./validate.sh` passes (120 tests).
- All three fixtures keep their reference results: `kerbin-example` selects
  burn/coast/burn (fuel 6368.979455 kg), the other two select Lawden
  burn/coast/burn with fuel 5456.189701 kg and 4002.051062 kg, all matching
  `docs/single-stage-test-cases.md`. On these fixtures every accepted seed
  converges to the same root, so the multistart confirms rather than widens
  the result. On `kerbin-example` all eight seeds hit the known
  finite-difference plateau at residual about $5\times10^{-5}$.
- Lawden wall time per fixture: 8.0 s (`kerbin-first-example`), 29.8 s
  (`kerbin-example`), 30.0 s (`kerbin-coast-first-example`). The three-fixture
  `validate_examples` run takes about 85 s. The ten-minute budget holds.
- Empirical adjustment: the fourth final-time shape was
  $0.8\tau_{\mathrm{limit}}$. That guess always exceeded the integration
  domain and returned the invalid-residual sentinel on all three fixtures. It
  is now bounded: $\min(0.8\tau_{\mathrm{limit}}, 1.5\tau_{f,\mathrm{base}})$.
  With this bound every seed contributes.

## Problem

`_solve_lawden` starts from the converged output of the explicit arc-time
solvers. `find_oracle` selects the best explicit candidate and passes it into
the Lawden solver (`offline/oracle_finder.py`, `find_oracle` near the call
`_solve_lawden(problem, explicit_seed)`). Inside `_solve_lawden`, when that
candidate is a successful burn/coast/burn result, the code copies its converged
primer angle, final time, and radial costate into the Lawden seed.

This creates a circular result:

1. The Lawden candidate can only confirm the structure it was given.
2. The main reason to run Lawden is to find trajectories outside the three
   explicit families (burn/coast/burn, coast/burn, burn). Lawden can represent
   a general bang-bang sequence. It cannot discover one when it starts on the
   solution of a fixed-sequence solver.
3. The explicit result is "solved" information. The task for the Lawden solver
   is to work from "computed" information, like the explicit solvers do.

The closed-form seed machinery already exists. The explicit solvers use it:

- `_impulse_timing_seed` computes the two-impulse timing estimate
  (burn to raise apoapsis, coast, burn to circularize). It is closed form.
- `_prograde_apoapsis_angle` computes the third primer direction.
- `_lambda_rho_seed` computes the radial costate from $H(0)=0$.
- `_explicit_seeds` combines three directions, three $\lambda_\rho$ factors,
  and three timing variants into 27 seeds per mode.

The Lawden solver uses none of this, except as a fallback when every explicit
solver failed.

## Goal

Change the Lawden candidate so that:

1. It never reads the output of an explicit solver.
2. It starts from the same closed-form estimates as the explicit solvers.
3. It starts from a collection of those estimates with deterministic
   variations, so one seed does not decide the discovered structure.

## Lawden Seed Vector

The Lawden shooting vector is

$$
(\alpha_0, \tau_f, \lambda_{\rho,0}, \lambda_{\eta,0}).
$$

The bounds are $\alpha_0\in[-\pi,\pi]$, $\tau_f\in[10^{-5},\tau_{\mathrm{limit}}]$,
and the costates in $[-100,100]$. The current analytic parts are:

- $\lambda_{\eta,0}=-\kappa$. Keep this value.
- $\tau_f$ from the impulse timing or a fraction of the time limit.
- $\alpha_0$ from a deterministic direction.
- $\lambda_{\rho,0}$ from `_lambda_rho_seed`, which solves the initial
  Hamiltonian equality for $\lambda_{\rho,0}$ given the direction.

## Proposed Seed Set

Build one function `_lawden_seeds(problem, timing)` that returns seed vectors
without any solved quantities.

Directions, the same three used by the explicit solvers:

$$
\hat u_1=\frac{(u_{r,0},u_{t,0})}{|(u_{r,0},u_{t,0})|},
\qquad
\hat u_2=(0,1),
\qquad
\hat u_3=\text{prograde at predicted apoapsis}.
$$

Final-time shapes, computed from `TimingSeed` (burn $\tau_{b1}$, coast
$\tau_c$, burn $\tau_{b2}$):

$$
\tau_f \in \left\{
\tau_{b1}+\tau_c+\tau_{b2},\;
\tau_c+\tau_{b2},\;
\min(\tau_{b1}+\tau_{b2},\,0.95\tau_{\max}),\;
0.8\tau_{\mathrm{limit}}
\right\}.
$$

These shapes mirror the three explicit families and the existing fallback,
without solving for any duration. Each value is clipped into the bounds.

Radial costate values, per direction:

$$
\lambda_{\rho,0} \in \left\{0.5\hat\lambda_\rho,\;\hat\lambda_\rho,\;2\hat\lambda_\rho\right\},
\qquad
\hat\lambda_\rho=\texttt{\_lambda\_rho\_seed}(u).
$$

Near apoapsis the helper returns zero, so the same fallback set
$\{-1,0,1\}$ as `_explicit_seeds` applies.

Seed matrix. Full Cartesian product gives
$3\times4\times3=36$ seeds. Each Lawden run is the eight-step epsilon ladder
plus the polish. That product is too expensive for the offline budget. Use
this reduced, structured set instead:

- Base seed: direction $\hat u_1$, base $\tau_f=\tau_{b1}+\tau_c+\tau_{b2}$,
  base $\hat\lambda_\rho$.
- All three directions at the base $\tau_f$ and base $\hat\lambda_\rho$
  (2 extra seeds).
- All four $\tau_f$ shapes at direction $\hat u_1$ and base $\hat\lambda_\rho$
  (3 extra seeds).
- The two remaining $\lambda_\rho$ factors at $\hat u_1$ and base $\tau_f$
  (2 extra seeds).

Total: $1+2+3+2=8$ seeds. Measure the runtime of this set on all three Kerbin
fixtures. If one example exceeds two minutes for the Lawden family, apply a
prescreen: run only the first epsilon step ($\varepsilon=1.0$) for every seed,
rank by residual, and continue the full ladder for the best four seeds. Record
which seeds were dropped.

## Changes

1. Add `_lawden_seeds(problem, timing)` as defined above.
2. Change `_solve_lawden(problem, explicit)` to
   `_solve_lawden(problem, timing)`. Remove the branch that copies the
   explicit candidate's parameters.
3. In `find_oracle`, compute `timing` once and pass it to both the explicit
   modes and the Lawden solver. Remove the explicit-seed selection loop.
4. Run every Lawden seed through the existing ladder and polish. Keep the
   existing acceptance checks. Select the best accepted seed result with
   `_candidate_is_better`.
5. Record per-seed results in `seed_results`, like `_solve_explicit_mode`
   does. Keep the epsilon history of the selected seed.
6. Update `docs/offline-oracle-finder.md` with the new seed policy.

No other solver behavior changes. The epsilon ladder, the polish budget, and
the supported-sequence checks stay as they are.

## Expected Effects

- The Lawden candidate becomes a true independent cross-check.
- On the fixtures where Lawden wins today, it must still converge to the same
  reference trajectory from the closed-form seeds. The plan must verify this.
  If a fixture regresses, extend the seed set. Do not restore explicit
  seeding.
- The coast-first fixture may now classify as coast/burn instead of
  burn/coast/burn with a 0.036 s first burn. That is an acceptable outcome of
  independent structure discovery. Update the expected-selection table after
  the empirical run.
- Runtime grows by the seed count. The current Lawden runs take about
  $1.0$ s to $3.8$ s per example. Eight seeds could reach about
  $8$ s to $30$ s per example. The ten-minute task budget still holds, but
  the plan must measure it.

## Independence Test

Add a test that forces every explicit mode to fail and verifies the Lawden
candidate still succeeds on `kerbin-first-example` and
`kerbin-coast-first-example` from the closed-form seeds alone. This test
fails if any code path reads an explicit result. It uses monkeypatching,
like the existing `test_all_candidates_failed_error_path`.

## Validation

1. Run `./validate.sh`.
2. Run `python3 -m offline.validate_examples` and compare fuel, final time,
   and phase sequences with `docs/single-stage-test-cases.md`.
3. Measure Lawden-only time for the three fixtures with the new seed set.
4. Confirm `seed_results` appears in the Lawden diagnostics JSON.

## Options Considered

| Option | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Keep explicit-seeded Lawden | No code change | Circular; cannot discover new structure | Reject |
| One closed-form seed | Small change | One initial position can miss the right basin | Reject |
| Full 36-seed product | Complete coverage | About 36 times the current Lawden cost | Reject |
| Structured 8-seed set with optional prescreen | Independent multistart, bounded cost | Some seed design work | Choose |
| Prescreen at $\varepsilon=1.0$ only | Cheapest multistart | Smooth residual at large epsilon can rank basins wrong | Reject as primary |
