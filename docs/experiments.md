# Experiments Catalog

## Purpose

This catalog describes every Python source program under `experiments/` and its
subfolders. It records why each program exists, what it does, and how it may
support future work.

The catalog contains 37 Python programs:

- 20 programs in `experiments/`.
- 17 programs in `experiments/sigmoid_primer_review/`.

The generated files in `experiments/__pycache__/` are not listed as separate
experiments. They are Python bytecode copies of source programs. They do not
contain an independent research question or purpose.

## Scope Decisions

Several catalog formats were possible. The selected format is based on the
following trade-offs.

| Option | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Include `.py` and `.pyc` files | Lists every filesystem entry | Lists generated duplicates and gives no useful source context for bytecode | Reject |
| List only root-level scripts | Short document and easy scanning | Omits the complete sigmoid review series | Reject |
| Use one large summary table | Compact and easy to search | Long technical descriptions make the table hard to read | Reject |
| Group source files by research line and give each file three descriptions | Shows relationships and preserves the purpose, implementation, and future use | Produces a longer document | Choose |

## Production Direction

The production planner will use explicit arc durations. It will solve these
single-stage arc families:

- burn/coast/burn
- coast/burn
- burn-only

Each arc duration is a solver parameter. The switching function is zero at the
end of each required arc, and each such condition is a residual. The planner
must select among the supported arc families. It must not infer the mode
sequence only by integrating a pure Lawden event system.

The following alternatives were considered during the experiments:

| Approach | Benefit | Problem | Production decision |
| --- | --- | --- | --- |
| Pure Lawden event mode selection | Few explicit timing variables and a direct primer interpretation | Event order changes the residual path; initial coast and weakly constrained switches are difficult to recover | Reject as the production direction |
| Sigmoid switching and continuation | Smooth residual path and a possible initial-coast seed | It can produce fractional throttle, wrong event structure, and stiff continuation | Reject as the production direction; retain only historical evidence |
| Explicit arc durations with switching residuals | Directly represents the three required families and makes each boundary testable | More variables and separate branch handling are required | Choose |

## Retention Guidance

The recommendations below describe what to keep in the working research set.
`Archive before deletion` means that the question is settled and the program is
not needed for the new solver. Keep the source or its result in project history
until the production implementation has passed the corresponding checks.

### Keep: Production-Solver References And Debugging

- `two_burn_reference.py`: authoritative direct oracle generator for the current two-burn family. Extend it for coast/burn and burn-only.
- `two_arc_shooting.py`: closest existing explicit-duration solver. Use it as a migration reference, not as the production API.
- `initial_guess_comparison.py`: timing and direction seeds for explicit-duration solves.
- `runtime_benchmark.py`: baseline for solve time and acceptance checks.
- `edge_two_arc_shooting.py`: direct-versus-shooting comparison harness to extend to all three families.
- `time_bound_sweep.py`: final-time domain and branch diagnostic.
- `multi_stage_primer.py`: staged direct-reference machinery and staged solver diagnostic.
- `sigmoid_primer_review/two_arc_residual_variant.py`: evidence for terminal switching residuals.

### Keep: General Validation And References

- `single_stage_research.py`: normalized equations, fixtures, direct propagation, and validation checks.
- `body_scaling_sweep.py`: physical-unit scaling check.
- `single_stage_boundary_checks.py`: exact-circular and near-radial boundary checks.
- `single_stage_edge_cases.py`: robustness fixtures and switching-sign diagnostics.
- `compare_single_stage_references.py`: direct-versus-relaxed falsification data.
- `multi_burn_reference.py`: offline three-burn falsifier.
- `staged_two_arc_research.py`: historical staging and mass-map reference.
- `staging_unit_checks.py`: normalized-versus-physical staging check.
- `multi_stage_unit_checks.py`: generalized staging transition check.
- `analyze_optimal_burn.py`: physical trajectory and thrust-direction diagnostics.
- `sigmoid_primer_review/sharp_oracle_profile.py`: per-arc switching-function diagnostic.
- `sigmoid_primer_review/sharp_residual_check.py`: structural-validity falsifier.

### Archive Before Deletion

These programs answer settled questions about rejected event-based or sigmoid
formulations. They are not production dependencies:

- `single_stage_lambda_scan.py`
- `compare_shooting_solvers.py`
- `sigmoid_primer.py`
- Every review program under `sigmoid_primer_review/` except `sharp_oracle_profile.py`, `sharp_residual_check.py`, and `two_arc_residual_variant.py`

## Dependency And Disposition Index

Paths in this table are relative to `experiments/`. `R/` means
`sigmoid_primer_review/`. `Depends on` lists direct Python dependencies among
experiment programs. `Used by` lists other experiment programs that import or
materially use the program. A program can be settled and still be retained as
a validation or debugging reference.

| Program | Depends on | Used by | Question status | Disposition |
| --- | --- | --- | --- | --- |
| `single_stage_research.py` | none | `single_stage_edge_cases.py`; `single_stage_boundary_checks.py`; `body_scaling_sweep.py`; `time_bound_sweep.py`; `runtime_benchmark.py`; `edge_two_arc_shooting.py`; `multi_burn_reference.py`; `single_stage_lambda_scan.py`; `compare_single_stage_references.py`; `initial_guess_comparison.py`; `sigmoid_primer.py`; `analyze_optimal_burn.py`; `two_burn_reference.py`; `two_arc_shooting.py`; `compare_shooting_solvers.py`; `R/two_arc_residual_variant.py`; `R/free_eps_slsqp.py`; `R/h0_lambda_rho_fixed.py`; `R/sharp_residual_check.py`; `R/adaptive_eps_pipeline.py`; `R/production_solver.py`; `R/hybrid_polish.py`; `R/free_eps_shooting.py`; `R/sharp_oracle_profile.py`; `R/analytic_jacobian.py` | Partly settled: equations and scaling are settled; the production API is not | Keep: validation/reference |
| `body_scaling_sweep.py` | `single_stage_research.py` | none | Settled | Keep: validation/reference |
| `single_stage_boundary_checks.py` | `single_stage_research.py` | none | Settled | Keep: validation/reference |
| `single_stage_edge_cases.py` | `single_stage_research.py`; `two_burn_reference.py` | `edge_two_arc_shooting.py` | Settled for the existing fixture catalog; rerun for new branches | Keep: validation/reference |
| `analyze_optimal_burn.py` | `initial_guess_comparison.py`; `single_stage_research.py` | none | Settled for the non-tangential first-burn explanation | Keep: validation/reference |
| `two_burn_reference.py` | `single_stage_research.py` | `single_stage_edge_cases.py`; `time_bound_sweep.py`; `edge_two_arc_shooting.py`; `multi_burn_reference.py`; `initial_guess_comparison.py`; `sigmoid_primer.py`; `two_arc_shooting.py`; `R/two_arc_residual_variant.py` | Open: oracle generation must cover all three production families | Keep: production reference/debugging |
| `two_arc_shooting.py` | `single_stage_research.py`; `two_burn_reference.py` | `time_bound_sweep.py`; `runtime_benchmark.py`; `edge_two_arc_shooting.py`; `initial_guess_comparison.py`; `R/two_arc_residual_variant.py` | Open: explicit durations are useful, but branch support and residuals must change | Keep: production reference/debugging |
| `initial_guess_comparison.py` | `single_stage_research.py`; `two_arc_shooting.py`; `two_burn_reference.py` | `time_bound_sweep.py`; `runtime_benchmark.py`; `sigmoid_primer.py`; `analyze_optimal_burn.py` | Open: seeds must support three explicit arc families | Keep: production reference/debugging |
| `runtime_benchmark.py` | `initial_guess_comparison.py`; `single_stage_research.py`; `two_arc_shooting.py` | none | Open: old benchmark must be replaced with three-mode measurements | Keep: production reference/debugging |
| `compare_shooting_solvers.py` | `single_stage_research.py` | none | Settled: optimizer status is not a validity test for the rejected event solver | Archive before deletion |
| `single_stage_lambda_scan.py` | `single_stage_research.py` | none | Settled: initial mass-costate sensitivity for the rejected event solver | Archive before deletion |
| `compare_single_stage_references.py` | `single_stage_research.py` | none | Settled for the reference hierarchy; direct checks remain useful | Keep: validation/reference |
| `edge_two_arc_shooting.py` | `single_stage_edge_cases.py`; `single_stage_research.py`; `two_arc_shooting.py`; `two_burn_reference.py` | none | Open: comparison must cover coast/burn and burn-only | Keep: production reference/debugging |
| `multi_burn_reference.py` | `single_stage_research.py`; `two_burn_reference.py` | none | Settled only for the tested first-outbound domain; remains a falsifier | Keep: validation/reference |
| `time_bound_sweep.py` | `initial_guess_comparison.py`; `single_stage_research.py`; `two_arc_shooting.py`; `two_burn_reference.py` | none | Open: explicit branch durations remain sensitive to the time domain | Keep: production reference/debugging |
| `staged_two_arc_research.py` | none | `multi_stage_unit_checks.py`; `staging_unit_checks.py`; `multi_stage_primer.py` | Partly settled: normalization and fixed jettison are settled; schedule is not | Keep: validation/reference |
| `staging_unit_checks.py` | `staged_two_arc_research.py` | none | Settled for the historical staging fixture | Keep: validation/reference |
| `multi_stage_primer.py` | `staged_two_arc_research.py` | `multi_stage_unit_checks.py` | Open: staged schedules and production boundaries need redesign | Keep: production reference/debugging |
| `multi_stage_unit_checks.py` | `multi_stage_primer.py`; `staged_two_arc_research.py` | none | Settled for the current transition and unit mapping | Keep: validation/reference |
| `sigmoid_primer.py` | `initial_guess_comparison.py`; `single_stage_research.py`; `two_burn_reference.py` | Every program under `R/` | Settled: smooth implicit switching is not the production direction | Archive before deletion |
| `R/adaptive_eps_pipeline.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled: adaptive sigmoid-to-event continuation is rejected | Archive before deletion |
| `R/analytic_jacobian.py` | `sigmoid_primer.py`; `single_stage_research.py` | `R/jacobian_check.py` | Settled: derivative prototype is specific to the rejected sigmoid model | Archive before deletion |
| `R/deep_continuation.py` | `sigmoid_primer.py` | none | Settled: deep sigmoid continuation does not recover a reliable event structure | Archive before deletion |
| `R/free_eps_shooting.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled: free epsilon does not select the required arc structure | Archive before deletion |
| `R/free_eps_slsqp.py` | `R/production_solver.py`; `sigmoid_primer.py`; `single_stage_research.py` | none | Settled: optimizing epsilon does not solve the production problem | Archive before deletion |
| `R/h0_lambda_rho_fixed.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled: an initial Hamiltonian condition cannot replace the terminal switching residual here | Archive before deletion |
| `R/hybrid_polish.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled: sigmoid-to-implicit polishing is not the production architecture | Archive before deletion |
| `R/jacobian_check.py` | `sigmoid_primer.py`; `R/analytic_jacobian.py` | none | Settled: sigmoid Jacobian check has no explicit-arc target | Archive before deletion |
| `R/multistart_basin.py` | `sigmoid_primer.py` | none | Settled: the desired hard structure is not reliably accessible at finite epsilon | Archive before deletion |
| `R/production_solver.py` | `sigmoid_primer.py`; `single_stage_research.py` | `R/free_eps_slsqp.py` | Settled: its smooth/event hybrid is not production despite its filename | Archive before deletion |
| `R/reverse_continuation.py` | `sigmoid_primer.py` | none | Settled: the sigmoid result is not only a downward-continuation artifact | Archive before deletion |
| `R/sensitivity_grid.py` | `sigmoid_primer.py` | none | Settled: local sigmoid seed perturbations do not recover the required structure | Archive before deletion |
| `R/sharp_oracle_profile.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled for the known hard oracle; per-arc diagnostics remain useful | Keep: validation/reference |
| `R/sharp_residual_check.py` | `sigmoid_primer.py`; `single_stage_research.py` | none | Settled as a falsifier for structurally invalid sigmoid roots | Keep: validation/reference |
| `R/solver_variants.py` | `sigmoid_primer.py` | none | Settled: solver selection for the rejected sigmoid formulation is closed | Archive before deletion |
| `R/two_arc_residual_variant.py` | `sigmoid_primer.py`; `two_arc_shooting.py`; `single_stage_research.py`; `two_burn_reference.py` | none | Settled for the value of terminal switching residuals; full three-mode solver remains open | Keep: production reference/debugging |
| `R/warm_start_lm.py` | `sigmoid_primer.py` | none | Settled: better warm starts do not remove sigmoid stiffness and structure ambiguity | Archive before deletion |

## Shared Model

Most programs use the same offline model. It is normalized, vacuum, two-body
motion in one orbital plane. The target state is

$$
(\rho_f,u_{r,f},u_{t,f})=(1,0,1).
$$

The main single-stage sequence is

$$
q=1\;\longrightarrow\;q=0\;\longrightarrow\;q=1.
$$

The first powered arc shapes a transfer orbit. The coast follows that arc. The
second powered arc completes circularization. Direct optimizers are offline
references and oracle generators. The old fixed-sequence primer solver is a
migration reference for the new explicit-arc production solver. It is not the
production mode-selection mechanism.

The main supporting documents are [Overview](overview.md), [Primer
Equations](primer-equations.md), [Burn Sequence Study](burn-sequence-study.md),
[Single-Stage Cases](single-stage-test-cases.md), [Initial Guess
Study](initial-guess-study.md), [Solver Comparison](solver-comparison.md),
[Final-Time Domain](time-domain-study.md), [Staging Study](staging-study.md),
[Sigmoid Primer Study](sigmoid-primer-study.md), and [Optimal Burn
Trajectory](optimal-burn-trajectory.md).

## Single-Stage Model And Checks

### `experiments/single_stage_research.py`

**Reason and purpose:** This is the independent research harness for the
single-stage model. It was created to validate the state equations and compare
candidate optimizer formulations before production code uses them.

**What it does:** It defines normalized polar, Cartesian, and physical-unit
equations. It defines synthetic and Kerbin fixtures, coast-to-apoapsis events,
direct one-burn optimization, explicit primer shooting, event-based implicit
shooting, and relaxed direct throttle optimization. It also checks polar
versus Cartesian propagation, costate derivatives, physical scaling, coast
energy, coast angular momentum, and randomized cases.

**Useful going forward:** This is the central offline model for many other
experiments. Its independent checks can become regression or property tests.
It must remain separate from kRPC and the flight controller.

### `experiments/body_scaling_sweep.py`

**Reason and purpose:** This program asks whether a change in physical body
scale changes the normalized trajectory. It tests the claim that the planner
depends on normalized parameters rather than one body's units.

**What it does:** It defines Kerbin, Mun, Minmus, and Duna fixtures. It
propagates one common normalized orbit and thrust direction in normalized and
physical polar equations. It converts the physical result back to normalized
units and reports the stage parameters and state error.

**Useful going forward:** It can become a unit or property regression for
normalization. It also supports keeping body data outside the dimensionless
planning equations.

### `experiments/single_stage_boundary_checks.py`

**Reason and purpose:** This program checks states that should not enter normal
primer shooting. It asks whether an exact target orbit can exit immediately
and whether a near-radial state remains numerically finite.

**What it does:** It coasts an exact circular target and checks terminal error
and mass change. It also coasts a near-radial state and checks that the state
remains finite. It can print results or write JSON diagnostics.

**Useful going forward:** The exact-circular case should support a zero-burn
early exit. The near-radial case should define a conditioning limit and an
explicit domain or error response.

### `experiments/single_stage_edge_cases.py`

**Reason and purpose:** This program asks whether the single-stage methods work
outside the moderate reference cases. It covers near-target, low-thrust,
very-high-thrust, early-outbound, and near-apoapsis conditions.

**What it does:** For each fixture, it runs direct one-burn mesh refinement,
explicit primer shooting, event-based implicit shooting, and an eight-angle
multistart two-burn reference. It reports fuel, residuals, switch times, and
switching-function signs.

**Useful going forward:** These fixtures provide robustness and generalization
coverage. They can become named regression tests after the simulator is stable.

### `experiments/analyze_optimal_burn.py`

**Reason and purpose:** This program investigates why the first optimal burn is
not generally aligned with the current velocity. It turns numerical output
into a physical explanation.

**What it does:** It builds the two supplied Kerbin cases, obtains the fixed
two-arc reference, and integrates the first powered primer arc with dense
DOP853 output. It samples primer and velocity angles. It also calculates
energy, angular momentum, apsides, thrust power, perpendicular acceleration,
and integrated angular-momentum effects.

**Useful going forward:** It supports the interpretation in [Optimal Burn
Trajectory](optimal-burn-trajectory.md). It is useful for diagnostic plots and
regression explanations, but it is not a planner.

## Single-Stage Solvers, References, And Initialization

### `experiments/two_burn_reference.py`

**Reason and purpose:** This program provides an independent direct reference
for two powered arcs. It was created to check primer solutions without using
costates.

**What it does:** It propagates an optional initial coast, two full-thrust
arcs with piecewise-constant angles, and a middle coast. SLSQP enforces the
terminal orbit and final-time cap while minimizing powered duration. It
supports direct, primer-derived, impulse-derived, and deterministic multistart
seeds.

**Useful going forward:** It is an offline oracle and falsifier. A mesh-refined
direct result with lower fuel can reject a primer result. It must not enter the
runtime path.

### `experiments/two_arc_shooting.py`

**Reason and purpose:** This program implements the provisional fixed-sequence
single-stage solver. It asks whether a known $q=1\rightarrow0\rightarrow1$
sequence can be solved reliably with one bounded shooting problem.

**What it does:** It shoots six variables: the initial primer angle, first
burn duration, coast duration, second burn duration, radial-position costate,
and mass costate. It integrates three sequential primer arcs. It enforces the
terminal orbit, two internal switching conditions, and either the free-time
Hamiltonian condition or the active final-time cap. It uses bounded
`least_squares`.

**Useful going forward:** It is the closest existing explicit-duration solver
and a migration reference for the new three-family production solver. It needs
separate coast/burn and burn-only branches, plus switching-zero residuals at
the required arc endpoints.

### `experiments/initial_guess_comparison.py`

**Reason and purpose:** This program asks which analytic initial values produce
reliable fixed-sequence solves. It replaced the need to run a direct optimizer
before every runtime solve.

**What it does:** It converts supplied inertial Kerbin vectors to local
normalized polar states. It estimates an impulse in the current velocity
direction that raises apoapsis to the target. It converts the impulse and
circularization estimates into finite-thrust durations, initializes costates,
and compares current-velocity, tangential, and predicted-apoapsis-prograde
angle seeds.

**Useful going forward:** It provides analytic timing and direction seeds for
the new explicit-duration branches. It also documents the near-zero
radial-velocity limitation of the initial costate formula.

### `experiments/runtime_benchmark.py`

**Reason and purpose:** This program measures the old analytic-seed and
fixed-sequence path. It provides a baseline before the three explicit arc
families replace that path.

**What it does:** It repeats the timing estimate, current-velocity angle seed,
and `two_arc_shooting.solve_from_initial` for two Kerbin cases. It reports
mean, minimum, and maximum wall time, acceptance, and residual.

**Useful going forward:** It provides a performance baseline for comparison
with the new burn/coast/burn, coast/burn, and burn-only solvers. It does not
establish numerical correctness.

### `experiments/compare_shooting_solvers.py`

**Reason and purpose:** This program compares local nonlinear solvers and
mass-costate seeds. It asks whether solver success status agrees with a valid
terminal solution.

**What it does:** It obtains informed implicit seeds, scales the initial
mass-costate by deterministic factors, and tests `root` with `hybr` and `lm`
plus bounded `least_squares`. It records status, residual norm, elapsed time,
fuel, switch times, and event count.

**Useful going forward:** It supports solver selection and acceptance rules.
The residual norm, not optimizer status alone, must determine whether a result
is usable.

### `experiments/single_stage_lambda_scan.py`

**Reason and purpose:** This program measures sensitivity to the initial
mass-costate. It asks whether one selected value of
$\lambda_{\eta,0}$ is sufficient for event-based shooting.

**What it does:** It creates direct and explicit one-burn seeds, then scales
only the initial mass-costate by deterministic factors from $0.25$ through
$4$. It runs the event-based implicit solver and records status, residual,
fuel, burn time, switch times, event count, and final state.

**Useful going forward:** It can falsify reliance on one mass-costate seed. It
also supports deterministic seed policies and explains why the event-based
solver remains a diagnostic rather than the runtime solver.

### `experiments/compare_single_stage_references.py`

**Reason and purpose:** This program compares the main single-stage
formulations. It asks whether direct and primer methods agree and whether a
restricted formulation hides a cheaper trajectory.

**What it does:** It runs direct one-burn mesh refinement at 4, 8, 16, and 32
angle intervals. It derives explicit and implicit primer results. It then uses
successful implicit results to seed relaxed throttle-angle meshes at 16, 32,
and 48 intervals. It records residuals, fuel, switch diagnostics, throttle
ranges, and primer arc signs.

**Useful going forward:** It provides offline falsification data and a common
comparison harness. It can detect a solver that reports a root but violates
the intended control sequence.

### `experiments/edge_two_arc_shooting.py`

**Reason and purpose:** This program asks whether fixed two-arc primer shooting
survives the five edge regimes in `single_stage_edge_cases.py`.

**What it does:** It refines a restricted direct one-burn solution, obtains a
multistart direct two-burn reference, and passes that result to
`two_arc_shooting.solve`. It compares direct and primer fuel and residuals. It
can write JSON output.

**Useful going forward:** It provides focused robustness coverage for low
thrust, high thrust, early outbound, near-target, and near-apoapsis states.
It can become an integration test for the fixed-sequence solver.

### `experiments/multi_burn_reference.py`

**Reason and purpose:** This program tests a key falsifier for the provisional
two-burn sequence. It asks whether three powered arcs can use less fuel than
the accepted two-burn reference.

**What it does:** It propagates an initial coast followed by three full-thrust
arcs. Each powered arc uses piecewise-constant thrust angles. SLSQP enforces
the terminal orbit and final-time cap while minimizing total powered duration.
It seeds the search from the two-burn result and tests two- and four-angle
meshes.

**Useful going forward:** A valid lower-fuel result would invalidate the
two-burn sequence for that domain. Continued mesh refinement and independent
multistart are needed before a negative result can be treated as strong
evidence.

### `experiments/time_bound_sweep.py`

**Reason and purpose:** This program studies the effect of the final-time cap.
It asks whether a cap-active result is a true interior optimum or only the
best result in a truncated time domain.

**What it does:** It tests a randomized case and two Kerbin cases. For cap
offsets $0$, $0.25$, $0.5$, and $1.0$, it rebuilds each case, reruns the
restricted direct seed and two-burn reference, and solves fixed-sequence
shooting. It records cap, fuel, final time, cap activity, and residual.

**Useful going forward:** It supports cap continuation and identifies hidden
lower-fuel later branches. Cap-dependent oracle values must be regenerated when
the cap coefficient changes.

## Staged Research

### `experiments/staged_two_arc_research.py`

**Reason and purpose:** This program establishes the historical fixed-schedule
two-stage baseline. It asks whether a staged vehicle can be represented with a
stage-specific mass jump and a fixed staging gap.

**What it does:** It models a full first-stage burn, fixed dry-mass jettison,
a two-second ballistic gap, and a variable full-thrust second-stage burn. It
compares a piecewise-angle direct reference with a four-variable fixed-schedule
primer solve. It reports stage budgets, terminal residuals, local Hamiltonian
drift, switching values, and junction continuity.

**Useful going forward:** It is a historical reference for the generalized
staged solver. It validates the fixed schedule only. It does not prove a
free-throttle staged sequence.

### `experiments/staging_unit_checks.py`

**Reason and purpose:** This program independently checks the historical staged
propagation in physical units. It asks whether normalized propagation, the
mass jump, and physical polar propagation agree.

**What it does:** It propagates stage one, applies the jettison mass jump,
propagates the staging gap, and propagates stage two in normalized and physical
polar equations. It checks converted trajectory error, stage mass budgets, gap
energy, and angular momentum. It raises an error above its tolerance.

**Useful going forward:** It protects the historical staging baseline against
unit and mass-map regressions. It also provides an independent comparison check
for the generalized staged implementation.

### `experiments/multi_stage_primer.py`

**Reason and purpose:** This program generalizes the staged model. It asks
whether scheduled on/off/on burns can be solved when a trajectory crosses
stage boundaries.

**What it does:** It defines normalized multi-stage cases, burn trackers,
jettison junctions, fixed staging gaps, direct piecewise-angle propagation,
and primer propagation. Its six-variable residual contains terminal orbit
conditions, augmented switching conditions at internal burn boundaries, and
either the terminal Hamiltonian condition or the active-cap condition. It
compares multistart direct SLSQP with bounded primer least squares.

**Useful going forward:** Its direct propagation and stage-boundary checks are
useful references for a new staged explicit-arc model. Its fixed schedule and
primer residuals are not a production API and need replacement before staged
deployment.

### `experiments/multi_stage_unit_checks.py`

**Reason and purpose:** This program checks the generalized staged solver
against an independent physical propagation. It asks whether stage-specific
parameters and jettison rules remain correct after generalization.

**What it does:** It uses two `multi_stage_primer.py` cases and fixed test
parameters. It independently integrates physical polar equations, tracks stage
exhaustion and jettison, converts the result to normalized units, checks for
exactly one transition, and fails when trajectory error exceeds $10^{-8}$.

**Useful going forward:** It provides regression coverage for stage transitions,
mass maps, physical-unit conversion, and active-stage parameters. It should run
with every staged solver change.

## Sigmoid Primer And Review

The root program and the programs in this folder study a smooth alternative to
hard switching. They replace the bang-bang control with

$$
q_\varepsilon=\frac{1}{1+\exp(-S/\varepsilon)}.
$$

The review found that continuation can solve terminal equations, but it can
lose the intended event sequence, produce fractional throttle, and become stiff
as $\varepsilon$ decreases. Explicit-duration references are more reliable for
known burn structures. See [Sigmoid Primer Study](sigmoid-primer-study.md).

### `experiments/sigmoid_primer.py`

**Reason and purpose:** This program tests smooth sigmoid switching as an
alternative to explicit event sequencing. It was created to explore initial
coast cases and to determine whether one smooth indirect solve can recover the
hard-switch trajectories.

**What it does:** It builds direct one-burn and two-arc references. It then
integrates an eight-component state-costate system with
$q_\varepsilon=\operatorname{expit}(S/\varepsilon)$. Its four shooting
variables are the initial primer angle, final time, radial-position costate,
and mass costate. The residual contains the three terminal orbit conditions
and $S(t_f)=0$. It continues through a fixed epsilon schedule and records fuel,
switch crossings, throttle range, effective burn, and Hamiltonian drift.

**Useful going forward:** It is a research tool for coast-first trajectories
and continuation behavior. The study found that it is not yet a replacement
for explicit fixed-sequence shooting because it can lose event structure and
become stiff at small epsilon.

### `experiments/sigmoid_primer_review/adaptive_eps_pipeline.py`

**Reason and purpose:** This program prototypes adaptive sigmoid continuation
followed by hard-switch polishing. It asks whether the smooth problem can
provide a useful seed for the event-based solver.

**What it does:** It starts at $\varepsilon=1$, divides epsilon by three, and
stops when sampled throttle has both near-off and near-on values or when it
reaches $3\times10^{-7}$. It then seeds the hard-switch solver and reports
residual, fuel, final time, switches, and thrust time for the Kerbin cases.

**Useful going forward:** It is a predecessor to `production_solver.py`. It
shows the pipeline idea, but its sharp residual and fallback behavior are older
than the later production-style prototype.

### `experiments/sigmoid_primer_review/analytic_jacobian.py`

**Reason and purpose:** This program tests whether exact sensitivities improve
sigmoid shooting. It asks whether finite-difference conditioning is the main
cause of continuation difficulty.

**What it does:** It forms a complex-step Jacobian for the sigmoid ODE. It
integrates an $8\times4$ variational sensitivity matrix and forms the residual
Jacobian, including final-time derivatives. It solves from oracle and smeared
seeds at several epsilon values and reports throttle profiles and crossings.

**Useful going forward:** It provides a basis for analytic sensitivities or
multiple shooting. The production-style pipeline does not currently use this
prototype.

### `experiments/sigmoid_primer_review/deep_continuation.py`

**Reason and purpose:** This program tests continuation below the study's
$\varepsilon=10^{-4}$ floor. It asks whether a smooth root can be sharpened
far enough to recover the hard event structure.

**What it does:** For `kerbin-example`, it continues from smeared and
fixed-sequence oracle parameters through $3\times10^{-5}$ to $3\times10^{-7}$.
It varies finite-difference Jacobian options, integrates dense sigmoid
trajectories, and reports residual, fuel, throttle minimum, and switch
crossings.

**Useful going forward:** It supports handing off to a sharp solver instead of
forcing finite-difference continuation to very small epsilon.

### `experiments/sigmoid_primer_review/free_eps_shooting.py`

**Reason and purpose:** This program treats epsilon as a shooting variable. It
asks whether the solver can select a useful smoothing scale without an outer
continuation schedule.

**What it does:** It solves five variables against four sigmoid residuals with
bounded `least_squares`, starting from an analytic guess and $\varepsilon=1$.
It profiles throttle and crossings, then propagates the same base parameters
through the hard event-based solver.

**Useful going forward:** It demonstrates that an underdetermined residual
manifold does not select the desired epsilon or event structure. It is a useful
warning against unconstrained free-epsilon shooting.

### `experiments/sigmoid_primer_review/free_eps_slsqp.py`

**Reason and purpose:** This program tests free-epsilon shooting with an
explicit fuel objective. It asks whether an objective can select among the
many endpoint roots left by free-epsilon least squares.

**What it does:** It uses SLSQP over the four shooting variables and epsilon.
It maximizes final mass subject to the four sigmoid residuals. It uses robust
integration and selectable tolerances from `production_solver.py`, records
epsilon history, profiles the result, and compares it with hard-switch
propagation.

**Useful going forward:** It is a more principled exploratory formulation than
`free_eps_shooting.py`. It remains a research variant because the objective and
relaxed control still do not guarantee the desired hard event sequence.

### `experiments/sigmoid_primer_review/h0_lambda_rho_fixed.py`

**Reason and purpose:** This program tests whether the initial Hamiltonian can
eliminate one shooting variable. It asks whether $\lambda_{\rho,0}$ computed
from $H(0)=0$ can replace the terminal switching residual.

**What it does:** It computes $\lambda_{\rho,0}$ from the initial smoothed
Hamiltonian, leaving three unknowns and the three terminal orbit residuals. It
continues over the standard epsilon schedule and records final time, fuel,
throttle range, switching values, crossings, and Hamiltonian drift.

**Useful going forward:** It is a negative-control experiment. It demonstrates
why finite-epsilon shooting needs a terminal $S(t_f)=0$ condition to constrain
free final time.

### `experiments/sigmoid_primer_review/hybrid_polish.py`

**Reason and purpose:** This program checks whether hard event-based shooting
can recover the correct structure from sigmoid outputs.

**What it does:** It seeds `implicit_solve_from_initial` with the fixed oracle,
the $\varepsilon=10^{-4}$ smeared root, and a rough $10^{-5}$ seed. It reports
sharp residual, fuel, switch times, thrust time, and solver status.

**Useful going forward:** It supplies evidence for the hybrid
continuation-plus-polish design used by `production_solver.py`.

### `experiments/sigmoid_primer_review/jacobian_check.py`

**Reason and purpose:** This program validates the analytic Jacobian prototype.
It asks whether the sensitivity implementation agrees with an independent
finite-difference calculation.

**What it does:** At oracle and smeared parameters, and at $\varepsilon=10^{-4}$
and $10^{-5}$, it compares `analytic_jacobian.residual_with_jac` with a
central finite-difference Jacobian using step $10^{-6}$. It prints relative
errors and both matrices.

**Useful going forward:** It is a regression check if analytic sensitivities
are developed further. It can identify derivative errors before they affect a
continuation solver.

### `experiments/sigmoid_primer_review/multistart_basin.py`

**Reason and purpose:** This program studies the sigmoid basin at
$\varepsilon=10^{-4}$. It asks whether a distinct two-burn root is accessible
near the known hard-switch oracle.

**What it does:** It generates 48 deterministic random seeds in a box around
the oracle and solves each with bounded `least_squares`. It classifies profiles
by crossings, final time, and throttle minimum. It also sweeps finite-difference
steps at several epsilon values.

**Useful going forward:** It provides basin and conditioning evidence. It
supports the conclusion that the smeared root is the accessible root at the
tested epsilon and that better local solver settings alone may not recover the
desired event structure.

### `experiments/sigmoid_primer_review/production_solver.py`

**Reason and purpose:** This program assembles the historical proposed sigmoid
continuation-plus-hard-switch-polish pipeline. It asks whether a robust hybrid
procedure can support future online use.

**What it does:** It provides DOP853 and LSODA integration, adaptive epsilon
steps, bounded `trf` first solves, cheaper `lm` warm starts with `trf` fallback,
residual non-worsening checks, throttle-render criteria, and hard-switch
polishing with $S(t_f)=0$. It supports online and high-accuracy modes,
repeated polish passes, a fuel guard, per-case selection, timing, and JSON
output.

**Useful going forward:** Its implementation records the historical sigmoid
continuation and event-polish failure modes. Do not use it as a production
candidate or authoritative oracle. Archive it after the new explicit-arc
solver has replacement diagnostics.

### `experiments/sigmoid_primer_review/reverse_continuation.py`

**Reason and purpose:** This program tests whether a sigmoid root depends on
the direction of continuation. It asks whether the smeared root is caused only
by starting at large epsilon and decreasing it.

**What it does:** For `kerbin-example`, it starts from the fixed hard-switch
oracle at $\varepsilon=10^{-6}$ and increases epsilon through $10^{-4}$. Each
step uses bounded least squares and reports residual, final time, fuel,
throttle minimum, and crossings.

**Useful going forward:** It provides path-dependence evidence. It can show
whether a continuation schedule selects a particular root basin rather than a
unique physical solution.

### `experiments/sigmoid_primer_review/sensitivity_grid.py`

**Reason and purpose:** This program maps local sensitivity around the
fixed-sequence oracle. It asks whether small changes in the initial mass
costate, final time, or other shooting variables reach a distinct sigmoid
solution.

**What it does:** It evaluates the sigmoid residual at several epsilon values.
It perturbs $\lambda_{\eta,0}$ and final time on a grid, and perturbs each
shooting variable independently. It solves each seed at $\varepsilon=10^{-4}$
and classifies it by crossings, final time, throttle minimum, fuel, and
residual.

**Useful going forward:** It provides local basin and conditioning evidence. It
can guide better seed selection or show when a different formulation is
needed.

### `experiments/sigmoid_primer_review/sharp_oracle_profile.py`

**Reason and purpose:** This program profiles the exact hard-switch oracle. It
asks whether the reference trajectory contains narrow or nearly singular
switching behavior that a sigmoid cannot resolve.

**What it does:** It propagates the known `kerbin-example` oracle through
`implicit_propagate`. It reports switch times, thrust time, fuel, terminal
residual, initial and final switching values, and switching-function ranges on
each arc.

**Useful going forward:** It is a fixed diagnostic baseline. It explains why a
finite epsilon can fail to render the second burn even when terminal residuals
are small.

### `experiments/sigmoid_primer_review/sharp_residual_check.py`

**Reason and purpose:** This program asks whether a sigmoid endpoint also solves
the sharp hard-switch equations.

**What it does:** It propagates the hard oracle and stored smeared parameters
through `implicit_propagate`. It reports hard switch times, terminal residual,
fuel, $S(0)$, $S(t_f)$, and switching ranges on every arc.

**Useful going forward:** It is a direct falsifier for treating a sigmoid
endpoint as a valid hard-switch solution. It should remain in any review suite
for a future smoothed solver.

### `experiments/sigmoid_primer_review/solver_variants.py`

**Reason and purpose:** This program compares nonlinear solver choices for the
sigmoid system. It asks whether a different optimizer or derivative option can
recover the intended event sequence at $\varepsilon=10^{-4}$.

**What it does:** It tests bounded `trf`, unbounded `lm`, finite-difference
step variants, three-point derivatives, `root` with `hybr`, `lm`, and `df-sane`,
and SLSQP maximizing final mass under terminal constraints. It profiles every
result for residual, fuel, throttle range, and crossings.

**Useful going forward:** It supports solver choices and evaluation-count
claims in `production_solver.py`. It also records which solver successes are
not physically useful.

### `experiments/sigmoid_primer_review/two_arc_residual_variant.py`

**Reason and purpose:** This program tests an alternative transversality
residual in the fixed two-arc solver. It asks whether $S(t_f)=0$ is better
conditioned than $H(0)=0$ for an interior final time.

**What it does:** It reuses the direct two-burn seed and compares the original
`two_arc_shooting.residual` with a local residual that uses terminal switching
value for free final time. It evaluates three Kerbin cases and records fuel,
final time, parameters, residual, evaluation count, and reference values. It
keeps the cap residual for cap-active cases.

**Useful going forward:** It provides the strongest existing evidence for using
terminal switching residuals. Reuse that result when implementing explicit
burn/coast/burn, coast/burn, and burn-only branches; do not reuse its sigmoid
plumbing.

### `experiments/sigmoid_primer_review/warm_start_lm.py`

**Reason and purpose:** This program tests whether tighter warm starts make
deep epsilon continuation practical. It asks whether `lm` can cheaply follow
the continuation path after a well-converged preceding solve.

**What it does:** For `kerbin-example`, it compares loose and tight convergence
at $\varepsilon=3\times10^{-4}$. It then starts from the oracle and runs `lm`
through $10^{-4}$ to $10^{-6}$. It reports evaluations, residuals, and final
times.

**Useful going forward:** It supports using `lm` for cheap continuation steps.
It also identifies the limits of improving only the preceding warm start when
the underlying sigmoid problem becomes stiff.

## Catalog Status

The experiments form a research record, not one supported application API.
Several programs duplicate model definitions so that a result can be checked
independently. This duplication is useful for falsification, but it increases
maintenance cost.

The next useful maintenance step is to promote stable checks into tests while
keeping direct optimizers, multistart searches, solver comparisons, and
diagnostic profiles offline. The first production-support extraction should
extend the direct oracle to all three explicit arc families and then replace
the old fixed-sequence residual with arc-end switching residuals. The runtime
path should use only a validated planner and should not depend on kRPC-specific
flight code from these experiments.

This update documents retention and deletion candidates only. It does not
delete any experiment files.
