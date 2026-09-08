# Plan: Offline Oracle Finder

## Status

Implementation approved. The plan records the current implementation contract.

## Goal

Create one standalone offline tool for high-accuracy single-stage trajectory
references. The tool will accept a 3D state and a segment list. It will return
the best valid trajectory found by the supported single-stage solver families.

The tool will use the first segment only. It will use a configurable effective
powered-time limit, with 150 seconds as the default. It will not model staging.

Define the named constant `MIN_PHASE_DURATION_SECONDS = 0.02`. For each
problem, derive `min_phase_duration_tau` as
$\texttt{MIN\_PHASE\_DURATION\_SECONDS}/t_*$. Do not emit or accept a phase
shorter than this value.

Every accepted plan must end at the end of a burn. Never report a terminal
coast. Once the target circular orbit is reached, any later coast has no fuel
cost and does not change the terminal orbit conditions, so trim it.

The tool will target an exact circular orbit. The default target is 80 km above
the Kerbin radius. The target radius will be configurable.

If the input state already satisfies the target state within tolerance, return
an empty plan immediately. Report zero fuel, zero delta-v, unchanged mass, and
an empty phase sequence.

## Repository Findings

- `experiments/two_burn_reference.py` is a useful algorithm reference, but it
  imports `single_stage_research.py` and its experimental data structures.
- `experiments/sigmoid_primer_review/production_solver.py` has an offline mode,
  but it is still coupled to experimental modules.
- The sigmoid method can converge to a fractional-throttle root. The
  `kerbin-example` case is a known example.
- `docs/experiments.md` identifies three explicit arc families as the intended
  production direction: burn/coast/burn, coast/burn, and burn-only.
- `docs/single-stage-test-cases.md` records the three Kerbin states and their
  normalized fixed-sequence reference results.
- `experiments/initial_guess_comparison.py` contains the useful impulse timing
  estimate and three deterministic primer-direction seeds.
- `experiments/single_stage_lambda_scan.py` scans an old mass-costate parameter.
  That scan is not directly useful for the implicit-mass explicit solvers.
- `sim.py` contains the reusable physical `RocketSegment` data shape, but the
  new tool should not depend on any `experiments/` or `examples/` module.

## Proposed Files

### `offline/oracle_finder.py`

Add a kRPC-free offline tool under `offline/`. Keep the implementation
independent from `experiments/`. Use small local data classes and functions for
the solver.

Accept any segment object with `thrust`, `ve`, and `initial_mass` attributes.
This structural input keeps the tool usable with `sim.RocketSegment` without a
runtime import from the simulator.

Expose a Python function with this shape:

```python
find_oracle(
    position,
    velocity,
    segments,
    *,
    mu=3.5316e12,
    target_radius=680_000.0,
    max_burn_time=150.0,
)
```

The command-line entry point will run the three Kerbin examples and support a
JSON output path. It will also support selecting one example.

### `offline/test_oracle_finder.py`

Add focused tests for input conversion, dynamics, output shape, acceptance
checks, and the three example selections. Keep the full offline solve out of
the normal unit-test path if it risks making validation exceed its time budget.

### `docs/offline-oracle-finder.md`

Document the public input, normalization, output schema, solver modes, known
limitations, and example command. Record the final runtime measurements after
implementation.

## Physical Input And Orbital Plane

Use SI units for public inputs.

For position $r$ and velocity $v$:

1. Set $e_r=r/|r|$.
2. Remove the radial velocity component from $v$.
3. Set $e_t$ to the normalized remaining component.
4. Set $e_n=e_r x e_t$.

The positive tangential direction will follow the supplied angular momentum.
This gives the initial 2D state $(r,0)$ and $(v_r,v_t)$ with $v_t>0$.

Reject a state with a zero or near-zero tangential component. Such a state does
not define a stable orbital plane direction for this solver.

Store all three basis vectors in the result. Use $\theta=0$ at the initial
position. The result will include the mapping from 2D plane coordinates back to
the original 3D frame.

## Normalization And Dynamics

For target radius $r_*$ and gravitational parameter $\mu$, define

$$
v_* = \sqrt{\frac{\mu}{r_*}},
\qquad
t_* = \frac{r_*}{v_*}.
$$

Use the normalized state

$$
(\rho,u_r,u_t,\eta)
=\left(\frac{r}{r_*},\frac{v_r}{v_*},\frac{v_t}{v_*},\frac{m}{m_0}\right).
$$

For the first segment, define

$$
\gamma=\frac{T r_*}{m_0v_*^2},
\qquad
\kappa=\frac{v_e}{v_*},
\qquad
\tau_{\max}=\frac{t_{\max}}{t_*},
\qquad t_{\max}=\texttt{max\_burn\_time}.
$$

Use the validated polar equations from `docs/primer-equations.md`. Do not add
tangential position to the optimization state or its costates.

The output post-processing will integrate

$$
\theta' = \frac{u_t}{\rho}
$$

from the solver samples. It will then provide 2D Cartesian position and
velocity arrays, while retaining the polar state used by the solver.

## Time Domain

Use the existing first-outbound search domain. Compute the next initial
apoapsis and the initial osculating orbital period. Set

$$
\tau_{\mathrm{limit}}
=\tau_{\mathrm{apo},0}+\frac{3}{4}\tau_{\mathrm{period},0}.
$$

This bound prevents equivalent later-revolution coast solutions. It also keeps
the three Kerbin examples comparable with the existing reference data.

The configured powered-time value limits total powered time, not elapsed coast
time. Reject any candidate with non-positive mass or powered time above this
limit.

If a candidate reaches the powered-time limit, record a warning and emit a
runtime warning. The result must identify the binding limit even when the
candidate remains otherwise valid. This warns that the returned trajectory is
constrained by the configured budget.

This time-domain choice is an assumption that needs user confirmation. If the
desired oracle must allow unrestricted coast time, remove this bound in a later
change and define how equivalent revolutions are ranked.

## Solver Families

Run all four requested candidates. Use deterministic seeds and bounded
`scipy.optimize.least_squares` solves. Do not trust an optimizer success flag
without checking the physical residuals and switching signs.

The three explicit-duration candidates will use thrust acceleration and thrust
direction as their controls. A burn has

$$
\mathbf a_T=\frac{T}{m}\hat u_T,
\qquad
\hat u_T=\cos(\alpha)e_r+\sin(\alpha)e_t.
$$

A coast has zero thrust acceleration. The solver will report acceleration and
direction, not a generic throttle value, for these candidates.

### 1. Pure Lawden With Sigmoid Continuation

Use the eight-component state/costate system:

$$
(\rho,u_r,u_t,\eta,\lambda_\rho,p_r,p_t,\lambda_\eta).
$$

Use the Lawden switching function

$$
\Phi=\frac{\sqrt{p_r^2+p_t^2}}{\eta}
      +\frac{\lambda_\eta}{\kappa}.
$$

Replace the bang-bang throttle with

$$
q_\varepsilon=\operatorname{expit}(\Phi/\varepsilon).
$$

Continue from a large epsilon toward a small epsilon. Use adaptive stopping
when the sampled throttle contains both near-off and near-on values. Use tight
offline integration tolerances, but cap evaluation counts and record timing.

After continuation, extract switch crossings and run a hard-switch polish. The
candidate is valid only when the hard trajectory reaches the target, has the
expected sign of $\Phi$ on each arc, and has no unsupported extra arc.

Canonicalize the hard result before validation. If event detection creates a
terminal coast, remove that coast and set the reported terminal time and final
state to the end of the last burn. Compute terminal residuals and objectives at
that endpoint. Do not count a zero-duration terminal event as an additional
phase. Reject or re-solve any short internal burn or coast that is below
$\tau_{\min}$.

Classify the canonical Lawden phase sequence. The only supported sequences are
`burn/coast/burn`, `coast/burn`, `burn`, and the empty sequence when the input
state already satisfies the target orbit. Mark every other sequence as
unsupported. Add its phase sequence, switch times, and reason to the result
diagnostics, emit a runtime warning, and report it in the CLI output.

Record a failed or fractional Lawden result as a diagnostic. Never select it
over a valid explicit-arc result only because its smooth residual is smaller.

### 2. Burn/Coast/Burn

Use explicit durations $(\tau_{b1},\tau_c,\tau_{b2})$. The first burn starts
at the input state.

Use the Prussing implicit-mass convention for this family. During powered time,

$$
\eta=1-\frac{\gamma}{\kappa}\tau_{\mathrm{burn,elapsed}}.
$$

During coast, hold $\eta$ constant. Do not add a mass costate.

The initial costate parameters are $(p_{r,0},p_{t,0},\lambda_{\rho,0})$.
The switching function is $1-P$ with $P=|\mathbf p|$. Use one zero-switch
residual at the end of every segment, including the final burn.

Use the three terminal orbit residuals plus the three end-of-segment switching
residuals. Do not use a final Hamiltonian residual.

### 3. Coast/Burn

Set the first burn duration to zero. Solve for the initial coast duration and
the final burn duration. Use the same implicit-mass equations and costate
parameters as the burn/coast/burn family.

Require the switching function to be zero at the end of the coast and at the
end of the burn. Check that it is on the correct side throughout both arcs.

This separate family avoids forcing a zero first burn through a residual that
assumes an interior first-burn switch.

If a burn/coast/burn solve drives its first burn below
`min_phase_duration_tau`, reject
that candidate as a degenerate phase and use the coast/burn family instead.

### 4. Pure Burn

Start thrust at the input state and use one powered arc. Solve for its duration
and the three initial costate values.

Use the terminal switching condition. Check that the switching function
supports thrust for the complete arc.

### Explicit-Duration Transversality Check

The explicit-duration residuals use a switching zero at the final burn instead
of a final Hamiltonian residual. For the Prussing convention, write the last
arc Hamiltonian as

$$
H=H_0+\Gamma(1-P),
\qquad
\Gamma=\frac{\gamma}{\eta}>0.
$$

At the required final state $(\rho,u_r,u_t)=(1,0,1)$, the non-thrust part
$H_0$ is zero. Therefore,

$$
H(t_f)=\Gamma(t_f)(1-P(t_f)).
$$

The final switching residual $1-P(t_f)=0$ is equivalent to $H(t_f)=0$ when
the terminal circular-orbit conditions hold. Verify this algebraically in the
implementation and numerically with accepted trajectories.

The time-limit condition remains a bound on elapsed time. It is not added as a
residual when the final switch residual is used.

## Initial Values

Build a deterministic impulse-based seed before running any nonlinear solve.
The seed will use physical units first, then convert all times and states to
normalized units.

### First Burn Estimate

Compute the initial osculating orbit and its apoapsis. If the apoapsis is below
the target radius, use an instantaneous prograde burn in the direction of the
initial velocity.

Find the post-impulse speed from the vis-viva energy equation and the
apoapsis angular-momentum relation. Do not use a numerical root finder. Let

$$
k=\frac{r v_t}{|v|}
$$

be the angular-momentum coefficient for an impulse in the current velocity
direction. For target apoapsis $r_a^*$, compute

$$
v_1^+=
\sqrt{\frac{2\mu(1/r-1/r_a^*)}
{1-k^2/(r_a^*)^2}}.
$$

Then set $\Delta v_1=v_1^+-|v|$. This is the analytic prograde impulse that
sets the post-impulse apoapsis to the target, when the state is below target.
The post-impulse velocity vector is

$$
v_1^+=v_0+\Delta v_1\frac{v_0}{|v_0|}.
$$

Convert the impulse to the actual first-segment burn time with

$$
t_b(\Delta v,m)=\frac{m v_e}{T}
\left(1-\exp\left(-\frac{\Delta v}{v_e}\right)\right).
$$

Use the initial mass for the first burn. Update the mass with the rocket
equation before estimating the second burn.

If the initial apoapsis is already at or above the target, skip the first burn.
For this branch, use the current apoapsis as the reference point for a
tangential circularization impulse. This follows the existing above-target seed
branch. The explicit solver will still enforce the exact target state.

### Second Burn And Coast Estimate

Propagate the post-first-impulse orbit to its next apoapsis. At that apoapsis,
compute the impulse delta-v needed for the circularization estimate. For an
apoapsis below or at the target, use

$$
\Delta v_2=v_{\mathrm{circ}}(r_a)-v_{t,a},
\qquad
v_{\mathrm{circ}}(r)=\sqrt{\frac{\mu}{r}}.
$$

Use the remaining mass in the burn-time equation to calculate $t_{b2}$. Set
the coast duration between the two finite burns to

$$
t_c=\max\left(0,
t_{\mathrm{apo},1}-t_{b1}-\frac{t_{b2}}{2}\right).
$$

This centers the estimated second finite burn on the post-first-impulse
apoapsis. Clamp the powered-time sum to the configured effective limit and
reject the seed if the required mass would become non-positive.

Use these timings for the initial vectors of the burn/coast/burn and
coast/burn candidates. For burn-only, use the sum of the estimated burn times
with zero coast. If the first burn is skipped, use the second-burn estimate as
the burn-only duration.

### Explicit Primer Values

For every explicit-duration seed, set the initial primer direction to the
initial velocity direction:

$$
\alpha_0=\operatorname{atan2}(u_{t,0},u_{r,0}).
$$

Set the initial primer magnitude to one. This makes the Prussing switching
function $1-P$ equal to zero at the initial epoch.

Estimate $\lambda_{\rho,0}$ from the initial Hamiltonian equality as a seed
only. Let

$$
g_r=\frac{u_t^2}{\rho}-\frac{1}{\rho^2}.
$$

When $|u_r|$ is above the configured threshold, use

$$
\lambda_{\rho,0}
=\frac{p_{r,0}g_r-p_{t,0}u_ru_t/\rho}{u_r}.
$$

Do not use this equality as a solver residual. Near apoapsis, retain
$\lambda_{\rho,0}$ as a free value and use the deterministic fallback seeds
described below.

### Additional Deterministic Seeds

Retain the useful direction sweep from
`experiments/initial_guess_comparison.py`:

- initial velocity direction;
- local tangential direction;
- prograde direction at the predicted apoapsis.

For each direction, set $P_0=1$ and recompute the Hamiltonian-based
$\lambda_{\rho,0}$ seed. If the base solve fails, retry with a small fixed set
of bounded $\lambda_{\rho,0}$ scale factors around the base value. Use the
same timing seed for each retry.

If the Lawden continuation produces a valid hard trajectory, also use its
extracted arc times and initial direction as an additional explicit seed.

Do not copy the old mass-costate factor sweep. The explicit-duration model has
no mass costate. Report every seed and its accepted or rejected status.

## Costate And Mass Conventions

The Lawden candidate will explicitly integrate mass and $\lambda_\eta$.
Its fuel objective is the final mass.

The three explicit-duration candidates will calculate mass from cumulative
powered time. Their objective is the integrated thrust acceleration, equivalent
to

$$
\Delta v=v_e\log\left(\frac{m_0}{m_f}\right)
$$

for this one-segment model. They will not include a mass state or mass costate
in their shooting vectors.

Report all three equivalent quantities:

- final mass in kg;
- fuel consumed in kg;
- delta-v consumed in m/s.

Use final mass as the primary selection value. Use fuel and delta-v as
cross-checks and tie-breakers within numerical tolerance.

## Candidate Acceptance And Selection

Accept a candidate only when all conditions pass:

- The final normalized residual is below the configured tolerance.
- The target state is $(\rho,u_r,u_t)=(1,0,1)$.
- Mass remains positive at every integration sample.
- Total powered time does not exceed 150 seconds.
- Elapsed final time is inside the first-outbound limit.
- Every requested switch has a zero residual.
- The switching function has the correct sign on each declared arc.
- The trajectory contains no extra switch or unsupported phase.
- The final phase is a burn.
- Every reported phase duration is at least
  `MIN_PHASE_DURATION_SECONDS`.
- A Lawden sequence is one of the four supported sequences.

Choose the accepted candidate with the highest final mass. Report every
candidate's status, residual, timing, and objective so the choice is auditable.

## Output Schema

Return a typed result object and provide a JSON representation with these
sections:

- `input`: target, normalization scales, first-segment parameters, and initial
  3D vectors;
- `plane`: `r_hat`, `t_hat`, `n_hat`, and the 2D-to-3D mapping convention;
- `selected`: solver family, initial costate, final metrics, arc boundaries,
  terminal residual, and full trajectory phases;
- `candidates`: one summary for every solver family, including rejected results;
- `diagnostics`: epsilon history, optimizer messages, switching ranges, and
  elapsed wall time.

Each selected phase will use a `solve_ivp`-style representation. Store a global
time array and parallel arrays for each phase. Include:

- normalized time and physical time;
- polar radius, radial velocity, tangential velocity, and mass;
- derived 2D Cartesian position and velocity;
- costate values;
- thrust acceleration and radial/tangential thrust direction for explicit
  candidates;
- throttle, thrust acceleration, direction, and switching function for the
  Lawden candidate.

Use separate phase records for burn and coast. Preserve the endpoint samples
from each integration call. Trim any terminal coast before serialization, so
the last phase is always a burn unless the sequence is empty. State clearly
whether costates use the Lawden or Prussing convention.

## Three Kerbin Examples

Add the three supplied fixtures without importing them from `experiments/`:

- `kerbin-example`;
- `kerbin-first-example`;
- `kerbin-coast-first-example`.

Use the vectors and masses recorded in `docs/single-stage-test-cases.md`. Use
Kerbin $\mu=3.5316\times10^{12}\ \mathrm{m^3/s^2}$, body radius
$600000\ \mathrm{m}$, and target radius $680000\ \mathrm{m}$.

The explicit-duration comparison should find burn/coast/burn for the first two
examples and coast/burn for the coast-first example. The Lawden candidate may
select any supported lower-fuel sequence after hard polishing. Compare
normalized fuel and timing with the documented reference rows, allowing solver
tolerance and integration differences.

## Validation

Run these checks after implementation:

1. Test the 3D-to-plane conversion against all three Kerbin fixtures.
2. Test polar dynamics against an independent 2D Cartesian propagation.
3. Test that the configurable effective burn limit defaults to 150 seconds and
   overrides the input segment's declared `max_burn_time`.
4. Test mass, fuel, and delta-v consistency.
5. Verify that the final switching residual equals the final Hamiltonian
   residual when the circular terminal conditions hold.
6. Verify explicit candidates report thrust acceleration and direction controls.
7. Run all three examples and verify family selection and terminal residuals.
8. Confirm that a fractional or wrong-structure sigmoid result is rejected.
9. Run `./validate.sh`.
10. Verify that a limit-binding candidate emits a warning and reports the
    binding limit.
11. Verify that no accepted result ends with a coast or contains a phase shorter
    than `MIN_PHASE_DURATION_SECONDS`.
12. Verify that an unsupported Lawden sequence is rejected and reported to the
    caller.
13. Measure the full three-example command. Keep the run below ten minutes.

The normal test suite will use cheap unit checks. The full offline solve will be
an explicit validation command because it can take minutes.

## Options Considered

| Option | Benefits | Costs | Decision |
| --- | --- | --- | --- |
| Import `single_stage_research.py` | Reuses validated equations and fixtures | Couples the tool to experimental code and its large dependency surface | Reject |
| Wrap `production_solver.py` | Already has sigmoid continuation and offline settings | Known fractional-throttle branch, incomplete requested output, and experimental imports | Reject |
| Copy the entire experiment stack | Maximum reuse of existing research | Keeps the baggage this task is intended to remove | Reject |
| Build a small tool in `offline/` | Standalone API, clear output, no `examples/` dependency | Repeats a small amount of equation code | Choose |
| Use only explicit arc families | Reliable mode selection and simple outputs | Does not satisfy the requested pure Lawden comparison | Reject |
| Use only pure Lawden continuation | One general solver | It can lose event structure and become stiff | Reject |
| Run both and select validated candidates | Satisfies the requested comparison and exposes disagreements | More implementation and runtime | Choose |
| Add a direct angle-mesh optimizer | Independent falsification reference | More code and another output format; not required for the first oracle tool | Defer |

## Implementation Note

The user approved implementation in `offline/`, the first-outbound time limit,
the switching residual choice, and the default Kerbin constants. The burn-time
limit is configurable, with 150 seconds as the default.
