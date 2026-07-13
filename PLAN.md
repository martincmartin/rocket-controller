# Plan: Reformulate `find_linear_tangent_params` as a Constrained SLSQP Problem

**Status: IMPLEMENTED.**

## Final Design

`find_linear_tangent_params(self, r3d, v3d, time_to_apoapsis) -> OptimizeResult`

Free parameters: `params = [coast_time, a_coeff, b_coeff, burn_time]`.

- **Objective**: `objective(params) -> params[3]` (burn_time) — find the earliest burn cutoff satisfying the constraints.
- **Memoized simulation helper**: a nested `simulate(params_tuple)` function decorated with `functools.lru_cache(maxsize=None)`, defined fresh on each call to `find_linear_tangent_params` (so the cache doesn't leak across calls). It coasts to `coast_time` then calls `propagate_linear_tangent` for `burn_time`. Both constraint functions call `simulate(tuple(params))`, so identical points are only simulated once.
- **Equality constraint** (2 components): `result.v - target_velocity(result.r, result.v)`, enforced to equal 0 via `NonlinearConstraint(eq_constraint, 0.0, 0.0)`.
- **Inequality constraint** (1 component): `norm(result.r) - target_radius`, enforced `>= 0` via `NonlinearConstraint(ineq_constraint, 0.0, np.inf)`.
- **Optimizer**: `scipy.optimize.minimize(..., method="SLSQP", constraints=[eq_nlc, ineq_nlc], bounds=..., options={"ftol": 1e-6, "maxiter": 200})`.
- `x0`/bounds reuse the old `Regime4D` values: `coast_time ∈ [0, time_to_apoapsis]`, `a ∈ [-5, 5]`, `b ∈ [-1, 1]`, `burn_time ∈ [0, total_burn_budget()]`.
- Naming: our own function parameters/variables use `params`/`initial_params` (not `x`/`x0`) to avoid clashing with the astrodynamics meaning of "x"; `x0=initial_params` is still passed to scipy's `minimize` since that keyword name is fixed by scipy's API.
- Print block (kept, extended): coast time, coefficients, burn time, RMS orbital error (from `BurnResult.error`, kept for continuity/diagnostics), apoapsis/periapsis, **velocity residual**, **radius residual**, optimizer success/message, and the `TimingContext` summary.

Actual run against the real flight data (from `main()`) converges to essentially exact circularization (apoapsis == periapsis == target altitude, residuals ≈ 0) in ~0.08s wall clock.

## Removed (Cascade Cleanup)

- `Regime` (ABC), `Regime3D`, `Regime4D` classes — replaced by the fixed 4-parameter SLSQP formulation above.
- `find_burn_params` (old top-level entry point) — no longer needed.
- `circularization_burn`, `find_burn_time` — only callers were `find_burn_params`; now dead code, removed.
- `solve_prograde`, `prograde_dynamics` — only caller was `circularization_burn`; now dead code, removed.
- Unused imports: `ABC`, `abstractmethod` (from `abc`), `minimize_scalar` (from `scipy.optimize`).
- Added imports: `functools` (for `lru_cache`), `NonlinearConstraint` (from `scipy.optimize`).

`main()` now makes a single call: `sim.find_linear_tangent_params(R3D, V3D, TIME_TO_APOAPSIS)`.

## Tests Added (`test_sim.py`)

1. **`target_velocity` unit tests** (previously untested):
   - `test_target_velocity_magnitude_and_perpendicularity`
   - `test_target_velocity_ccw_direction`
   - `test_target_velocity_cw_direction`
   - `test_target_velocity_rotated_position`
   - `test_target_velocity_independent_of_input_speed_magnitude`

2. **`test_find_linear_tangent_params_converges`**: end-to-end regression test using the real flight data from `main()`. Checks `res.success`, bounds on `coast_time`/`burn_time`, and that the resulting orbit (recomputed via the public API: `project`, `solve_coast`, `propagate_linear_tangent`, `orbital_elements`) is circular and at/near the target radius.

3. **`test_find_linear_tangent_params_memoizes_simulation`**: monkeypatches `Simulator.propagate_linear_tangent` with a spy that fingerprints each call's arguments (rounded), runs `find_linear_tangent_params` on real flight data, and asserts no fingerprint repeats — proving the memoized `simulate` helper is actually deduping identical evaluations requested by SLSQP's objective/constraint calls.

All 17 tests pass; `mypy --strict` and `pyright --warnings` are clean.

## Notes / Deviations from Original Sketch

- Considered attaching `res.final_r` / `res.final_v` / `res.final_orbit` onto the returned `OptimizeResult` for test convenience, but `mypy --strict` rejects dynamic attributes on `OptimizeResult` (`attr-defined`). Reverted; the regression test instead recomputes the final state via the existing public API (`project` + `solve_coast` + `propagate_linear_tangent`).
- Switched from dict-style `{"type": "eq", "fun": ...}` constraints to `scipy.optimize.NonlinearConstraint` objects, since the dict form didn't type-check cleanly against scipy's stubs under `mypy --strict` (constraint dicts with mixed-type values inferred as `dict[str, object]`, not matching the expected `_ConstraintDict` TypedDict). `NonlinearConstraint` is the modern, well-typed equivalent and behaves identically for SLSQP.
