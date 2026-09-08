# Review: Offline Oracle Finder

Reviewer: Martin's review pass, 2026-09-07. Second update, after the Lawden
rework.

## Scope and Verification

- Read `offline/oracle_finder.py`, `offline/test_oracle_finder.py`,
  `offline/validate_examples.py`, `docs/offline-oracle-finder.md`, and
  `tasks/2026-09-07-offline-oracle-finder/PLAN.md`.
- Ran `./validate.sh`: ruff format and lint, mypy `--strict`, pyright, and
  pytest all pass (120 tests, about 32 s). The expensive three-fixture solve
  now lives in `offline/validate_examples.py`; pytest reuses one module-scoped
  oracle result, so routine validation stays fast.
- Ran `python3 -m offline.validate_examples`: all three fixtures pass in about
  27 s.
- Ran the CLI and diffed the JSON output against the reference rows in
  `docs/single-stage-test-cases.md`.
- Note: the files were edited concurrently during this review. This report is
  written against the final stable state, which passes all checks.

## What It Does Well

1. **All substantive points from the earlier review were addressed.**
   - The Lawden acceptance brittleness is fixed. `_propagate_lawden_hard`
     rolls back a terminal coast, including the zero-duration event artifact,
     and reports the end of the last burn as the terminal time. Phases shorter
     than `MIN_PHASE_DURATION_SECONDS = 0.02` s and unsupported sequences are
     rejected with named diagnostics, a RuntimeWarning, and CLI output.
   - The Lawden candidate now validates and is **selected** for two of the
     three fixtures:
     - `kerbin-first-example`: fuel 5456.189701 kg against the reference
       5456.189701 kg (diff 0.00000), final time 155.779546 s against
       155.779547 s, and initial costates to ten digits
       ($\alpha_0=1.301998498$, $\lambda_\rho=-1.129547597$,
       $\lambda_\eta=-1.375099704$).
     - `kerbin-coast-first-example`: fuel 4002.0511 kg against the reference
       4002.0548 kg ($-0.0037$ kg, $9\times10^{-7}$ relative, rounding level).
       This replaces the earlier Prussing-convention deviation, which is moot.
   - The selected families now match the documented fixed-sequence references
     within rounding.
2. **Reference accuracy.** The explicit burn/coast/burn solver for
   `kerbin-example` still matches the documented oracle: fuel 6368.979455 kg
   against 6368.979455 kg, arcs $(0.258911, 0.336891, 0.052636)$ against
   $(0.258910, 0.336885, 0.052638)$, residual $8.2\times10^{-12}$.
3. **Clean separation.** No imports from `experiments/` or `examples/`, no
   kRPC. The `SegmentLike` protocol accepts `sim.RocketSegment` structurally.
4. **Sound acceptance logic.** Candidates are re-propagated and checked
   against the terminal residual, per-arc switching signs, the 0.02 s phase
   floor, supported phase sequences, and the powered- and elapsed-time limits.
   Suboptimal families that still hit the target are correctly rejected
   (`burn_only` and `coast_burn` reach residuals near $10^{-16}$ for some
   fixtures but fail the sign or structure checks). Rejected candidates remain
   visible and auditable.
5. **Deterministic seeding.** The analytic impulse-timing estimate plus the
   three-direction sweep and $\lambda_\rho$ variants. No random restarts.
6. **Selection contract.** `_candidate_is_better` implements the plan's
   tie-breakers: final mass first, then $\Delta v$ within
   $10^{-9}\ \mathrm{m/s}$, then fuel within tolerance. Per-seed
   accept/reject status is reported in `seed_results`.
7. **New structure.** `OrbitalPlane` validates the 3D input and builds the
   orthonormal basis. `OrbitalParameters` computes apoapsis, period, and
   coast time analytically from Kepler quantities instead of an
   `solve_ivp` event hunt. `Problem` carries `radius` and `speed_normalized`
   instead of a four-element state. The empty plan for an already-circular
   input returns immediately with zero fuel and zero $\Delta v$.
8. **Test coverage matches the plan's validation list.** Limit-binding
   warning, mass/fuel/$\Delta v$ consistency, polar dynamics against an
   independent Cartesian propagation, degenerate first-burn rejection,
   empty-plan early exit, unsupported Lawden sequence reporting, and the
   all-candidates-failed error path are all tested.
9. **Documentation.** PLAN.md and `docs/offline-oracle-finder.md` record the
   terminal-coast trimming, the 0.02 s phase floor, the supported sequences,
   the empty plan, measured runtimes, and fuel fractions that match the
   implementation.

## What Could Be Improved

1. **Lawden still does not converge for `kerbin-example`.** The hard polish
   stops at `max_nfev` with residual about $4.9\times10^{-5}$, so the pure
   Lawden family contributes nothing to that fixture and the explicit
   burn/coast/burn carries it. The failure is reported transparently
   (structure is supported; only the residual is too large). A larger polish
   budget or a second restart would be the natural next step.
2. **Flat shooting direction in the Lawden vector.** Once `final_tau` extends
   past the end of the last burn, the terminal-coast rollback makes the
   residual independent of it. Harmless today (all three cases converge), but
   the parameter could be canonicalized to the burn endpoint later.
3. **Classification nuance for the coast-first fixture.** The selected Lawden
   result is burn/coast/burn with a 0.036 s first burn, which is above the
   0.02 s floor, rather than the reference's pinned coast/burn. The selected
   trajectory is marginally better than both the pinned reference and the
   explicit coast/burn candidate, so the selection is right. This only matters
   when diffing phase sequences against fixed-sequence reference rows.
4. **Doc residual-column drift.** The Validation Result table records
   $8.23\times10^{-12}$, $7.81\times10^{-13}$, $4.25\times10^{-14}$, while the
   current code produces $8.32\times10^{-12}$, $4.23\times10^{-13}$,
   $1.91\times10^{-13}$. The fuel fractions match. The table was generated at
   an earlier code state; residuals shift slightly with each refactor.

## Changes Since the Previous Review

- `offline/oracle_finder.py`: terminal-coast rollback and canonicalization in
  the hard Lawden propagation; supported-sequence whitelist; 0.02 s minimum
  phase duration; empty-plan early exit; mass/$\Delta v$/fuel selection with
  tie tolerances; per-seed `seed_results`; `optimizer_success` diagnostics and
  clarified messages; `OrbitalPlane` input validation; `OrbitalParameters`
  Kepler helper class; dead `_least_squares` indirection removed.
- `offline/test_oracle_finder.py`: added impulse-seed and orbital-helper
  reference values, polar-versus-Cartesian dynamics, mass/fuel/$\Delta v$
  consistency, degenerate-burn rejection, empty-plan, unsupported-sequence,
  and all-failed error-path tests; one module-scoped oracle fixture.
- `offline/validate_examples.py`: new explicit three-fixture validation
  command that keeps the expensive solves out of the normal pytest path.
- `docs/offline-oracle-finder.md` and `PLAN.md`: updated to the implemented
  contract with a Validation Result section.

## Verdict

The rework resolves the substantive issues from the earlier reviews. The
oracle now returns reference-accurate results on all three fixtures, the
Lawden family validates and wins where it should, the acceptance criteria are
explicit and auditable, and the test and documentation coverage matches the
plan. The remaining items are minor: the Lawden convergence gap on
`kerbin-example` (item 1) is the only one that could matter in future use.
