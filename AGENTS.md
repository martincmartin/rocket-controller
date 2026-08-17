# Validation

After making code changes, run:

```bash
./validate.sh
```

Checks format, lint, typing (mypy `--strict` and pyright) and runs tests. The
script activates `.venv` if one exists, runs every step regardless of failures,
and prints a summary so one invocation shows the full picture.

`validate.sh` only *checks* (it never rewrites files). The Python files it
type-checks are listed explicitly in `validate.sh` itself -- when you add a new
non-test `.py` module, append it to both the `mypy` and `pyright` invocations
there.

# Formatting

- `ruff format .` -- format all Python files.
- `ruff check --fix .` -- auto-fix lint findings.

Rule set is in `pyproject.toml`. Note `RUF003` is intentionally ignored (the
codebase uses Unicode like ×, µ, °, ── in comments and strings).

# Architecture

Two layers, no overlap:

- `sim.py` -- game-independent astrodynamics. Physics, orbital mechanics,
  trajectory propagation, numerical optimization. Has no kRPC / KSP imports.
  This is where you change planning logic.
- `gravity_turn.py` -- flight controller. Talks to KSP via kRPC, executes plans
  on the live vehicle: launch, gravity turn, staging, throttle, telemetry,
  circularization burns. This is where you change flight behavior.

Supporting modules: `KSPUtils.py` (KSP-specific helpers / stream wrappers),
`autopilot.py` (autopilot state machine used by `gravity_turn.py`),
`autopilot_thread.py` (background replanning thread), `guidance_link.py` and
`krpc_batch.py` (kRPC plumbing).

The split is enforced conceptually: planner code must not import kRPC; flight
code must not contain physics. If you're tempted to cross the line, you
probably want a new helper instead.

# Scratch files (not part of the repo)

`explore.py` is a gitignored scratchpad for ad-hoc analysis scripts. Write
throwaway scripts there rather than in the repo root -- they won't be committed
and won't show up in `git status`. PLAN.md is also gitignored.

# Planning

When asked to create or modify a plan, write it to `PLAN.md`, overwriting any
previous contents so the file always reflects the most up-to-date plan.

# kRPC connection note (from TODO.md)

If a second reader thread is ever introduced sharing a `KSPStreams` / kRPC
connection with an existing one, beware the lock-ordering deadlock risk from
holding multiple streams' `.condition`s at once. Acquire them in one consistent
order (e.g. sorted by `id()`), or route multi-condition reads through a single
gatekeeper lock. The simpler answer is one kRPC connection per thread -- those
calls block until they get a reply.
