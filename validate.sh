#!/usr/bin/env bash
#
# Run every check this project's AGENTS.md requires after a code change:
# type checking (mypy/pyright), tests (pytest), and ruff (formatting +
# linting). Runs everything and reports a summary at the end, rather than
# stopping at the first failure, so a single run shows the full picture.
#
# Usage: ./validate.sh

set -u

cd "$(dirname "${BASH_SOURCE[0]}")"
shopt -s globstar

# Activate the project virtualenv if one exists and isn't already active
# (mypy/pyright/pytest are installed there, not necessarily on the caller's
# plain PATH).
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

failed=()

run() {
    local name="$1"
    shift
    echo
    echo "=== ${name} ==="
    if "$@"; then
        echo "--- ${name}: OK ---"
    else
        echo "--- ${name}: FAILED ---"
        failed+=("${name}")
    fi
}

run "ruff format --check" ruff format --check .
run "ruff check" ruff check .
run "mypy --strict" mypy --strict sim.py sim_experiment.py timing.py KSPUtils.py autopilot_thread.py guidance_link.py autopilot.py gravity_turn.py krpc_batch.py experiments/**/*.py
run "pyright --warnings" pyright --warnings sim.py sim_experiment.py timing.py KSPUtils.py autopilot_thread.py guidance_link.py autopilot.py gravity_turn.py krpc_batch.py experiments/**/*.py
run "pytest" python3 -m pytest test_sim.py test_sim_experiment.py test_autopilot.py test_KSPUtils.py test_guidance_link.py test_autopilot_thread.py test_krpc_batch.py -v

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All checks passed."
    exit 0
else
    echo "FAILED: ${failed[*]}"
    exit 1
fi
