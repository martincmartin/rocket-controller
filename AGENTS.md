# Validation

After making code changes, run:

```bash
mypy --strict sim.py
pyright --warnings sim.py
python3 -m pytest test_sim.py -v
```

# Formatting
use `ruff format .` for formatting all Python files.

# Planing
When asked to create a new plan, write it to PLAN.md, overwriting anything there
from a previous task.  When asked to modify the plan, modify the contents of
PLAN.md so it always reflects the most up to date plan.
