# Validation

After making code changes, run:

```bash
./validate.sh
```

Checks format, lint, typing and runs tests.

# Formatting
use `ruff format .` for formatting all Python files, and
`ruff check --fix .` to auto-fix lint findings (rule set configured in
`pyproject.toml`). `validate.sh` only checks (`ruff format --check`,
`ruff check`) -- it never rewrites files itself.

# Planing
When asked to create a new plan, write it to PLAN.md, overwriting anything there
from a previous task.  When asked to modify the plan, modify the contents of
PLAN.md so it always reflects the most up to date plan.
