# Validation

After making code changes, run:

```bash
mypy --strict sim.py
pyright --warnings sim.py
python3 -m pytest test_sim.py -v
```

