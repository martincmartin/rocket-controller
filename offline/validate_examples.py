"""Run the complete offline oracle validation fixtures.

This command keeps the expensive multi-fixture solver check out of the normal
unit-test path::

    python3 -m offline.validate_examples
"""

from __future__ import annotations

from offline.oracle_finder import find_oracle, kerbin_examples

EXPECTED_SELECTED = {
    "kerbin-example": "burn_coast_burn",
    "kerbin-first-example": "lawden",
    "kerbin-coast-first-example": "lawden",
}


def main() -> None:
    for name, (position, velocity, segments) in kerbin_examples().items():
        result = find_oracle(position, velocity, segments)
        selected = result.selected
        if not selected.success:
            raise RuntimeError(f"{name}: selected candidate was not accepted")
        if selected.name != EXPECTED_SELECTED[name]:
            raise RuntimeError(
                f"{name}: expected {EXPECTED_SELECTED[name]}, got {selected.name}"
            )
        if selected.residual_norm >= 3e-6:
            raise RuntimeError(
                f"{name}: residual {selected.residual_norm:.3e} is too large"
            )
        fuel = selected.fuel_consumed_kg(result.problem)
        if fuel is None:
            raise RuntimeError(f"{name}: selected candidate has no fuel metric")
        print(
            f"{name:28s} {selected.name:18s} "
            f"fuel={fuel:.6f} kg "
            f"res={selected.residual_norm:.3e}"
        )


if __name__ == "__main__":
    main()
