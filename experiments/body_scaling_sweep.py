#!/usr/bin/env python3
"""Check invariance under physical body scaling."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from single_stage_research import (
    Stage,
    integrate_physical_polar,
    integrate_polar,
    norm,
    synthetic_case,
)


def run_body(
    name: str,
    radius: float,
    mu: float,
    target_altitude: float,
    initial_mass: float,
    thrust: float,
    exhaust_velocity: float,
) -> dict[str, float | str]:
    target_radius = radius + target_altitude
    target_velocity = math.sqrt(mu / target_radius)
    time_scale = target_radius / target_velocity
    gamma = thrust * target_radius / (initial_mass * target_velocity**2)
    kappa = exhaust_velocity / target_velocity
    stage = Stage(gamma=gamma, kappa=kappa, max_burn=0.25)
    case = synthetic_case(
        name,
        stage,
        rp=0.64,
        ra=0.87,
        true_anomaly=math.radians(120.0),
    )
    duration = 0.17
    alpha = 0.37
    normalized_final = integrate_polar(case.x0, duration, stage, 1.0, alpha)
    physical_initial = np.array(
        [
            case.x0[0] * target_radius,
            case.x0[1] * target_velocity,
            case.x0[2] * target_velocity,
            initial_mass,
        ]
    )
    physical_final = integrate_physical_polar(
        physical_initial,
        duration * time_scale,
        mu,
        thrust,
        exhaust_velocity,
        1.0,
        alpha,
    )
    converted_final = np.array(
        [
            physical_final[0] / target_radius,
            physical_final[1] / target_velocity,
            physical_final[2] / target_velocity,
            physical_final[3] / initial_mass,
        ]
    )
    return {
        "body": name,
        "gamma": gamma,
        "kappa": kappa,
        "error": norm(converted_final - normalized_final),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    bodies = [
        ("Kerbin", 600_000.0, 3.5316e12, 80_000.0, 13_885.65, 215_000.0, 3138.13),
        ("Mun", 200_000.0, 6.51384e10, 10_000.0, 2_500.0, 60_000.0, 3200.0),
        ("Minmus", 60_000.0, 1.7658e9, 10_000.0, 1_500.0, 30_000.0, 3200.0),
        ("Duna", 320_000.0, 3.01363e11, 30_000.0, 8_000.0, 120_000.0, 3300.0),
    ]
    records = [run_body(*body) for body in bodies]
    for record in records:
        print(
            f"{record['body']:8s} gamma={record['gamma']:.6f} "
            f"kappa={record['kappa']:.6f} error={record['error']:.3e}"
        )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
