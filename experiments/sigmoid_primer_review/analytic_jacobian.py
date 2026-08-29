#!/usr/bin/env python3
"""Analytic Jacobian prototype for the sigmoid shooting system, via
variational sensitivities with a complex-step Jacobian of the ODE right-hand
side.

Evidence for `review-sigmoid-primer.md` section 3.3. This prototype is
intentionally NOT used by `production_solver.py` in this directory (the final
suggestion uses finite differences only), but it is kept here as a record in
case we pursue the analytic Jacobian later.

What it demonstrates:

- With the exact Jacobian, the eps=1e-4 solve from the oracle seed converges
  cleanly (67 evaluations) to the same smeared root the finite-difference
  solvers find, confirming that the smeared root is the system's only
  accessible root there, not an artifact of finite differences.
- Below eps~1e-5, even exact gradients plateau at residual ~1e-5 because the
  shooting Jacobian is nearly singular (condition number ~1e8 at eps=1e-6).

Implementation notes:

- The 8x8 Jacobian of `sigmoid_primer_rhs` is computed with a complex step
  (h = 1e-100) through a complex-safe copy of the RHS.
- The sensitivity matrix d(joint)/d(parameters) (8x4) is integrated alongside
  the 8 state-costate components, giving the 4x4 residual Jacobian without
  finite differences.
- The tf column of the residual Jacobian is d(r)/d(tf) = f(joint(tf)) for the
  state rows and grad(S).f(joint(tf)) for the switching row.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/analytic_jacobian.py
"""

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import Case

Array = NDArray[np.float64]

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)
SMEARED = np.array(
    [1.3557489599336592, 0.7149047496714878, -1.1228459941245825, -1.3730707279997247]
)


def stable_expit(z: Any) -> Any:
    """Overflow-safe complex sigmoid 1/(1+exp(-z)) for a scalar argument."""
    value = complex(z)
    if value.real > 0:
        return 1.0 / (1.0 + np.exp(-value))
    return np.exp(value) / (1.0 + np.exp(value))


def rhs_complex(
    _time: float, joint: Any, gamma: float, kappa: float, epsilon: float
) -> Any:
    """Complex-step-safe copy of sigmoid_primer_rhs."""
    (
        rho,
        ur,
        ut,
        mass,
        lam_rho,
        p_r,
        p_t,
        lam_mass,
    ) = joint
    if rho.real <= 0.0 or mass.real <= 0.0:
        return np.full(8, np.nan)
    primer_length = np.sqrt(p_r * p_r + p_t * p_t)
    if primer_length.real <= 1e-14:
        return np.full(8, np.nan)
    switch = primer_length / mass + lam_mass / kappa
    throttle = stable_expit(switch / epsilon)
    acceleration = throttle * gamma / mass
    return np.array(
        [
            ur,
            ut * ut / rho - 1.0 / rho**2 + acceleration * p_r / primer_length,
            -ur * ut / rho + acceleration * p_t / primer_length,
            -throttle * gamma / kappa,
            p_r * (2.0 / rho**3 - ut * ut / rho**2) + p_t * ur * ut / rho**2,
            lam_rho + p_t * ut / rho,
            -2.0 * p_r * ut / rho + p_t * ur / rho,
            -throttle * gamma * primer_length / mass**2,
        ],
        dtype=complex,
    )


def jf_complex(joint: Array, gamma: float, kappa: float, epsilon: float) -> Array:
    """8x8 Jacobian of sigmoid_primer_rhs via complex step."""
    size = len(joint)
    out = np.zeros((size, size))
    step = 1e-100
    for column in range(size):
        perturbed = joint.astype(complex)
        perturbed[column] += 1j * step
        out[:, column] = rhs_complex(0.0, perturbed, gamma, kappa, epsilon).imag / step
    return out


def initial_augmented(case: Case, parameters: Array) -> tuple[Array, Array]:
    alpha, _final_time, _lambda_rho, _lambda_mass = parameters
    joint0 = sp.initial_joint(case, parameters)
    sensitivity0 = np.zeros((8, 4))
    sensitivity0[4, 2] = 1.0  # d/d lambda_rho0
    sensitivity0[5, 0] = -math.sin(float(alpha))  # d/d alpha0 of cos
    sensitivity0[6, 0] = math.cos(float(alpha))  # d/d alpha0 of sin
    sensitivity0[7, 3] = 1.0  # d/d lambda_mass0
    return joint0, sensitivity0


def augmented_rhs(
    _time: float, y: Array, gamma: float, kappa: float, epsilon: float
) -> Array:
    joint = y[:8]
    sensitivity = y[8:].reshape(8, 4)
    flow = sp.sigmoid_primer_rhs(_time, joint, gamma, kappa, epsilon)
    sensitivity_dot = jf_complex(joint, gamma, kappa, epsilon) @ sensitivity
    return np.concatenate((flow, sensitivity_dot.ravel()))


def residual_with_jac(
    case: Case, parameters: Array, epsilon: float
) -> tuple[Array, Array]:
    gamma, kappa = case.stage.gamma, case.stage.kappa
    joint0, sensitivity0 = initial_augmented(case, parameters)
    solution = solve_ivp(
        augmented_rhs,
        (0.0, float(parameters[1])),
        np.concatenate((joint0, sensitivity0.ravel())),
        args=(gamma, kappa, epsilon),
        method="DOP853",
        rtol=sp.RTOL,
        atol=sp.ATOL,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    final = solution.y[:, -1]
    joint_f, sensitivity_f = final[:8], final[8:].reshape(8, 4)
    rho, radial_velocity, tangential_velocity, mass = joint_f[:4]
    residual = np.array(
        [
            rho - 1.0,
            radial_velocity,
            tangential_velocity - 1.0,
            sp.switching_function(joint_f, kappa),
        ]
    )
    grad_switch = np.zeros(8)
    primer_length = math.hypot(float(joint_f[5]), float(joint_f[6]))
    grad_switch[3] = -primer_length / mass**2
    grad_switch[5] = joint_f[5] / (primer_length * mass)
    grad_switch[6] = joint_f[6] / (primer_length * mass)
    grad_switch[7] = 1.0 / kappa
    jacobian = np.zeros((4, 4))
    jacobian[0] = sensitivity_f[0]
    jacobian[1] = sensitivity_f[1]
    jacobian[2] = sensitivity_f[2]
    jacobian[3] = grad_switch @ sensitivity_f
    # d r / d tf = (d r / d joint) . f(tf); the state rows are e_i . f = f[i]
    flow_final = sp.sigmoid_primer_rhs(0.0, joint_f, gamma, kappa, epsilon)
    jacobian[0, 1] = flow_final[0]
    jacobian[1, 1] = flow_final[1]
    jacobian[2, 1] = flow_final[2]
    jacobian[3, 1] += float(grad_switch @ flow_final)
    return residual, jacobian


def solve_analytic(seed: Array, epsilon: float, tag: str, max_nfev: int = 200) -> Array:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0], dtype=float)
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0], dtype=float)

    def residual_function(parameters: Array) -> Array:
        return residual_with_jac(CASE, parameters, epsilon)[0]

    def jacobian_function(parameters: Array) -> Array:
        return residual_with_jac(CASE, parameters, epsilon)[1]

    result = least_squares(
        residual_function,
        seed,
        jac=jacobian_function,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=max_nfev,
    )
    parameters = np.asarray(result.x, dtype=float)
    residual = sp.sigmoid_residual(CASE, parameters, epsilon)
    solution = sp.integrate_sigmoid(CASE, parameters, epsilon, dense_output=True)
    if solution.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, parameters[1], 1601)
    joints = np.asarray(solution.sol(samples).T)
    switch = np.array([sp.switching_function(j, CASE.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    crossings = sp.switch_crossings(samples, switch, parameters[1])
    print(
        f"{tag:16s} eps={epsilon:.1e} ok={result.success} {result.message[:26]:26s} "
        f"nfev={result.nfev:4d} res={np.linalg.norm(residual):.3e} "
        f"tf={parameters[1]:.9f} fuel={1 - joints[-1][3]:.9f} "
        f"qmin={throttle.min():.3e} crosses={[f'{c:.5f}' for c in crossings]}"
    )
    return parameters


def solve_analytic_safe(seed: Array, epsilon: float, tag: str) -> Array | None:
    try:
        return solve_analytic(seed, epsilon, tag)
    except (ValueError, FloatingPointError) as error:
        print("failed:", error)
        return None


if __name__ == "__main__":
    print("=== analytic Jacobian, from ORACLE, various eps ===")
    seed = ORACLE.copy()
    for eps in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        result = solve_analytic_safe(seed, eps, "analytic-oracle")
        if result is None:
            break
        seed = result
    print("=== analytic Jacobian, from SMEARED at 1e-4 ===")
    solve_analytic_safe(SMEARED, 1e-4, "analytic-smeared")
