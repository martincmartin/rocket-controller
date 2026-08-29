# Single-Stage Ascent Optimizer

## Scope

This simulator uses Lawden's primer vector formulation. It models a launch from
a celestial body in Kerbal Space Program to a circular orbit. The model assumes
no drag and no atmosphere. The equations describe vacuum two-body motion in one
orbital plane. The simulator does not use kRPC. The flight controller is outside
this work.

The inputs are:

- Body radius $R$.
- Body gravitational parameter $\mu$.
- Target altitude $h$.
- Initial position and velocity in the orbital plane.
- Initial vehicle mass $m_0$.
- Stage thrust $T$.
- Effective exhaust velocity $v_e$.
- Maximum burn time.

The local state is radius, radial velocity, tangential velocity, and mass. The
tangential position is not a state because it does not affect the equations or
the target conditions.

## Normalization

The target radius, circular velocity, and time scale are

$$
r_\star=R+h,
\qquad
v_\star=\sqrt{\frac{\mu}{r_\star}},
\qquad
t_\star=\frac{r_\star}{v_\star}.
$$

The normalized state is

$$
\rho=\frac{r}{r_\star},
\qquad
u_r=\frac{v_r}{v_\star},
\qquad
u_t=\frac{v_t}{v_\star},
\qquad
\eta=\frac{m}{m_0},
\qquad
\tau=\frac{t}{t_\star}.
$$

The normalized stage parameters are

$$
\gamma=\frac{T r_\star}{m_0v_\star^2},
\qquad
\kappa=\frac{v_e}{v_\star}.
$$

The positive tangential axis is the desired orbital direction. For a physical
initial state, choose the local tangential basis from the sign of angular
momentum.

## Target

The target is an exact prograde circular orbit at the target radius:

$$
\rho_f=1,
\qquad
u_{r,f}=0,
\qquad
u_{t,f}=1.
$$

The objective is maximum final payload mass. For fixed initial mass and payload
definition, this is equivalent to minimum fuel consumption and minimum
delta-v consumed.

Final time is free. Final position is free. The final position can be any angle
within the target circular orbit.

## Time Domain

The current final-time cap is

$$
t_{\max}=t_{\mathrm{apo},0}+\frac{3}{4}T_0,
$$

where $t_{\mathrm{apo},0}$ is the time to the next initial apoapsis and

$$
T_0=2\pi\sqrt{\frac{a_0^3}{\mu}}.
$$

The normalized cap is used as a bound on the final time. The value of the
coefficient is part of the current research configuration. See
[Single-Stage Final-Time Domain](time-domain-study.md) for the time-domain
study.

## Provisional Solution

The current provisional single-stage mode sequence is

$$
q=1\;\longrightarrow\;q=0\;\longrightarrow\;q=1.
$$

The first burn starts at the supplied initial time. The vehicle then coasts and
performs a second burn until $t_f$. The solved parameter vector is

$$
z=(\alpha_0,\tau_{b1},\tau_c,\tau_{b2},
\lambda_{\rho,0},\lambda_{\eta,0}).
$$

The provisional solver is bounded `scipy.optimize.least_squares` applied to
the fixed-sequence segmented-shooting residual. It solves the terminal
conditions, both internal switching conditions, and either the free-final-time
Hamiltonian condition or the active final-time-bound condition.

The initial guesses are:

- Primer angle in the direction of the initial velocity:

  $$
  \alpha_0^{\mathrm{guess}}
  =\operatorname{atan2}(u_{t,0},u_{r,0}).
  $$

- First-burn duration from an impulse in the initial velocity direction that
  raises the apoapsis to $r_\star$, converted with the rocket equation.
- Coast duration from the post-first-impulse time to the new apoapsis, less
  the first-burn duration and half the second-burn duration estimate.
- Second-burn duration from the vis-viva circularization impulse, converted
  with the rocket equation.
- Mass costate initialized with

  $$
  \lambda_{\eta,0}^{\mathrm{guess}}=-\kappa.
  $$

- Radial-position costate initialized from $H_0=0$ when
  $|u_{r,0}|$ is above the configured threshold. A separate branch is required
  when $u_{r,0}$ is zero or close to zero.

## Research Records

The equations are defined in [Primer Equations](primer-equations.md).

The choice of the on/off/on sequence is documented in
[Burn Sequence Study](burn-sequence-study.md).

The removal of the direct optimizer from the runtime initialization path is
documented in [Initial Guess Study](initial-guess-study.md).

Solver measurements are in [Solver Comparison](solver-comparison.md).

The complete fixture catalog is in
[Single-Stage Cases and Generalization](single-stage-test-cases.md).

The first-burn physical interpretation is in
[Optimal Burn Trajectory](optimal-burn-trajectory.md).
