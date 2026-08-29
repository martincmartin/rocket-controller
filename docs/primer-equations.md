# Single-Stage Primer Equations

## Status

This document records the single-stage model under test. The equations are not
accepted yet. They must pass the numerical checks in
`experiments/single_stage_research.py` before they are used by an optimizer.

The model is vacuum two-body motion in one orbital plane. The independent
variable is normalized time, and the tangential position is omitted because it
does not affect the dynamics or the terminal orbit conditions.

## Coordinates

Use local radial and tangential components:

$$
r,\qquad v_r,\qquad v_t,\qquad m.
$$

The tangential basis is selected so that positive $v_t$ is the desired orbital
direction. The thrust angle is the primer angle $\alpha$, measured from the
radial basis toward the tangential basis.

## Normalization

For body radius $R$, target altitude $h$, and gravitational parameter $\mu$,
define

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

For a constant-thrust stage, define

$$
\gamma=\frac{T r_\star}{m_0v_\star^2},
\qquad
\kappa=\frac{v_e}{v_\star}.
$$

The target circular orbit is $\rho=1$, $u_r=0$, and $u_t=1$.

## State Equations

The primary control is bang-bang throttle $q\in\{0,1\}$. A relaxed
$q\in[0,1]$ model is reserved for independent direct optimization and
diagnostics.

$$
\rho'=u_r,
$$

$$
u_r'=\frac{u_t^2}{\rho}-\frac{1}{\rho^2}
       +\frac{q\gamma}{\eta}\cos\alpha,
$$

$$
u_t'=-\frac{u_ru_t}{\rho}
       +\frac{q\gamma}{\eta}\sin\alpha,
$$

$$
\eta'=-\frac{q\gamma}{\kappa}.
$$

The terms containing $u_t^2$ and $u_ru_t$ are the polar-frame curvature
terms. Omitting them would not describe the same motion as the Cartesian
two-body equations.

## Primer Equations

Use the minimum-fuel convention $J=-\eta_f$. Define

$$
\mathbf p=(p_r,p_t)=-(\lambda_{u_r},\lambda_{u_t}),
\qquad
P=\sqrt{p_r^2+p_t^2}.
$$

The thrust direction under test is

$$
\cos\alpha=\frac{p_r}{P},
\qquad
\sin\alpha=\frac{p_t}{P}.
$$

The switching function under test is

$$
\Phi=\frac{P}{\eta}+\frac{\lambda_\eta}{\kappa}.
$$

For the stated sign convention, the control law under test is

$$
q=
\begin{cases}
1,&\Phi>0,\\
0,&\Phi<0.
\end{cases}
$$

At $\Phi=0$, the first-order Hamiltonian condition does not determine the
throttle. A sustained near-zero interval must be tested for a singular arc.

The costate equations under test are

$$
\lambda_\rho'
=p_r\left(\frac{2}{\rho^3}-\frac{u_t^2}{\rho^2}\right)
 +p_t\frac{u_ru_t}{\rho^2},
$$

$$
p_r'=\lambda_\rho+\frac{p_tu_t}{\rho},
\qquad
p_t'=-\frac{2p_ru_t}{\rho}+\frac{p_tu_r}{\rho},
$$

$$
\lambda_\eta'=-\frac{q\gamma P}{\eta^2}.
$$

The minimized Hamiltonian is

$$
H=\lambda_\rho u_r
 -p_r\left(\frac{u_t^2}{\rho}-\frac{1}{\rho^2}\right)
 +p_t\frac{u_ru_t}{\rho}
 -q\gamma\left(\frac{P}{\eta}+\frac{\lambda_\eta}{\kappa}\right).
$$

## Terminal Conditions

The exact target conditions are

$$
\rho_f=1,
\qquad
u_{r,f}=0,
\qquad
u_{t,f}=1.
$$

Free final time requires $H(t_f)=0$. For a continuous single-stage
trajectory, $H(0)=0$ should be equivalent, but both are measured in the
experiments.

The costate at the terminal point will additionally be checked against

$$
\boldsymbol{\lambda}(t_f)
=\sigma_0\nabla(-\eta_f)
 +Dg(x_f)^T\boldsymbol{\zeta},
$$

where $g$ contains the three terminal constraints. This is a normality check,
not an additional shooting variable requirement.

## Initial Versus Final Hamiltonian

The problem has no explicit time dependence. On every smooth arc,

$$
\frac{dH}{dt}=-\frac{\partial H}{\partial t}=0.
$$

At a throttle switch, $\Phi=0$ makes the powered and unpowered Hamiltonians
equal. Therefore, for a continuous single-stage trajectory with no state or
costate jump,

$$
H(t_0)=H(t_f).
$$

The free-final-time transversality condition is formally applied at the final
time:

$$
H(t_f)=0.
$$

Using $H(t_0)=0$ as a shooting residual is equivalent only after the solver
checks Hamiltonian conservation and switch continuity. Forward shooting makes
the initial value convenient to evaluate, but it does not change the
transversality condition.

If the final-time cap is active, final time is not free. The correct residual is

$$
t_f-t_{\max}=0,
$$

and $H(t_f)=0$ must not also be imposed. A future staged model with a state or
costate jump must apply the appropriate interface condition instead of assuming
Hamiltonian continuity.

## Numerical Tests

The following tests are required before accepting the equations:

- Transform randomized normalized polar states and fixed thrust directions to
  Cartesian coordinates. Integrate both models and compare the recovered polar
  states.
- Compare the analytic costate right-hand side with finite differences of the
  minimized Hamiltonian.
- Change physical unit scales while preserving the normalized parameters and
  compare the converted trajectories.
- Check two-body energy and angular-momentum behavior during coast.
- Check the impulsive limit against the vis-viva and rocket-equation estimate.

Any systematic coordinate mismatch, derivative error, or unit dependence
rejects this formulation before optimization work continues.

## First Numerical Result

The validation experiment was run with DOP853 integration and relative and
absolute tolerances of approximately $2\times10^{-11}$ and
$2\times10^{-13}$ in normalized units. It checked one fixed state and thirty
two independently generated states with positive radial velocity.

The observed maximum errors were:

| Check | Maximum error |
| --- | ---: |
| Polar versus Cartesian propagation | $4.04\times10^{-12}$ |
| Costate equation versus Hamiltonian finite difference | $1.54\times10^{-10}$ |
| Physical-unit rescaling | $2.77\times10^{-13}$ |
| Coast specific-energy drift | $6.99\times10^{-15}$ |
| Coast angular-momentum drift | $1.44\times10^{-15}$ |

These results support the state and costate equations for the tested regular
domain. They do not validate the optimizer, the switching sequence, singular
arcs, nearly radial motion, or global optimality. The remaining falsifiers are
the optimizer comparisons and the edge-case sweep.
