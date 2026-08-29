# Single-Stage Initial-Guess Study

## Scope

This document records the change from direct-optimizer initialization to
analytic initialization. The current burn sequence is

$$
q=1\;\longrightarrow\;q=0\;\longrightarrow\;q=1.
$$

The sequence and its residual are defined in
[Burn Sequence Study](burn-sequence-study.md). The equations are defined in
[Primer Equations](primer-equations.md).

## Initial Direct Method

The first research implementation ran a direct two-burn optimizer before the
primer solve. It parameterized the two burn durations, the coast duration, and
piecewise constant thrust angles. It enforced the terminal circular-orbit
conditions.

The direct result supplied values for

$$
(\alpha_0,\tau_{b1},\tau_c,\tau_{b2}).
$$

The primer solver then supplied the costate values. This method was used to
obtain comparison data. It is not part of the runtime path because each direct
optimizer evaluation integrates every control interval and the solver uses
multiple deterministic seeds.

## Analytic Initial Values

### Initial Primer Angle

The current velocity angle, measured from the local radial axis, is

$$
\alpha_0^{\mathrm{guess}}
=\operatorname{atan2}(u_{t,0},u_{r,0}).
$$

This is the initial velocity direction. The local tangential angle
$\pi/2$ is a different direction when $u_{r,0}\ne0$.

The prograde direction at the predicted apoapsis remains available as a
diagnostic. It is not the current runtime seed.

### First-Burn Timing

Apply an instantaneous impulse in the initial velocity direction. Define

$$
V_0=\sqrt{u_{r,0}^2+u_{t,0}^2},
\qquad
\widehat{\mathbf v}_0=\frac{(u_{r,0},u_{t,0})}{V_0}.
$$

For a trial impulse magnitude $\Delta v_1$, the post-impulse velocity is

$$
(u_{r,1},u_{t,1})
=(u_{r,0},u_{t,0})+\Delta v_1\widehat{\mathbf v}_0.
$$

Find $\Delta v_1>0$ such that the resulting orbit has apoapsis
$r_\star$. For that trial velocity,

$$
\epsilon_1
=\frac{1}{2}(u_{r,1}^2+u_{t,1}^2)-\frac{1}{\rho_0},
\qquad
h_1=\rho_0u_{t,1},
$$

$$
a_1=-\frac{1}{2\epsilon_1},
\qquad
e_1=\sqrt{1+2\epsilon_1h_1^2},
\qquad
\rho_{a,1}=a_1(1+e_1).
$$

Solve $\rho_{a,1}=1$. The first-burn estimate is the constant-thrust time
for $\Delta v_1$.

Convert this impulse to an ideal constant-thrust duration:

$$
\eta_1=\exp\left(-\frac{\Delta v_1}{\kappa}\right),
\qquad
\tau_{b1}^{\mathrm{guess}}
=\frac{\kappa}{\gamma}(1-\eta_1).
$$

### Coast And Second-Burn Timing

Propagate the post-impulse state to its new apoapsis. Use the vis-viva equation

$$
v^2=\mu\left(\frac{2}{r}-\frac{1}{a}\right)
$$

to compute the circularization impulse at $r_\star$:

$$
\Delta v_2=v_{\mathrm{circ}}(r_\star)-v_{a,1}.
$$

The ideal second-burn duration is

$$
\eta_2=\eta_1\exp\left(-\frac{\Delta v_2}{\kappa}\right),
\qquad
\tau_{b2}^{\mathrm{guess}}
=\frac{\kappa}{\gamma}(\eta_1-\eta_2).
$$

If $\tau_{a,1}$ is the coast time from the initial point to the new
apoapsis, use

$$
\tau_c^{\mathrm{guess}}
=\tau_{a,1}-\tau_{b1}^{\mathrm{guess}}
 -\frac{1}{2}\tau_{b2}^{\mathrm{guess}}.
$$

Clamp the result to the configured final-time domain.

### Mass Costate

Set

$$
\lambda_{\eta,0}^{\mathrm{guess}}=-\kappa.
$$

This value follows from

$$
P_0=1,
\qquad
\eta_0=1,
\qquad
\Phi_0=\frac{P_0}{\eta_0}
 +\frac{\lambda_{\eta,0}}{\kappa}=0.
$$

The equality is an initialization relation. The first arc is powered, so the
runtime solver does not impose $\Phi_0=0$. The solver changes
$\lambda_{\eta,0}$ and imposes the two internal switch equations.

### Radial-Position Costate

For $u_{r,0}\ne0$, solve $H_0=0$ for $\lambda_{\rho,0}$. With

$$
p_{r,0}=\cos\alpha_0,
\qquad
p_{t,0}=\sin\alpha_0,
$$

the initial value is

$$
\lambda_{\rho,0}^{\mathrm{guess}}
=\frac{
p_{r,0}\left(\dfrac{u_{t,0}^2}{\rho_0}-\dfrac{1}{\rho_0^2}\right)
-p_{t,0}\dfrac{u_{r,0}u_{t,0}}{\rho_0}
+\gamma\left(\dfrac{P_0}{\eta_0}
+\dfrac{\lambda_{\eta,0}^{\mathrm{guess}}}{\kappa}\right)
}{u_{r,0}}.
$$

The current initialization does not apply this formula when
$|u_{r,0}|$ is below its threshold. It must keep
$\lambda_{\rho,0}$ as a solved variable and use continuation or another finite
initial value.

## Evidence

The direct optimizer was used offline to define the oracle. The analytic seed
was then passed to the fixed-sequence solver without the direct result.

For the four named cases and eight deterministic randomized cases:

| Angle seed | Accepted solves |
| --- | ---: |
| Initial velocity direction | $12/12$ |
| Local tangential direction $\pi/2$ | $12/12$ |
| Prograde direction at predicted apoapsis | $7/12$ |

The initial velocity direction is now the runtime seed. It matches the control
direction of the first burn in the tested cases.

The timing estimate produced the following results for the four named cases:

| Case | $\tau_{b1}^{\mathrm{guess}}/\tau_{b1}$ | $\tau_c^{\mathrm{guess}}/\tau_c$ | $\tau_{b2}^{\mathrm{guess}}/\tau_{b2}$ |
| --- | ---: | ---: | ---: |
| Synthetic moderate | $1.008$ | $1.004$ | $0.995$ |
| Synthetic high thrust | $1.014$ | $1.010$ | $0.986$ |
| Kerbin second example | $0.913$ | $0.788$ | $1.470$ |
| Kerbin first example | $1.939$ | $1.075$ | $0.720$ |

The timing estimate gives a valid initial point for all twelve tested cases.
The Kerbin results show that the burn-duration allocation can differ from the
ideal impulse allocation. The solver must keep all three durations as variables.

## Runtime Cost

The research scripts perform direct optimization, costate scans, and cap scans.
Those operations are not runtime requirements. The runtime path performs one
timing estimate and one fixed-sequence bounded least-squares solve.

`experiments/runtime_benchmark.py` measures this path. The previous benchmark
measured less than $0.1\ \textrm{s}$ per supplied Kerbin case on the current
machine. The benchmark uses the initial velocity direction.

## Current Status

The direct optimizer is retained as an offline falsification reference. It is
not required to initialize the provisional runtime solver.

The oracle values in the test-case catalog were generated before the final-time
cap changed from $1/4$ to $3/4$ of the initial orbital period after apoapsis.
Regenerate cap-dependent oracle values before production regression tests use
them.

## Falsifiers

- A physically valid case where the analytic seed fails and the direct seed
  succeeds.
- A case where the impulse timing selects the wrong arc sequence.
- A case with $u_{r,0}=0$ or near zero that cannot be initialized by the
  independent-costate branch.
- A singular throttle interval that does not converge to bang-bang control.
