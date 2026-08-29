# Single-Stage Burn Sequence Study

## Scope

This document records the sequence selection for the single-stage solver. The
state and costate equations are in [Primer Equations](primer-equations.md).
The initial values are in [Initial Guess Study](initial-guess-study.md).

## Sequence Definition

The selected sequence is

$$
q=1\;\longrightarrow\;q=0\;\longrightarrow\;q=1.
$$

The first burn starts at the supplied initial time $t_0=0$. The second arc is
a coast. The third arc is the second burn. If the normalized durations are
$\tau_{b1}$, $\tau_c$, and $\tau_{b2}$, then

$$
\tau_f=\tau_{b1}+\tau_c+\tau_{b2}.
$$

The first burn raises the transfer-orbit apoapsis. The second burn raises the
transfer-orbit periapsis to the target radius.

## Explicit One-Burn Formulation

This formulation coasts to a selected burn start and then applies one
continuous powered arc. It solves for the burn start, burn duration, primer
angle, and selected costate values.

The formulation was rejected as the general solution for these reasons:

- It cannot represent a second powered arc after a coast.
- Its roots matched a restricted one-burn direct calculation but violated the
  switching-function sign on the coast and powered intervals.
- The synthetic moderate case used fuel fraction $0.602543$ for the explicit
  one-burn root and $0.117540$ for the accepted two-burn solution.

The formulation remains an offline comparison case.

## Event-Based Single Shooting

This formulation propagates one state-costate initial value problem. It uses
the switching function to detect all control changes. The solver must change
the integration path when a trial parameter changes the number or order of
switches.

The corrected implementation used the actual initial throttle when it computed
$H_0$. It then failed the residual test for all ninety-nine deterministic
mass-costate trials across eleven cases. `root(method="hybr")`,
`root(method="lm")`, and bounded `least_squares` each entered no-thrust or
nonzero-residual branches for most trials.

This formulation remains an event-sequence diagnostic. It is not the
provisional runtime solver.

## Fixed-Sequence Segmented Shooting

The selected formulation fixes the sequence $q=1\rightarrow0\rightarrow1$
during one nonlinear solve. It does not run one optimizer for each burn. It
does not run an optimizer for the coast.

For one parameter vector

$$
z=(\alpha_0,\tau_{b1},\tau_c,\tau_{b2},
\lambda_{\rho,0},\lambda_{\eta,0}),
$$

one residual evaluation performs these integrations in order:

1. Integrate the first powered arc with $q=1$ for $\tau_{b1}$.
2. Require $\Phi_1=0$ at the end of the first burn.
3. Integrate the coast with $q=0$ for $\tau_c$.
4. Require $\Phi_2=0$ at the end of the coast.
5. Integrate the second powered arc with $q=1$ for $\tau_{b2}$.
6. Require the three terminal circular-orbit conditions.
7. Require $H_0=0$ when $\tau_f<\tau_{\max}$, or require
   $\tau_f-\tau_{\max}=0$ when the final-time cap is active.

The endpoint of each integration is the initial state for the next
integration. The state and costate are not independently guessed at the arc
boundaries.

This is segmented single shooting. It is not classical multiple shooting.
Classical multiple shooting would introduce independent initial state-costate
vectors for each arc and matching equations. That formulation has not been
selected or implemented.

## Direct Two-Burn Reference

The direct reference parameterizes two full-thrust arcs with piecewise constant
thrust angles. It solves for the arc durations and angles without using primer
costates. It uses deterministic seeds based on the impulse transfer estimate
and the restricted one-burn result.

The direct reference is an offline check. It is not part of the provisional
runtime path. It can falsify a primer result when it finds a valid trajectory
with lower fuel after mesh refinement.

## Evidence

The fixed-sequence segmented solver matched the direct two-burn reference for
all eleven core and randomized cases and all five edge cases. The largest
relative fuel difference was approximately $8.3\times10^{-6}$. A three-burn
direct search found no lower-fuel result in those cases.

The sequence remains a provisional result for the first-outbound domain. It has
not been validated for an initial coast, more than two powered arcs, singular
throttle intervals, or an unrestricted number of revolutions.

## Falsifiers

- A valid case with a lower-fuel three-or-more-burn direct solution.
- A valid case that requires a coast before the first burn.
- A singular interval with $\Phi=0$ that does not converge to an on/off limit.
- A case where classical multiple shooting changes the result after matching
  residuals are enforced.
