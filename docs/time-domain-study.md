# Single-Stage Final-Time Domain

## Current Cap

The current experiments use a numerical final-time cap derived from the
initial osculating orbit. If $t_{\mathrm{apo},0}$ is the time to the next
initial apoapsis and $T_0$ is the initial orbital period, then

$$
t_{\max}=t_{\mathrm{apo},0}+\frac{3}{4}T_0,
$$

where

$$
T_0=2\pi\sqrt{\frac{a_0^3}{\mu}}.
$$

In normalized units this is

$$
\tau_{\max}=\tau_{\mathrm{apo},0}+\frac{3\pi}{2}a_0^{3/2}.
$$

This is a search-domain limit. It is not derived from KSP, it is not a physical
vehicle constraint, and it is not an optimal-control terminal condition.

## How It Is Used

The cap currently appears in three places:

- The direct two-burn reference constrains the sum of the first burn, coast, and
  second burn durations to be no greater than $t_{\max}$.
- The fixed-sequence primer solve uses $H(t_f)=0$ when the solution is interior
  and uses $t_f-t_{\max}=0$ when the cap is active.
- Event-based shooting uses the cap as an upper bound on the final integration
  time.

The two final-time conditions must not be imposed simultaneously. If the cap is
active, the trajectory is optimal only within the truncated time domain.

## Does An Active Cap Prove More Fuel Is Available?

No. An active cap proves only that the constrained solution lies on the edge of
the allowed domain. It is a warning that the unconstrained problem has not been
resolved. Extending the cap can produce:

- A lower-fuel trajectory at a later time.
- The same fuel at an earlier interior time, showing that the previous active
  result was a local or numerical artifact.
- No improvement, showing that the cap was harmless for that case.
- A different event sequence or a physically invalid trajectory.

The only reliable test is cap continuation: increase the cap, re-solve, and
compare fuel, final time, event sequence, and terminal residual.

## Previous Cap-Continuation Evidence

The following results were obtained with the previous coefficient $1/4$. They
are historical evidence. They do not validate the current coefficient $3/4$.
The `random-00` case was active at its base cap. Increasing the cap produced

| Cap offset | Cap $\tau_{\max}$ | Fuel fraction | Final $\tau_f$ | Active |
| ---: | ---: | ---: | ---: | --- |
| $0$ | $1.604429690$ | $0.143637356$ | $1.604429690$ | yes |
| $0.25$ | $1.854429690$ | $0.138212251$ | $1.854429690$ | yes |
| $0.50$ | $2.104429690$ | $0.138122630$ | $1.894728307$ | no |
| $1.00$ | $2.604429690$ | $0.138122630$ | $1.894728307$ | no |

This result shows that an active cap can change the fuel result. Increasing the
cap lowered fuel by approximately $3.6\%$ before the solution became interior.

The cap-continuation experiment must be rerun after any change to the cap
coefficient.

## Supplied Kerbin Cases

With the current coefficient, neither supplied Kerbin case reaches the cap.

| Case | $t_f$ (normalized) | $t_{\max}$ (normalized) | $t_f$ (s) | $t_{\max}$ (s) | Active |
| --- | ---: | ---: | ---: | ---: | --- |
| First example | $0.522074908$ | $2.054721218$ | $155.780$ | $613.099$ | no |
| Second example | $0.648432961$ | $1.853230303$ | $193.483$ | $552.977$ | no |

The current analytic-seed runs also remain interior for both cases. The cap is
therefore not active in either supplied Kerbin solution.

## Do We Need A Cap?

The mathematical free-final-time problem does not require this cap. The current
research configuration retains it as a numerical domain boundary:

- Without a cap, the solver can enter extra-revolution or long-coast branches.
- A fixed cap can hide a lower-fuel later solution, as the previous
  `random-00` sweep demonstrated.
- A cap-active result is domain-limited. It is not an unconstrained optimum.

For each cap-active case, increase the cap and solve again. Record the fuel,
final time, and event sequence. If the fuel decreases, the base-cap result is
not the free-final-time result. If the solution becomes interior and the
reported values stop changing, the cap does not affect that case.

The current fixed cap may remain a bounded fallback for numerical safety, but an
active-cap result must be labeled domain-limited rather than globally optimal.

## Falsifiers

- A cap-continuation run that keeps finding lower fuel as the cap grows falsifies
  the current cap as a valid bound for that case.
- A physically valid later-revolution solution with lower fuel falsifies the
  first-outbound branch restriction for an unrestricted mission objective.
- A stable interior solution over a broad cap range would support using the
  current cap as a safe numerical bound for that regime.
- A case whose event sequence changes as the cap grows requires a wider
  multi-arc formulation before the single-stage result can be accepted.
