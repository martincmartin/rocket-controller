# Negative Initial Burn Study

## Conclusion

The current fixed on/off/on solver does not allow a negative first burn.
It also does not allow a negative coast or a negative second burn.

The solver converges for `kerbin-coast-first-example`, but it does not return
the requested diagnostic value `tau_b1 <= 0`. It returns

$$
\tau_{b1}=0.000120170297203.
$$

This value is positive. Its physical duration is about $0.0359\ \mathrm{s}$.
The result is therefore almost coast-first, but a test for
`tau_b1 <= 0` does not select the coast-first branch.

The initial values are not close to the final values for the arc durations.
The direct seed contains an initial coast, but the fixed on/off/on seed drops
that coast. The shooting solve then recovers most of the coast duration.

The current solver is not robust for the proposed strict test. A future
implementation should use an explicit zero-first-burn branch, or use a
documented positive tolerance before switching to an off/on solve.

## Solver Identification

The target case is not in the case list of `experiments/multi_stage_primer.py`.
That script accepts `kerbin-stage-data` and `kerbin-first-example` only at
`experiments/multi_stage_primer.py:1026-1035`.

The target case is the single-stage case used by
`experiments/sigmoid_primer.py:90-106`. The fixed on/off/on solver is
`experiments/two_arc_shooting.py`.

Its shooting vector is

$$
z=(\alpha_0,\tau_{b1},\tau_c,\tau_{b2},
\lambda_{\rho,0},\lambda_{m,0}).
$$

The residual integrates a full-thrust arc, a coast, and a full-thrust arc.
It then imposes the three terminal orbit conditions, the two interior switch
conditions, and either $H(0)=0$ or the final-time limit. See
`experiments/two_arc_shooting.py:24-50`.

## Code Inspection

`solve_from_initial` sets these bounds at
`experiments/two_arc_shooting.py:84-93`:

| Parameter | Lower bound | Negative allowed | Exact zero allowed |
| --- | ---: | --- | --- |
| $\tau_{b1}$ | $10^{-8}$ | No | No |
| $\tau_c$ | $0$ | No | Yes |
| $\tau_{b2}$ | $10^{-8}$ | No | No |

The direct two-burn reference uses the same non-negative policy. Its bounds
are at `experiments/two_burn_reference.py:130-145`.

The raw residual does not check the signs of the durations. It passes each
duration to `integrate_primer` at `experiments/two_arc_shooting.py:32-36`.
SciPy can integrate a negative duration in reverse time. This does not make a
negative duration a valid burn schedule. Some negative trials fail during
integration, and the bounded solver never sends them.

The review variant uses the same bounds at
`experiments/sigmoid_primer_review/two_arc_residual_variant.py:97-106`.
The staged solver also uses positive powered-duration lower bounds at
`experiments/multi_stage_primer.py:763-773`.

## Reproduction

The experiment used the repository virtual environment and wrote its JSON
results to `negative-initial-burn-results.json`.

```text
PYTHONPATH=experiments .venv/bin/python \
  tasks/negative-initial-burn/run_negative_initial_burn.py
```

The case uses the same input vectors as `sigmoid_primer.py`:

| Quantity | Value |
| --- | ---: |
| $\rho_0$ | $0.929086774226051$ |
| $u_{r,0}$ | $0.354859194038108$ |
| $u_{t,0}$ | $0.463406947156854$ |
| $\gamma$ | $2.390241108697565$ |
| $\kappa$ | $1.377017172145421$ |
| Maximum burn duration | $0.502705508434977$ |
| Time scale | $298.385431396962\ \mathrm{s}$ |
| Initial time to apoapsis (normalized) | $0.418062970903238$ |
| Initial time to apoapsis (physical) | $124.743899924\ \mathrm{s}$ |

The direct one-burn reference confirms that the useful trajectory starts with
a coast:

| Angle intervals | Coast | Burn | Fuel fraction | Residual norm |
| ---: | ---: | ---: | ---: | ---: |
| 4 | $0.323555945$ | $0.195809934$ | $0.339888974$ | $6.8\times10^{-12}$ |
| 8 | $0.314849813$ | $0.195776651$ | $0.339831202$ | $1.3\times10^{-11}$ |
| 16 | $0.314832534$ | $0.195768751$ | $0.339817487$ | $6.5\times10^{-13}$ |
| 32 | $0.314793924$ | $0.195766803$ | $0.339814106$ | $2.4\times10^{-13}$ |

The direct reference uses a variable coast before one powered arc. The
32-interval result has a coast of about $93.93\ \mathrm{s}$ and a burn of
about $58.41\ \mathrm{s}$.

## Initial Values

The fixed shooting path first runs the deterministic direct two-burn
multistart. The selected direct result was

| Direct parameter | Value |
| --- | ---: |
| Initial coast before first burn | $0.314560623043$ |
| First burn | $0.107136055550$ |
| Middle coast | $1.27\times10^{-18}$ |
| Second burn | $0.088632625825$ |
| Fuel fraction | $0.339817367193$ |
| Terminal residual norm | $2.31\times10^{-12}$ |

The direct vector has an initial coast parameter. The fixed shooter does not
have that parameter. `two_arc_shooting.seed` reads the first angle from the
direct vector and returns the direct first burn, middle coast, and second burn
at `experiments/two_arc_shooting.py:53-75`. It therefore drops the initial
coast.

The resulting fixed-shooter initial vector was

$$
\begin{aligned}
\alpha_0 &= 1.699390947298,\\
\tau_{b1} &= 0.107136055550,\\
\tau_c &= 1.27\times10^{-18},\\
\tau_{b2} &= 0.088632625825,\\
\lambda_{\rho,0} &= -0.159532320120,\\
\lambda_{m,0} &= -1.377017172145.
\end{aligned}
$$

The initial residual norm was $0.2184525781$. The initial powered durations
correspond to about $31.97\ \mathrm{s}$ and $26.45\ \mathrm{s}$, with no
middle coast.

These values are not close to the final arc partition. The mass costate is
close, but the first costate, primer angle, and all three arc durations move
substantially.

## Fixed Shooting Result

The solver reported success with the `gtol` termination condition. The final
vector was

$$
\begin{aligned}
\alpha_0 &= 2.169539300250,\\
\tau_{b1} &= 0.000120170297203,\\
\tau_c &= 0.314079433877,\\
\tau_{b2} &= 0.195645801438,\\
\lambda_{\rho,0} &= 1.060920041071,\\
\lambda_{m,0} &= -1.376999791448.
\end{aligned}
$$

The result has these diagnostics:

| Quantity | Value |
| --- | ---: |
| Success | `true` |
| Final-time condition | Free final time |
| Total time | $0.509845405613$ |
| Total time | $152.130441\ \mathrm{s}$ |
| First burn | $0.000120170297203$ ($0.0359\ \mathrm{s}$) |
| Coast | $0.314079433877$ ($93.7167\ \mathrm{s}$) |
| Second burn | $0.195645801438$ ($58.3779\ \mathrm{s}$) |
| Fuel fraction | $0.339812663772$ |
| Residual norm | $5.66\times10^{-14}$ |

The result is a valid root of the fixed residual. It is not a zero-first-burn
root because the residual assumes that the first arc is powered and imposes a
switch condition at its end.

## Comparison With The Coast-First Oracle

The cataloged coast-first solution uses

$$
\begin{aligned}
\alpha_0 &= 2.180458606526,\\
\tau_{b1} &= 0,\\
\tau_c &= 0.314482092435,\\
\tau_{b2} &= 0.195766154962,\\
\lambda_{\rho,0} &= 1.087412012183,\\
\lambda_{m,0} &= -1.370608266588.
\end{aligned}
$$

The fixed shooter result is close to this oracle after it has converged. The
largest duration issue is the positive first burn of $0.00012017$.

| Parameter | Fixed shooter | Coast-first oracle | Difference |
| --- | ---: | ---: | ---: |
| $\alpha_0$ | $2.169539300250$ | $2.180458606526$ | $-0.010919306276$ |
| $\tau_{b1}$ | $0.000120170297$ | $0$ | $+0.000120170297$ |
| $\tau_c$ | $0.314079433877$ | $0.314482092435$ | $-0.000402658558$ |
| $\tau_{b2}$ | $0.195645801438$ | $0.195766154962$ | $-0.000120353524$ |
| $\lambda_{\rho,0}$ | $1.060920041071$ | $1.087412012183$ | $-0.026491971111$ |
| $\lambda_{m,0}$ | $-1.376999791448$ | $-1.370608266588$ | $-0.006391524861$ |

The oracle vector reaches the terminal orbit conditions under the segmented
propagation. When passed to the current on/off/on residual, its terminal
conditions and coast switch are near zero, but the first switch residual is
$0.004654194361$ and the $H(0)$ residual is $-0.011124646689$.
The combined residual norm is $0.01205899204$.

This occurs because the current residual evaluates $H(0)$ with full thrust and
expects a powered first arc. The coast-first oracle requires an off/on residual
with a coast at the initial epoch. Setting only $\tau_{b1}=0$ does not change
those residual equations.

## Signed-Duration Diagnostic

As a diagnostic, the least-squares lower bounds were manually changed to $-1$
for all three durations. This was not a source-code change to the solver.

Several starts still converged to the positive root reported above. Other
starts converged to algebraic roots with negative durations. One example was

$$
(\tau_{b1},\tau_c,\tau_{b2})
=(-0.008512128710,\;0.302346796275,\;0.203400762909),
$$

with residual norm below $8\times10^{-16}$. Other roots had negative second
burn durations. These roots use reverse-time integration and are not physical
burn schedules.

This diagnostic has two implications:

- The bounds are the only sign protection in the fixed solver.
- Removing the bounds is not a safe way to expose an optional first burn.

## Options Considered

### Use `tau_b1 <= 0` with the current solver

This option needs no code change. It is not viable because the lower bound is
$10^{-8}$ and the observed result is $0.00012017$.

### Widen the duration bounds

This option would allow a numerical sign test. It is not viable by itself.
Negative powered durations produce reverse-time trajectories and unphysical
roots. The raw residual also has no duration-sign validation.

### Add an explicit off/on branch

This is the recommended option. Set $\tau_{b1}=0$, remove the first powered
arc, propagate the costate through the initial coast, and use the off/on
switching and terminal residuals. This matches the cataloged coast-first
oracle. It requires a separate residual and an explicit branch decision.

### Use a positive classification tolerance

This is a small interim option. A rule such as
`tau_b1 <= tau_b1_tolerance` would classify the observed root as coast-first.
It does not repair the on/off/on residual, so the threshold must be treated as
a heuristic. The tolerance must be tied to integration accuracy and physical
time, not selected only from this one case.

## Recommendation

Do not use `tau_b1 <= 0` as the current branch detector. The solver cannot
return a negative value or exact zero, and its zero-first-burn oracle is not a
root of its current residual.

Use the current result as a diagnostic only. Add an explicit off/on solve for
the production branch. Until that branch exists, a documented small-duration
threshold can identify this case, but it must not be treated as proof that the
on/off/on equations solved the coast-first problem.
