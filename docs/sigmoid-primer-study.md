# Sigmoid Switching Single-Stage Study

## Conclusion

The supplied third Kerbin state should coast before applying thrust. A direct
single-stage solve gives a coast of approximately $93.930\ \textrm{s}$ followed
by one powered arc of approximately $58.414\ \textrm{s}$. The direct two-arc
check drives the first powered duration to its lower bound, so this is not an
on/off/on trajectory with a useful first burn.

The sigmoid indirect solve can reach the terminal equations for all three
Kerbin cases with the continuation schedule

$$
[1,\;0.3,\;0.1,\;0.03,\;0.01,\;0.003,\;0.001,\;0.0003,\;0.0001].
$$

That is not enough to make it a replacement for the fixed-sequence solver. At
large $\varepsilon$ the control is substantially fractional. At small
$\varepsilon$ the solver becomes sensitive and can satisfy the endpoint
conditions while reporting the wrong final switching time or a long
near-singular interval. The new coast-first case is the cleanest success; the
older `kerbin-example` is the clearest failure of event reconstruction.

The implementation is in
[`experiments/sigmoid_primer.py`](../experiments/sigmoid_primer.py). It is an
offline experiment and does not modify `sim.py`, `sim_experiment.py`, or the
flight controller.

## Model And Scope

The experiment uses the normalized polar state and primer equations in
[Primer Equations](primer-equations.md). The target is

$$
(\rho_f,u_{r,f},u_{t,f})=(1,0,1).
$$

The three cases are:

| Case | Description |
| --- | --- |
| `kerbin-example` | Existing low-tangential-speed Kerbin case |
| `kerbin-first-example` | Existing first supplied Kerbin case |
| `kerbin-coast-first-example` | The state supplied for this study |

For consistency with the existing single-stage research, the first-stage
engine is given the common artificial single-stage limit of $150\ \textrm{s}$.
The $28.2756668\ \textrm{s}$ limit in the `sim_experiment.py` snippet is the
remaining physical segment capacity, not a comparable single-stage study
limit. With that literal limit the engine has only about $564\ \textrm{m/s}$
of ideal rocket-equation delta-v, while the new state needs about
$1297\ \textrm{m/s}$ of tangential correction at its apoapsis. It is therefore
propellant-infeasible for the exact single-stage target and was not mixed into
the comparison below.

The common Kerbin normalization is

$$
r_\star=680000\ \textrm{m},\qquad
v_\star=2278.931638238564\ \textrm{m/s},\qquad
t_\star=298.3854313969623\ \textrm{s}.
$$

## Direct Result For The New Case

The physical input was

$$
\begin{aligned}
m_0&=11777.2275390625\ \textrm{kg},\\
T&=215000\ \textrm{N},\\
v_e&=3138.128\ \textrm{m/s},\\
\mathbf r_0&=(433284.5917063,-704.8282711,-459791.995176)\ \textrm{m},\\
\mathbf v_0&=(1323.15860984,11.49193645,135.66254872)\ \textrm{m/s}.
\end{aligned}
$$

The computed local normalized state is

$$
(\rho_0,u_{r,0},u_{t,0},\eta_0)
=(0.929086774226051,\;0.354859194038108,\;0.463406947156854,\;1).
$$

The corresponding local physical state is $r_0=631779.006\ \textrm{m}$,
$v_{r,0}=808.700\ \textrm{m/s}$, and $v_{t,0}=1056.073\ \textrm{m/s}$. The
normalized stage parameters are

$$
\gamma=2.390241108697565,\qquad
\kappa=1.377017172145421.
$$

The supplied time to apoapsis, $124.74389992322136\ \textrm{s}$, agrees with
the independent propagation result $124.743899924\ \textrm{s}$. The initial
osculating apoapsis is $\rho_a=1.001629074$, or approximately
$81.108\ \textrm{km}$ altitude. It is already just above the target, but its apoapsis
tangential speed is only about $980\ \textrm{m/s}$.

The direct one-burn transcription uses a variable coast duration followed by
piecewise-constant full-thrust angles. Its mesh-convergence results are:

| Angle intervals | Coast $\tau$ | Burn $\tau$ | Fuel fraction | Terminal residual |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.323555945 | 0.195809934 | 0.339888974 | $6.8\times10^{-12}$ |
| 8 | 0.314849813 | 0.195776651 | 0.339831202 | $1.3\times10^{-11}$ |
| 16 | 0.314832534 | 0.195768751 | 0.339817487 | $6.5\times10^{-13}$ |
| 32 | 0.314793924 | 0.195766803 | 0.339814106 | $2.4\times10^{-13}$ |

The $32$-interval result corresponds to:

| Quantity | Value |
| --- | ---: |
| Coast before ignition | $93.930\ \textrm{s}$ |
| Burn duration | $58.414\ \textrm{s}$ |
| Burn start relative to initial apoapsis | $30.814\ \textrm{s}$ before apoapsis |
| Burn end from the initial epoch | $152.344\ \textrm{s}$ |
| Fuel fraction | $0.339814106$ |

Thus the optimum does not wait all the way to apoapsis. It coasts first, then
starts the finite-duration burn while still outbound. The burn ends about
$27.6\ \textrm{s}$ after the original apoapsis.

Two direct checks distinguish this from a numerical preference for a tiny first
burn:

| Direct parameterization | Result |
| --- | ---: |
| One burn, free initial coast | $\tau_b=0.195766803$, fuel $0.339814106$ |
| One burn, initial coast fixed to zero | $\tau_b=0.246961171$, fuel $0.428677837$ |
| Two powered arcs, $8$ angle intervals per arc | first burn $10^{-8}$, fuel $0.339831204$ |

The zero-coast alternative is feasible, but consumes about $26.2\%$ more fuel
than the free-coast result. In the two-arc solve the initial coast is only
$1.99\times10^{-6}$ normalized time and the first burn is at its
$10^{-8}$ lower bound. The useful middle coast is $0.314780448$ and the second
burn is $0.195776643$. This confirms, within the direct meshes, that the new
case should start with a coast and has no useful first burn.

## Sigmoid Indirect Formulation

The hard switching law in the existing event-based experiment is

$$
q=\begin{cases}1,&S>0,\\0,&S<0,\end{cases}
\qquad
S=\frac{P}{\eta}+\frac{\lambda_\eta}{\kappa}.
$$

The new experiment substitutes

$$
q_\varepsilon=\frac{1}{1+\exp(-S/\varepsilon)}
$$

into the state and costate right-hand sides. The code uses SciPy's `expit`,
which is the same function evaluated without overflow for large negative or
positive arguments. The primer direction remains

$$
\cos\alpha=\frac{p_r}{P},\qquad
\sin\alpha=\frac{p_t}{P}.
$$

The indirect shooting vector is only

$$
z=(\alpha_0,\tau_f,\lambda_{\rho,0},\lambda_{\eta,0}).
$$

There are no burn or coast durations in the propagation function. One smooth
state-costate integration runs from $0$ to $\tau_f$. The bounded
`least_squares` solve enforces the three terminal orbit conditions and a
terminal switching condition.

At the exact circular target, the non-thrust part of the Hamiltonian is zero.
For any finite $\varepsilon$, $q_\varepsilon$ is strictly positive, so

$$
H(t_f)=-\gamma q_\varepsilon(t_f)S(t_f)=0
$$

is equivalent to $S(t_f)=0$. The implementation uses $S(t_f)$ as the fourth
residual instead of $q_\varepsilon S$. This avoids making the free-time
residual exponentially small when the terminal control is close to zero. The
initial guesses are analytic impulse estimates; they are not mode schedules.

The continuation procedure solves at $\varepsilon=1$ first and passes the
four-component solution to the next smaller value.

## Continuation Results

The following table compares the endpoints of the continuation with the
existing fixed-sequence references and the new direct reference. The smooth
fuel fraction is computed from the integrated mass loss, not from a guessed
burn duration.

| Case | Reference fuel | Reference $\tau_f$ | Sigmoid fuel at $10^{-4}$ | Sigmoid $\tau_f$ | Final residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kerbin-example` | 0.458673471 | 0.648432961 | 0.458675884 | 0.714904750 | $1.5\times10^{-10}$ |
| `kerbin-first-example` | 0.417870055 | 0.522074908 | 0.417875930 | 0.535981844 | $1.9\times10^{-12}$ |
| `kerbin-coast-first-example` | 0.339814106 | 0.510560727 | 0.339813732 | 0.510605809 | $1.5\times10^{-13}$ |

The reference values for the first two cases are the fixed on/off/on primer
solutions in [Single-Stage Cases](single-stage-test-cases.md). The new
reference final time is the $32$-interval direct coast plus burn.

At $\varepsilon=1$, all three controls are strongly relaxed:

| Case | $q(0)$ | $q$ range | $\tau_f$ | Fuel fraction | Hamiltonian drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kerbin-example` | 0.554 | $(0.497,0.554)$ | 0.612743 | 0.464036 | 0.0118 |
| `kerbin-first-example` | 0.476 | $(0.476,0.500)$ | 0.552265 | 0.418931 | 0.0025 |
| `kerbin-coast-first-example` | 0.418 | $(0.418,0.500)$ | 0.444244 | 0.350150 | 0.0321 |

These are endpoint roots of a relaxed problem, not interpretable bang-bang
trajectories. The small-$\varepsilon$ endpoint diagnostics are:

| Case | Interior $S=0$ crossings | $q$ range at $10^{-4}$ | Effective burn $\tau$ | Hamiltonian drift |
| --- | --- | ---: | ---: | ---: |
| `kerbin-example` | 0.2533, then terminal $\tau_f=0.7149$ | $(0.051,1.000)$ | 0.311549 | $1.4\times10^{-4}$ |
| `kerbin-first-example` | 0.0604, 0.3134, then terminal $\tau_f=0.5360$ | $(0.0024,1.000)$ | 0.266901 | $1.5\times10^{-4}$ |
| `kerbin-coast-first-example` | 0.3124, then terminal $\tau_f=0.5106$ | $(1.5\times10^{-22},1.000)$ | 0.195767 | $9.5\times10^{-5}$ |

The terminal entry in each crossing column is the imposed terminal $S=0$ root;
because the sigmoid has $q=0.5$ at $S=0$, it is not itself a hard throttle
switch.

### `kerbin-example`

The fixed reference burns from $0$ to $0.258910$, coasts to $0.595795$, and
burns again until $0.648433$. The sigmoid endpoint has one clear interior
positive-to-negative crossing at $0.2533$, but it does not produce a distinct
positive second-burn interval. Instead, $S$ remains close to zero and the
fractional throttle supplies the required integrated fuel while the reported
final time drifts to $0.714905$. This is a terminally accurate but
event-structure-inaccurate solution.

### `kerbin-first-example`

The endpoint has a recognizable on/off/on pattern. Its first two interior
crossings are close to the hard reference's first-burn end and second-burn
start. The final time is still about $4.15\ \textrm{s}$ late, and the minimum
throttle in the transition layer is not exactly zero. This is usable as an
approximate continuation result, but it is not as accurate as the explicit
event solve.

### `kerbin-coast-first-example`

This case is the strongest result. At $10^{-4}$ the effective burn is
$0.195766646$, within $2\times10^{-7}$ normalized time of the direct burn.
The interior crossing is $0.312384$, close to the direct burn start
$0.314794$, and the terminal time is within $0.014\ \textrm{s}$ of the direct
reference in physical units. The initial throttle is still $0.0878$ rather
than exactly zero because the sigmoid never turns completely off, but it
quickly falls below numerical significance before the burn transition.

## Problems Encountered

### Relaxation does not preserve the event sequence

At $\varepsilon=1$, $q$ is close to one half for the whole horizon in these
cases. The first solve therefore does not identify the desired arcs. The
continuation can move toward a nearby endpoint root rather than toward the
desired hard-switch root. Fuel converges more reliably than individual switch
times.

### Small epsilon makes shooting stiff

The final nonlinear solve required 438 function evaluations for
`kerbin-example`, compared with 32 for `kerbin-first-example` and 46 for the
new case. The steep sigmoid creates a narrow sensitivity layer around each
switch. Smaller epsilon values can require still more ODE steps and make the
reported zero crossings sensitive to integration tolerances.

### A nearly zero throttle hides final-time error

The hard Hamiltonian residual contains the factor $qS$. Once $S$ is negative
enough, $q$ is tiny and the residual gives almost no information about where a
final burn ended. The direct event sequence can therefore be replaced by a
trajectory that reaches the target and then coasts with an exponentially small
throttle. Using $S(t_f)$ rather than $qS(t_f)$ fixes the conditioning of this
one residual, but it does not force an interior second-burn interval. The
`kerbin-example` result demonstrates the remaining ambiguity.

### The substituted Hamiltonian is not conserved at finite epsilon

For diagnostics the experiment evaluates

$$
H_{\mathrm{diag}}=H_0-\gamma q_\varepsilon S.
$$

The state and costate equations were obtained by substituting the sigmoid into
the bang-bang equations. At finite $\varepsilon$, this diagnostic Hamiltonian
has an observable drift: as large as $0.05$ in intermediate continuation
results and about $10^{-4}$ at $10^{-4}$. The drift tends to zero as the
sigmoid sharpens, but it means the finite-$\varepsilon$ trajectory is a
heuristic continuation, not an exact solution of the original nonsmooth PMP.

A variational smoothing would replace the switching term by a soft-plus
Hamiltonian such as

$$
H_s=H_0-\gamma\varepsilon\log(1+\exp(S/\varepsilon)).
$$

That requires deriving the corresponding costates and choosing the associated
running-cost normalization. The unshifted expression also has a terminal
problem here: at the exact circular target $H_0=0$, while the soft-plus term is
strictly positive for every finite $\varepsilon$. A direct $H_s(t_f)=0$ solve
therefore stalled at the time bound with nonzero terminal residuals. That
variant was not used for the reported results.

### Solver status is not enough

SciPy reports successful least-squares termination even when a trial has the
wrong arc pattern or a weakly constrained final time. The experiment accepts a
result based on terminal residual and records mass, switching-function range,
effective burn, and Hamiltonian drift for separate validation. The direct
transcription remains necessary as an offline reference.

## Assessment

The sigmoid continuation is useful for exploring the basin of the indirect
problem and it handles an initial coast without adding a coast-duration
variable. It solves the terminal equations for all three tested Kerbin cases,
and it recovers the new coast-first case particularly well.

It does not yet replace the explicit on/off/on formulation. The continuation
trades discontinuous event sensitivity for stiffness, relaxed controls, and
weakly constrained final-time/event locations. A production replacement would
need at least adaptive epsilon step control, multiple shooting or variational
sensitivities, and an explicit validation that the limiting $S$ signs produce
the intended sequence. For now the fixed-sequence solver remains the more
reliable method for the two cases known to require an initial burn, while the
sigmoid result provides a candidate path for adding an initial-coast branch.
