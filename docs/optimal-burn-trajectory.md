# Why The First Burn Is Not Along Velocity

## Angle Convention

The primer/thrust angle $\alpha$ is measured from the local radial direction
toward the local tangential direction. Therefore

$$
\alpha=\frac{\pi}{2}
$$

means pure tangential thrust. It does **not** generally mean thrust along the
current velocity.

The current velocity direction is

$$
\psi=\operatorname{atan2}(v_t,v_r)
 =\operatorname{atan2}(u_t,u_r).
$$

Only at an apsis, where $v_r=0$, does $\psi=\pi/2$ for prograde motion.

## What The First Burn Does

The fixed two-arc trajectory is not one burn that directly circularizes the
current orbit. It is

$$
\text{first burn}\;\rightarrow\;\text{coast}\;\rightarrow\;
\text{second burn}.
$$

The first burn primarily shapes a transfer orbit whose apoapsis is near the
target radius. The second burn, performed near that new apoapsis, raises the
periapsis to the target radius.

Thus the first burn is allowed to trade instantaneous energy efficiency for a
better future state. The primer direction is selected from the sensitivity of
the complete endpoint problem, not merely from the instantaneous velocity
vector.

## Velocity-Parallel And Perpendicular Components

Let $a_T=T/m$ be the thrust acceleration magnitude. The velocity-parallel and
velocity-perpendicular components are

$$
a_\parallel=a_T\cos(\alpha-\psi),
\qquad
a_\perp=a_T\sin(\alpha-\psi).
$$

The instantaneous specific-energy rate due to thrust is

$$
\dot\epsilon=\mathbf v\cdot\mathbf a_T
 =\lVert\mathbf v\rVert a_\parallel.
$$

Therefore $a_\perp$ does not directly change orbital energy at that instant.
It is not nevertheless useless. It rotates the velocity vector and changes
the angular momentum

$$
h=rv_t,
\qquad
\dot h=r a_t=r a_T\sin\alpha.
$$

It also changes radial velocity, radius, orbital phase, eccentricity, and the
future locations of the apsides. Those state changes alter the energy and
burn cost that can be achieved by the later burn.

The radial/tangential components should not be confused with the
velocity-parallel/perpendicular components. When $v_r\ne0$, a radial thrust
component can contribute directly to energy, and a tangential thrust component
can be substantially perpendicular to the current velocity.

## Kerbin Second Example

For the supplied state used as `kerbin-example`, the initial angles are:

| Quantity | Value |
| --- | ---: |
| Current velocity angle $\psi$ | $38.37^\circ$ |
| Initial optimal primer/thrust angle $\alpha_0$ | $77.77^\circ$ |
| Difference $\alpha_0-\psi$ | $39.41^\circ$ |
| Velocity-parallel alignment | $0.7726$ |
| Velocity-perpendicular alignment | $0.6348$ |

The initial thrust has a radial component of $0.212$ and a tangential component
of $0.977$ in the local radial/tangential basis:

$$
\cos(77.77^\circ)=0.212,
\qquad
\sin(77.77^\circ)=0.977.
$$

The $39.41^\circ$ difference occurs because the initial state has
$u_r=0.252573$ and $u_t=0.199946$. Pointing along that
velocity would add a large radial component that would later need to be
removed. The selected thrust instead increases angular momentum while using
the existing outward radial motion to carry the vehicle to a higher orbit.

The first burn lasts approximately $77.25\ \textrm{s}$, while the original
unpowered apoapsis would occur after approximately $70.90\ \textrm{s}$. At
the end of the first burn, the normalized state is approximately

$$
(\rho,u_r,u_t,\eta)
=(0.977823,0.121668,0.834271,0.618822),
$$

and its osculating apoapsis is

$$
\rho_a=0.999958.
$$

The first burn has therefore created the transfer orbit whose apoapsis is
$\rho_a=0.999958$. It has not attempted to make the vehicle circular at that
point.

During the first burn, the normalized specific-energy increase was
$0.353435$, and the angular-momentum increase was $0.629356$. The integrated
velocity-perpendicular acceleration was $0.165058$ in normalized velocity
units. That perpendicular component changed the orbit geometry and angular
momentum. It did not directly change specific orbital energy.

## Kerbin First Example

For `kerbin-first-example`:

| Quantity | Value |
| --- | ---: |
| Current velocity angle $\psi$ | $40.16^\circ$ |
| Initial optimal primer/thrust angle $\alpha_0$ | $74.60^\circ$ |
| Difference $\alpha_0-\psi$ | $34.44^\circ$ |
| Velocity-parallel alignment | $0.8247$ |
| Velocity-perpendicular alignment | $0.5655$ |

The first burn lasts approximately $18.87\ \textrm{s}$, whereas the original
unpowered apoapsis would occur after approximately $103.31\ \textrm{s}$. At
first-burn end, the osculating apoapsis is already

$$
\rho_a=0.997964.
$$

This is a transfer-orbit shaping burn before the original apoapsis. The second
burn is longer than in the second Kerbin example because
the first burn has produced a different transfer orbit.

## Why Waiting For Apoapsis Is Worse Here

The supplied Kerbin states have substantial outward radial velocity but low
tangential velocity. The current osculating apoapsis is therefore reached with
very little tangential speed. A pure tangential impulse at that apoapsis would
need to supply the missing angular momentum after the radial velocity has
fallen to zero.

For the second Kerbin example, the vis-viva estimate for a pure tangential
impulse at the original apoapsis is

$$
\Delta v_{\mathrm{apo}}=0.8356v_\star,
$$

compared with $0.5890v_\star$ for a tangential impulse at the initial point
that raises the apoapsis directly to the target. The corresponding ideal burn
durations are approximately $0.3090t_\star$ and $0.2364t_\star$.

For the first Kerbin example, the corresponding impulse estimates are

$$
\Delta v_{\mathrm{apo}}=0.7402v_\star,
\qquad
\Delta v_{t_0}=0.2935v_\star.
$$

The initial radial velocity is already part of the vehicle's orbital energy.
Burning before apoapsis can exploit that existing motion while adding angular
momentum. Waiting removes that radial motion through gravity and then requires
a larger tangential correction.

The finite-thrust optimum is not exactly the instantaneous-impulse estimate,
because the burn itself changes radius, velocity, mass, and the subsequent
coast. Nevertheless, the estimates explain both why the burn starts early and
why its direction is close to tangential rather than aligned with the current
velocity.

## Conclusion

The statement that the optimal initial angle is near $76^\circ$ should be read
as an angle from the radial axis. It is about $14^\circ$ to $15^\circ$ away
from the local tangential direction, not a thrust direction that is $76^\circ$
away from local horizontal.

The initial burn is a transfer-orbit-shaping burn. Its velocity-perpendicular
component changes angular momentum and future geometry, enabling the later
apoapsis circularization burn. The current numerical evidence supports this
interpretation for both supplied Kerbin trajectories, but the conclusion
remains a property of the complete finite-thrust endpoint optimization, not a
universal instantaneous steering rule.
