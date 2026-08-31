# Staging Initial Implementation 

Staging (in practice) has two different, separable parts: an instantaneous change in $T_{max}, Ve$ and mass, and a 2 second coast.  We can implement this as the coast followed by the change.

For the coast, H changes during the coast, but H just after the coast (engines burning again) is exactly the same as H just before the coast (engines burning, just about to shut off).

NOTE: If you make mass implicit, there's no mass costate to solve for and the instantaneous staging proper is even easier!

## Changes (Staging Proper, Instantaneous)

Hamiltonian is continuous, as are the position and velocity costates, so the mass costate must absorb all the changes.  Although the Hamiltonian is zero on the optimal route, it is non-zero during many evaluations during optimization, so we want to conserve H during staging proper, not solve from H=0, as we're using H as a residual.

## Coast

Everything evolves as always, following the equations of motion/ODE.  In particular, even though the thrust is discontinuous, the position, velocity and mass are all continuous, and so are the costates for the position, velocity and mass.  The mass costate doesn't change during the coast.  "According to the Weierstrass-Erdmann corner conditions, as long as the states (position, velocity, mass) do not experience a discontinuous jump (e.g., no staging event) and there are no interior point state constraints, the costates cannot jump."  I don't know what that means either.  The Hamiltonian is discontinuous though, and the Hamiltonian after the coast is not the same as the one before.  Since H=0 at the final time (because final time is free), this means H may be non-zero at the start.  So using H=0 to choose the initial $\lambda_\rho$ becomes a sketchy approximation.

## Implementation

In our current setup, we break the integration into a bunch of separate calls to `solve_ivp`: an optional initial burn, then a coast, then a second burn.  Before calling `solve_ivp`, we know the length of each burn.  So, we can compute whether or not we'll run out of fuel and stage during that burn.  If so, that burn, which is currently a single call to `solve_ivp`, becomes three: finish the first stage, a 2 second coast, and staring the next stage.  In general, there could be multiple stagings during any burn.  Interpret the burn time as the time spent actually burning, not including the coast.  That is, if the root finder parameter says $\tau_{b1} = 5$ seconds, and the first stage runs out of fuel after 4 seconds, then we simulate a 4 second burn followed by the 2 second coast/staging followed by a 1 second burn of that new stage.