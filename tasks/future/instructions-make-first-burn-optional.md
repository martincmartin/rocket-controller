# Make First Burn Optional

According to Claude, the only burn patterns we need to achieve circular orbit under about 7 Mm (6,893 km) are subsets of **Burn₁ → Coast → Burn₂**, including **Coast → Burn₂** and a single **Burn** as a degenerate case of coast time going to zero.

The experiments involving a pure Lawden primer vector setup, where we don't specify a burn pattern but rather let the switching function do it, showed it was impractical.  With only three examples taken from KSP, and not chosen to be particularly difficult but rather typical, one of them (`kerbin-example`) had a switching function that was very small (|S| ≤ 2.3e-3 everywhere, coast |S| = 5.75e-5).  So you need advanced stuff like adapting sigmoid with adaptive $\epsilon$, and even then planning takes 140 seconds lol.  See  review-sigmoid-primer.md .

So the existing (in experiments) burn/cost/burn aka on/off/on formulation, with explicit $\tau_{b1}, \tau_c, \tau_{b2}$ parameters is the way to go.  The current script assumes we always start with a burn, we need to make that optional and `kerbin-coast-first-example` can be used to debug it. 

A big question is how we decide whether the first burn is optional, e.g. do we start planning on/off/on and see if the first burn time becomes zero?  There's a residual for the switching function to be zero at the end of the first burn, that might mess everything up if the optimal solution is coast/burn.