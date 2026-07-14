# Sim.py
- Save a dimension in search by doing a 1D search for best burn time, that minimizes some error?
  - Like delta semi-major axis divided by target semi-major axis (= target radius) + absolute eccentricity.
  
- Thrust profile?
  - Start with just constant % thrust per stage, or linearly interpolate between input thrust values.
  
- General clean up and review.

# Gravity_turn.py

- Recompute again after burn starts, maybe lowering target altitude threshold a little.
