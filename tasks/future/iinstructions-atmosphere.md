# Atmosphere

This document isn't actually an instruction to an coding agent, it's more some brainstorming thoughts, would need to be fleshed out more.

This is tricky, I think we abandon the Lawden stuff here.  Things like the coefficient of drag and cross sectional area depend on what direction you're facing, and while we'll be close to prograde, we can be off by 5 or 10 degrees or more.  For a long thin rocket that's probably quite significant.

## Option 1: Just simulate in KSP

This is what I'm doing now, but it's flaky and slow, sometimes the state doesn't reset properly, and it leaks memory.  But implementation is easy.

## Option 2: Create a simulator

### Atmosphere

Kerbin's [atmosphere](https://wiki.kerbalspaceprogram.com/wiki/Kerbin#Atmosphere) seems simple enough to model.  The pressure is a globally constant function of altitude and independent of temperature.  "Temperature-altitude profile is not globally constant, therefore neither is the density-altitude profile, however variance is slight."

"Kerbin's "base" temperature and atmospheric pressure can be very closely approximated using the [equations of the USSA](http://www.braeunig.us/space/atmmodel.htm#table4), where Kerbin's geometric altitude, z, is converted to Earth's geopotential altitude, h, using the equation:

$$h = \dfrac{7963.75 z}{6371 + 1.25 z}$$

where z and h are in km, not meters.

### Drag

The formula for aerodynamic drag is:

$$F_D = \frac{1}{2}\rho v^2 A C_d$$

$\rho$ is air density, you can get that from pressure and temperature using the universal gas law $P=\rho RT$.

Have to look in game how the $AC_d$ factor changes with angle.

Also, velocity here is measured relative to the air, and because Kerbin is rotating, that's not the same as velocity we've been using in the inertial frame.