# Single-Stage Cases And Generalization

## Purpose

This document is the fixture catalog for the single-stage numerical research.
The cases are not production tests. Each row can become a named regression or
property test after the simulator implementation is stable.

The catalog records the exact normalized state used by the experiments and a
physical-unit reconstruction. The normalized values are the authoritative
fixture values because they avoid loss of precision from rounded physical
values.

The cases cover the three supplied Kerbin states, synthetic states, randomized
rockets and states, boundary states, and body-scaling checks.

The physical fixture definitions are independent of the final-time cap. The
timing and oracle tables were generated with the previous coefficient $1/4$.
The current cap is $3/4$, as specified in `overview.md`. Regenerate all
cap-dependent oracle values before using this document for current regression
tests.

## Coordinate Convention

All single-stage optimizer fixtures use the local two-dimensional polar basis.
The initial position and velocity are represented as

$$
\mathbf r_0=(r_0,0),
\qquad
\mathbf v_0=(v_{r,0},v_{t,0}),
$$

where the second component is the positive prograde tangential component. The
normalized initial mass is

$$
\eta_0=1.
$$

For a physical body and target, define

$$
r_\star=R+h,
\qquad
v_\star=\sqrt{\frac{\mu}{r_\star}},
\qquad
t_\star=\frac{r_\star}{v_\star}.
$$

The conversion from a normalized fixture is

$$
r_0=\rho_0r_\star,
\qquad
v_{r,0}=u_{r,0}v_\star,
\qquad
v_{t,0}=u_{t,0}v_\star.
$$

Stage conversion is

$$
T=\frac{\gamma m_0v_\star^2}{r_\star},
\qquad
v_e=\kappa v_\star,
\qquad
t_{\max}=\tau_{\max}t_\star,
\qquad
\dot m=\frac{T}{v_e}.
$$

## Common Physical Fixtures

### Synthetic Body

All cases whose body is `Synthetic-680` use

$$
R=400{,}000\ \textrm{m},
\qquad
\mu=3.5316\times10^{12}\ \textrm{m}^3\textrm{/s}^2,
$$

with target altitude

$$
h=280{,}000\ \textrm{m}.
$$

Therefore every synthetic optimizer case has

$$
r_\star=680{,}000\ \textrm{m},
\qquad
v_\star=2278.931638238564\ \textrm{m/s},
\qquad
t_\star=298.3854313969623\ \textrm{s}.
$$

The synthetic body is intentional. Using Kerbin's radius with these normalized
states would put some randomized initial positions below the surface. The
synthetic body preserves the normalized dynamics while keeping every listed
initial state above the body surface.

Synthetic optimizer cases use

$$
m_0=10{,}000\ \textrm{kg}.
$$

Every optimizer fixture in this document is a single segment. There is no
staging event, no staging coast, and no discarded dry-mass jump in these
oracles.

### Kerbin

The physical Kerbin case uses

$$
R=600{,}000\ \textrm{m},
\qquad
\mu=3.5316\times10^{12}\ \textrm{m}^3\textrm{/s}^2,
\qquad
h=80{,}000\ \textrm{m},
$$

so it has the same $r_\star$, $v_\star$, and $t_\star$ as the synthetic
fixture. Its supplied inertial vectors are

$$
\mathbf r_0=(424370.58766631,-1093.08696926,-470992.64951719)\ \textrm{m},
$$

$$
\mathbf v_0=(723.81414935,-1.20334290,-122.60883836)\ \textrm{m/s}.
$$

In the local polar basis these become

$$
r_0=633976.077162691\ \textrm{m},
\qquad
v_{r,0}=575.596818331\ \textrm{m/s},
\qquad
v_{t,0}=455.663912116\ \textrm{m/s}.
$$

The Kerbin single-stage study uses

$$
m_0=13885.650390625\ \textrm{kg},
\qquad
T=215000\ \textrm{N},
\qquad
v_e=3138.128\ \textrm{m/s},
$$

and extends the available single-stage burn limit to

$$
t_{\max}=150\ \textrm{s}.
$$

The original first stage alone does not complete the full orbit-insertion task.

The first supplied Kerbin example is also a named fixture. It uses the same
body, target, stage thrust, exhaust velocity, and extended $150\ \textrm{s}$
single-stage burn limit, but has the supplied initial vectors

$$
\mathbf r_0=(428392.15435586,-1053.61873734,-455905.93323801)\ \textrm{m},
$$

$$
\mathbf v_0=(1030.31015,-0.932270447,-119.588146)\ \textrm{m/s}.
$$

In the local polar basis, this is

$$
r_0=625596.649598\ \textrm{m},
\qquad
v_{r,0}=792.681215\ \textrm{m/s},
\qquad
v_{t,0}=668.952682\ \textrm{m/s}.
$$

The coast-first Kerbin fixture uses the same target, thrust, exhaust velocity,
and extended $150\ \textrm{s}$ single-stage burn limit, with

$$
\mathbf r_0=(433284.5917063,-704.8282711,-459791.995176)\ \textrm{m},
$$

$$
\mathbf v_0=(1323.15860984,11.49193645,135.66254872)\ \textrm{m/s},
\qquad
m_0=11777.2275390625\ \textrm{kg}.
$$

Its local polar state is

$$
r_0=631779.006474\ \textrm{m},
\qquad
v_{r,0}=808.699844\ \textrm{m/s},
\qquad
v_{t,0}=1056.072753\ \textrm{m/s}.
$$

## Core And Randomized Cases

The following table gives the exact normalized state, its physical local-polar
conversion, the initial osculating ellipse, and the timing bounds. The ellipse
columns are normalized radii; $f_0$ is the initial true anomaly measured from
periapsis.

`S` means `Synthetic-680` with target altitude $280\ \textrm{km}$.
`K` means Kerbin with target altitude $80\ \textrm{km}$.

| Case | Body | $(\rho_0,u_{r,0},u_{t,0})$ | $r_0$ (m) | $(v_{r,0},v_{t,0})$ (m/s) | $(\rho_p,\rho_a,f_0)$ | $t_{\mathrm{apo}}$ (s) | $t_{\max}$ (s) |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `synthetic-moderate` | S | $(0.779915950272908,0.168676757033227,1.088382144173001)$ | $530342.846$ | $(384.403,2480.349)$ | $(0.620000,0.860000,118.000^\circ)$ | $263.002$ | $561.365$ |
| `synthetic-high-thrust` | S | $(0.827023814541433,0.129618252745517,1.050146588376609)$ | $562376.194$ | $(295.391,2393.212)$ | $(0.660000,0.880000,128.000^\circ)$ | $231.446$ | $548.135$ |
| `kerbin-example` | K | $(0.932317760533369,0.252573095512292,0.199946283806988)$ | $633976.077$ | $(575.597,455.664)$ | $(0.017695,0.962016,177.200^\circ)$ | $70.896$ | $231.589$ |
| `kerbin-first-example` | K | $(0.919995072937620,0.347830185784471,0.293537844816401)$ | $625596.650$ | $(792.681,668.953)$ | $(0.037875,0.979018,174.175^\circ)$ | $103.314$ | $273.242$ |
| `kerbin-coast-first-example` | K | $(0.929086774226051,0.354859194038108,0.463406947156854)$ | $631779.006$ | $(808.700,1056.073)$ | $(0.102136,1.001629,169.194^\circ)$ | $124.744$ | $316.906$ |
| `random-00` | S | $(0.807824351439237,0.131986793361823,1.044291476188662)$ | $549320.559$ | $(300.789,2379.869)$ | $(0.611929,0.850251,136.911^\circ)$ | $185.748$ | $478.738$ |
| `random-01` | S | $(0.825152310008678,0.217646718200955,1.038842720510804)$ | $561103.571$ | $(496.002,2367.452)$ | $(0.604112,0.937634,120.410^\circ)$ | $291.583$ | $608.811$ |
| `random-02` | S | $(0.744449933219338,0.226788650662387,1.142186491981606)$ | $506225.955$ | $(516.836,2602.965)$ | $(0.605042,0.898125,98.494^\circ)$ | $352.727$ | $658.123$ |
| `random-03` | S | $(0.836792625938108,0.064080027091064,1.078573406045483)$ | $569018.986$ | $(146.034,2457.995)$ | $(0.765848,0.869941,114.652^\circ)$ | $277.763$ | $624.455$ |
| `random-04` | S | $(0.778318594216227,0.170353444068123,1.098865337218691)$ | $529256.644$ | $(388.224,2504.239)$ | $(0.631876,0.868368,112.442^\circ)$ | $287.507$ | $592.013$ |
| `random-05` | S | $(0.824516422683998,0.065163792060441,1.074458023272924)$ | $560671.167$ | $(148.504,2448.616)$ | $(0.729969,0.848615,129.818^\circ)$ | $208.302$ | $536.967$ |
| `random-06` | S | $(0.875711245694692,0.070163827047738,1.041849346669014)$ | $595483.647$ | $(159.899,2374.303)$ | $(0.770102,0.905663,127.691^\circ)$ | $239.337$ | $598.815$ |
| `random-07` | S | $(0.814610726593042,0.159321346449047,1.045661387467994)$ | $553935.294$ | $(363.082,2382.991)$ | $(0.617904,0.878688,128.847^\circ)$ | $229.322$ | $532.717$ |

The randomized cases were generated with NumPy seed `20260826`. Their source
sampling ranges were

$$
0.58\le\rho_p\le0.78,
\qquad
\max(\rho_p+0.08,0.78)\le\rho_a\le0.96,
$$

$$
95^\circ\le f_0\le145^\circ,
\qquad
1.0\le\gamma\le3.8,
\qquad
1.05\le\kappa\le1.8.
$$

For each randomized stage, the declared burn limit is

$$
\tau_{\max}=0.99\frac{\kappa}{\gamma},
$$

leaving a positive $1\%$ normalized mass reserve if the limit is reached.

### Stage Specifications

The following table is the complete physical stage fixture. The final-mass
column is the mass remaining after burning for the declared maximum duration;
it is a validity check, not the mass used by the optimal trajectory.

| Case | $\gamma$ | $\kappa$ | $\tau_{\max}$ | $m_0$ (kg) | $T$ (N) | $v_e$ (m/s) | $\dot m$ (kg/s) | $t_{\max}$ (s) | $m(t_{\max})$ (kg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic-moderate` | $1.800000000000000$ | $1.350000000000000$ | $0.600000000000000$ | $10000$ | $137475.779$ | $3076.558$ | $44.685$ | $179.031$ | $2000.000$ |
| `synthetic-high-thrust` | $4.200000000000000$ | $1.350000000000000$ | $0.300000000000000$ | $10000$ | $320776.817$ | $3076.558$ | $104.265$ | $89.516$ | $666.667$ |
| `kerbin-example` | $2.027302475464758$ | $1.377017172145421$ | $0.502705508434977$ | $13885.650391$ | $215000.000$ | $3138.128$ | $68.512$ | $150.000$ | $3608.823$ |
| `kerbin-first-example` | $2.155939481483041$ | $1.377017172145421$ | $0.502705508434977$ | $13057.144531$ | $215000.000$ | $3138.128$ | $68.512$ | $150.000$ | $2780.317$ |
| `kerbin-coast-first-example` | $2.390241108697565$ | $1.377017172145421$ | $0.502705508434977$ | $11777.227539$ | $215000.000$ | $3138.128$ | $68.512$ | $150.000$ | $1500.400$ |
| `random-00` | $2.604468013748685$ | $1.185767289921453$ | $0.450729135787157$ | $10000$ | $198917.371$ | $2702.283$ | $73.611$ | $134.491$ | $100.000$ |
| `random-01` | $2.733916448752875$ | $1.445830928514737$ | $0.523561215589648$ | $10000$ | $208804.051$ | $3294.950$ | $63.371$ | $156.223$ | $100.000$ |
| `random-02` | $1.333757728493972$ | $1.454596210068347$ | $1.079694023286908$ | $10000$ | $101866.323$ | $3314.925$ | $30.730$ | $322.165$ | $100.000$ |
| `random-03` | $3.080828795305368$ | $1.143814443765095$ | $0.367555737291531$ | $10000$ | $235299.632$ | $2606.675$ | $90.268$ | $109.673$ | $100.000$ |
| `random-04` | $3.427969927163351$ | $1.636344475823382$ | $0.472577375381378$ | $10000$ | $261812.686$ | $3729.117$ | $70.208$ | $141.010$ | $100.000$ |
| `random-05` | $1.245280590582974$ | $1.728765403247961$ | $1.374371175587229$ | $10000$ | $95108.844$ | $3939.738$ | $24.141$ | $410.092$ | $100.000$ |
| `random-06` | $3.328491756636065$ | $1.593025285318591$ | $0.473816715730519$ | $10000$ | $254214.998$ | $3630.396$ | $70.024$ | $141.380$ | $100.000$ |
| `random-07` | $1.196961748931338$ | $1.650292772123792$ | $1.364947414452651$ | $10000$ | $91418.471$ | $3760.904$ | $24.308$ | $407.280$ | $100.000$ |

## Two-Burn Primer Oracles

The standard trajectory has two powered arcs separated by a coast. A zero
duration first powered arc is allowed for cases whose optimum starts with a
coast.
The oracle parameters below are normalized. They are the fixed-sequence
segmented-shooting result described in
[Fixed-Sequence Segmented Shooting](burn-sequence-study.md), not trial input
values.

The parameter meanings are

$$
(\alpha_0,\tau_{b1},\tau_{\mathrm{coast}},\tau_{b2},
\lambda_{\rho,0},\lambda_{\eta,0}).
$$

For rows with nonzero $\tau_{b1}$, the first burn begins at the initial state.
The final time is

$$
\tau_f=\tau_{b1}+\tau_{\mathrm{coast}}+\tau_{b2}.
$$

`active` indicates whether the first-outbound final-time cap was used instead
of the free-final-time Hamiltonian condition.

For `kerbin-coast-first-example`, the first powered arc is absent. Its oracle
therefore uses $\tau_{b1}=0$, followed by a coast and the only powered arc in
$\tau_{b2}$. The listed $\alpha_0$, $\lambda_{\rho,0}$, and
$\lambda_{\eta,0}$ are still initial-epoch costate values; they are propagated
through the coast before the second-burn switching condition is evaluated.

| Case | $\alpha_0$ | $\tau_{b1}$ | $\tau_{\mathrm{coast}}$ | $\tau_{b2}$ | $\lambda_{\rho,0}$ | $\lambda_{\eta,0}$ | Fuel fraction | $\tau_f$ | active | Residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `synthetic-moderate` | $1.487621589094740$ | $0.043459832710058$ | $1.728826945647739$ | $0.044695015340527$ | $-1.442760845580149$ | $-1.348789065633785$ | $0.117539797401$ | $1.816981793698323$ | no | $2.18\times10^{-15}$ |
| `synthetic-high-thrust` | $1.506126997024772$ | $0.017574484406716$ | $1.785714718601850$ | $0.015090700611859$ | $-1.324635086026456$ | $-1.349724509094434$ | $0.101625020058$ | $1.818379903620425$ | no | $4.77\times10^{-15}$ |
| `kerbin-example` | $1.357422910819774$ | $0.258909705735881$ | $0.336885348727488$ | $0.052637906396772$ | $-1.120006399999580$ | $-1.373896477245226$ | $0.458673470511$ | $0.648432960860141$ | no | $8.68\times10^{-16}$ |
| `kerbin-first-example` | $1.301998497886693$ | $0.063223698031544$ | $0.255177693503371$ | $0.203673515995724$ | $-1.129547596616809$ | $-1.375099704109453$ | $0.417870054825$ | $0.522074907530639$ | no | $9.88\times10^{-16}$ |
| `kerbin-coast-first-example` | $2.180458606526267$ | $0$ | $0.314482092435442$ | $0.195766154962485$ | $1.087412012182590$ | $-1.370608266588306$ | $0.339812981819$ | $0.510248247397927$ | no | $2.65\times10^{-16}$ |
| `random-00` | $1.239804612479387$ | $0.036840516672452$ | $1.539034193310046$ | $0.028554979627370$ | $-1.537796939157507$ | $-1.152059213002539$ | $0.143637356000$ | $1.604429689609868$ | yes | $1.48\times10^{-15}$ |
| `random-01` | $1.461718579953581$ | $0.016173041755944$ | $1.347547249590145$ | $0.036203042612770$ | $-1.327872930120024$ | $-1.445364143023450$ | $0.099037747604$ | $1.399923333958859$ | no | $4.90\times10^{-15}$ |
| `random-02` | $1.461215074550734$ | $0.037062182302178$ | $1.601172373570937$ | $0.078261910904414$ | $-1.542722793685555$ | $-1.452765631782100$ | $0.105743710544$ | $1.716496466777529$ | no | $7.30\times10^{-15}$ |
| `random-03` | $1.383739344054379$ | $0.017536807097521$ | $2.059265288377157$ | $0.015976305776372$ | $-1.342318064813348$ | $-1.135389054597601$ | $0.090266532063$ | $2.092778401251050$ | yes | $1.12\times10^{-15}$ |
| `random-04` | $1.487221427883699$ | $0.020226981652751$ | $1.750329178016678$ | $0.024372766706107$ | $-1.447020161113065$ | $-1.635647177776289$ | $0.093431791650$ | $1.794928926375536$ | no | $5.20\times10^{-15}$ |
| `random-05` | $1.084573627057953$ | $0.060084771107680$ | $1.687823412841049$ | $0.051668218882705$ | $-1.453899761417061$ | $-1.607550901586431$ | $0.080498967132$ | $1.799576402831434$ | yes | $1.55\times10^{-15}$ |
| `random-06` | $1.490433427399960$ | $0.015148761941717$ | $1.979374914158261$ | $0.012327042827403$ | $-1.240187810201673$ | $-1.590213742973144$ | $0.057408372939$ | $2.006850718927381$ | yes | $1.57\times10^{-15}$ |
| `random-07` | $1.490945614409436$ | $0.069082715447696$ | $1.625013452527046$ | $0.061090352268316$ | $-1.354109823258457$ | $-1.648558375890214$ | $0.094414873184$ | $1.755186520243058$ | no | $4.67\times10^{-15}$ |

The physical burn and coast times are obtained by multiplying each normalized
time by $t_\star$. For example, the synthetic-moderate oracle has physical
durations approximately $12.968\ \textrm{s}$,
$515.857\ \textrm{s}$, and $13.336\ \textrm{s}$ for the first burn, coast,
and second burn.

## Edge Cases

The edge cases use `Synthetic-680`, target altitude $280\ \textrm{km}$, and
$m_0=10{,}000\ \textrm{kg}$. Their exact states and timing bounds are:

| Case | $(\rho_0,u_{r,0},u_{t,0})$ | $r_0$ (m) | $(v_{r,0},v_{t,0})$ (m/s) | $(\rho_p,\rho_a,f_0)$ | $t_{\mathrm{apo}}$ (s) | $t_{\max}$ (s) |
| --- | --- | ---: | --- | --- | ---: | ---: |
| `near-target` | $(0.876433561110304,0.109812410337372,1.058532732369112)$ | $595974.822$ | $(250.255,2412.324)$ | $(0.780000,0.960000,100.000^\circ)$ | $388.005$ | $768.348$ |
| `low-thrust` | $(0.779915950272908,0.168676757033227,1.088382144173001)$ | $530342.846$ | $(384.403,2480.349)$ | $(0.620000,0.860000,118.000^\circ)$ | $263.002$ | $561.365$ |
| `very-high-thrust` | $(0.827023814541433,0.129618252745517,1.050146588376609)$ | $562376.194$ | $(295.391,2393.212)$ | $(0.660000,0.880000,128.000^\circ)$ | $231.446$ | $548.135$ |
| `early-outbound` | $(0.655292005577328,0.287685454880135,1.260952498423414)$ | $445598.564$ | $(655.615,2873.625)$ | $(0.550000,0.900000,80.000^\circ)$ | $405.585$ | $694.922$ |
| `near-apoapsis` | $(0.889681849865437,0.063494032998192,0.976545203065486)$ | $604983.658$ | $(144.699,2225.480)$ | $(0.650000,0.900000,160.000^\circ)$ | $96.359$ | $416.138$ |

Their stage specifications are:

| Case | $\gamma$ | $\kappa$ | $\tau_{\max}$ | $T$ (N) | $v_e$ (m/s) | $\dot m$ (kg/s) | $t_{\max}$ (s) | $m(t_{\max})$ (kg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `near-target` | $1.800000000000000$ | $1.350000000000000$ | $0.600000000000000$ | $137475.779$ | $3076.558$ | $44.685$ | $179.031$ | $2000.000$ |
| `low-thrust` | $0.500000000000000$ | $1.350000000000000$ | $1.500000000000000$ | $38187.716$ | $3076.558$ | $12.412$ | $447.578$ | $4444.444$ |
| `very-high-thrust` | $10.000000000000000$ | $1.350000000000000$ | $0.100000000000000$ | $763754.325$ | $3076.558$ | $248.250$ | $29.839$ | $2592.593$ |
| `early-outbound` | $2.000000000000000$ | $1.350000000000000$ | $0.600000000000000$ | $152750.865$ | $3076.558$ | $49.650$ | $179.031$ | $1111.111$ |
| `near-apoapsis` | $2.000000000000000$ | $1.350000000000000$ | $0.600000000000000$ | $152750.865$ | $3076.558$ | $49.650$ | $179.031$ | $1111.111$ |

Their fixed two-arc primer oracles are:

| Case | $\alpha_0$ | $\tau_{b1}$ | $\tau_{\mathrm{coast}}$ | $\tau_{b2}$ | $\lambda_{\rho,0}$ | $\lambda_{\eta,0}$ | Fuel fraction | $\tau_f$ | active | Residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `near-target` | $1.516602517235983$ | $0.011064010771531$ | $1.659848095355963$ | $0.029339907790798$ | $-1.216397561552053$ | $-1.349906288734626$ | $0.053871891416$ | $1.700252013918292$ | no | $1.41\times10^{-15}$ |
| `low-thrust` | $1.433983803367139$ | $0.162217063963106$ | $1.563295353534075$ | $0.155829616912271$ | $-1.468437022688288$ | $-1.312832372899953$ | $0.117795066991$ | $1.881342034409451$ | yes | $8.56\times10^{-16}$ |
| `very-high-thrust` | $1.506128816099643$ | $0.007350937079308$ | $1.794155648411117$ | $0.006368045488266$ | $-1.324621959997202$ | $-1.349884093034865$ | $0.101622093093$ | $1.807874630978692$ | no | $2.50\times10^{-15}$ |
| `early-outbound` | $1.437589753698373$ | $0.020222879667095$ | $1.648217450404746$ | $0.067522110489159$ | $-1.852573406713204$ | $-1.348140276803419$ | $0.129992578009$ | $1.735962440561000$ | no | $4.21\times10^{-15}$ |
| `near-apoapsis` | $1.071327398725223$ | $0.049341662098238$ | $1.318999966219061$ | $0.026291865060102$ | $-1.562897437190621$ | $-1.271652334877388$ | $0.112049669864$ | $1.394633493377402$ | yes | $5.63\times10^{-15}$ |

## Boundary Fixtures

The exact-circular boundary case uses `Synthetic-680`, target altitude
$280\ \textrm{km}$, $m_0=10{,}000\ \textrm{kg}$, and

$$
\gamma=2,
\qquad
\kappa=1.4,
\qquad
\tau_{\max}=0.5.
$$

Its state is

$$
(\rho_0,u_{r,0},u_{t,0},\eta_0)=(1,0,1,1).
$$

Coasting for normalized time $3$ returned zero terminal error and zero mass
change. The optimizer should recognize this state and return zero burn rather
than entering primer shooting.

The near-radial boundary check uses the same body, target, mass, and stage, with

$$
(\rho_0,u_{r,0},u_{t,0},\eta_0)=(0.8,0.25,0.05,1).
$$

After normalized coast time $0.1$, the state was

$$
(0.817337881125333,0.097843641932390,0.048939368801706,1).
$$

This confirms finite state propagation, but not good conditioning for an
orbital-plane or primer initialization.

## Body-Scaling Fixtures

These cases validate physical-unit conversion rather than optimizer event
selection. They use the common normalized initial orbit

$$
(\rho_0,u_{r,0},u_{t,0})
=(0.798279569892473,0.153605010709000,1.075774195852128),
$$

corresponding to

$$
(\rho_p,\rho_a,f_0)=(0.64,0.87,120^\circ).
$$

The scaling experiment uses normalized duration $0.17$, normalized throttle
$q=1$, normalized thrust angle $\alpha=0.37$, and normalized stage limit
$\tau_{\max}=0.25$. The physical fixtures are:

| Body | $R$ (m) | $\mu$ (m$^3$/s$^2$) | $h$ (m) | $m_0$ (kg) | $T$ (N) | $v_e$ (m/s) | $t_{\max}$ (s) | $r_0$ (m) | $(v_{r,0},v_{t,0})$ (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Kerbin | $600000$ | $3.5316\times10^{12}$ | $80000$ | $13885.65$ | $215000$ | $3138.13$ | $74.596$ | $542830.108$ | $(350.055,2451.616)$ |
| Mun | $200000$ | $6.51384\times10^{10}$ | $10000$ | $2500$ | $60000$ | $3200$ | $94.265$ | $167638.710$ | $(85.549,599.142)$ |
| Minmus | $60000$ | $1.7658\times10^9$ | $10000$ | $1500$ | $30000$ | $3200$ | $110.183$ | $55879.570$ | $(24.396,170.861)$ |
| Duna | $320000$ | $3.01363\times10^{11}$ | $30000$ | $8000$ | $120000$ | $3300$ | $94.297$ | $279397.849$ | $(142.533,998.233)$ |

The converted-state errors were $3.03\times10^{-14}$ for Kerbin,
$1.05\times10^{-12}$ for Mun, $4.23\times10^{-13}$ for Minmus, and
$5.48\times10^{-14}$ for Duna.

## Generalization Results

The state-equation scaling test used the four body definitions in the table
above. The normalized state was unchanged after conversion to physical units.
The maximum converted-state error was

$$
1.05\times10^{-12}.
$$

The randomized optimizer cases used NumPy seed `20260826`. The generator
sampled the initial ellipse and stage parameters within the ranges listed in
the core case section. The fixed two-arc solver reached the terminal orbit for
all eight randomized cases under the previous cap coefficient $1/4$.

These cases do not establish validity for zero tangential velocity,
near-parabolic motion, a target below the current orbit, a singular throttle
arc, a collision constraint, or an unrestricted number of revolutions.

## What Each Case Changed

- `synthetic-moderate`: A restricted one-burn root used fuel fraction
  $0.602543$, while the accepted two-burn reference used $0.117540$. This was
  the clearest counterexample to treating an explicit coast followed by one
  burn as the general solution.
- `synthetic-high-thrust`: The same two-arc structure survived a high-thrust
  regime with very short burns. It became the stiffness and short-event test.
- `kerbin-example`: The supplied physical state is almost at apoapsis but has
  low tangential speed. It verified that the result is not an artifact of the
  synthetic state generator and required the extended single-stage burn budget.
- `kerbin-first-example`: The earlier supplied state has higher radial and
  tangential velocity than the second example. It showed that the same local
  prograde and impulse-timing initialization works for both supplied Kerbin
  states, despite a substantially different burn split.
- `kerbin-coast-first-example`: The initial apoapsis is already just above the
  target, but the angular momentum is too low for a circular orbit. The direct
  two-arc solve drives the first burn to zero, establishing a coast followed by
  one burn.
- `random-00`: The minimum-fuel trajectory within the old capped domain reached
  the final-time cap. It
  forced the replacement of $H(t_f)=0$ with the active time-bound condition.
- `random-01`: A lower-energy free-final-time case with a different burn-time
  split. It tested that the two-arc solution was not tied to one timing ratio.
- `random-02`: A higher tangential-speed case with a free final time. It tested
  the costate and direct solver away from the moderate reference parameters.
- `random-03`: Another active-time-bound case with a long coast. It tested the
  active-boundary residual in a different orbital geometry.
- `random-04`: A high-$\gamma$ and high-$\kappa$ case with a free final time.
  It tested parameter scaling and a short second burn.
- `random-05`: An active-boundary case with comparatively low $\gamma$. It
  tested the low-thrust side of the randomized range.
- `random-06`: A high-$\gamma$ case with a free final time. It tested the
  stiffness and event timing at another thrust-to-gravity ratio.
- `random-07`: A higher-exhaust-velocity case with a free final time. It tested
  that the mass-costate scaling did not depend on one exhaust velocity.
- `near-target`: The target is only slightly above the initial apoapsis. It
  tested the small-transfer limit and produced a clean two-burn solution.
- `low-thrust`: The final-time cap is active and both burns are relatively long.
  It tested the boundary condition and low-thrust conditioning together.
- `very-high-thrust`: The stage has $\gamma=10$ but a short physically valid
  burn limit. It tested short event handling without allowing negative mass.
- `early-outbound`: The initial true anomaly is $80^\circ$ with substantial
  radial velocity. It tested states far from apoapsis and rejected reliance on
  an apoapsis-start assumption.
- `near-apoapsis`: The state is close to apoapsis with small radial velocity.
  It tested the conditioning of the initial costate estimate and an active
  final-time boundary.
- `exact-circular`: Zero burn is exactly optimal. It established the required
  early-exit boundary behavior.
- `near-radial-coast`: The state equations remain finite, but the local orbital
  direction becomes sensitive. It established a numerical conditioning limit.
- Body-scaling fixtures: Kerbin, Mun, Minmus, and Duna produced the same
  normalized result after physical conversion. This supported keeping body
  data outside the dimensionless equations.

## Future Test Assertions

When production code exists, the following assertions should be attached to
these fixtures:

- Exact terminal orbit within the documented normalized residual tolerance.
- Positive mass throughout and at the declared burn limit.
- Correct $\Phi$ sign on every powered and coast arc.
- Correct use of $H(t_f)=0$ for interior final time and the active time-bound
  equation when the cap is reached.
- Agreement with the listed two-burn fuel and timing oracle after integration
  tolerance and mesh refinement.
- No lower-fuel three-burn trial in the tested first-outbound domain.
- Zero-burn handling for `exact-circular`.
- Explicit infeasibility or domain reporting for zero tangential velocity,
  target-below-current-orbit cases, insufficient propellant, and singular arcs.

## Remaining Falsifiers

The catalog does not prove global optimality. It can still be falsified by a
valid first-outbound case with a lower-fuel three-or-more-burn solution, a
sustained singular throttle arc, a body-collision constraint that becomes
active, or a physically valid state that needs a different event sequence.
