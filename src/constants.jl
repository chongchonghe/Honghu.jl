"""
    Honghu.CGS

Physical constants in CGS units (2018 CODATA where applicable).

Honghu is unit-agnostic: every physical parameter (`RadiationPhysics.c`,
`IdealGas.cv`, opacities, ...) is supplied by the user. Use these when you want
to run in CGS; use your own numbers when you want code units.
"""
module CGS

"Speed of light [cm s⁻¹]"
const c_light = 2.99792458e10
"Boltzmann constant [erg K⁻¹]"
const k_B = 1.380649e-16
"Atomic mass unit [g]"
const m_u = 1.66053907e-24
"Hydrogen mass [g]"
const m_H = 1.6735575e-24
"Planck constant [erg s]"
const h_planck = 6.62607015e-27
"Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴]"
const sigma_SB = 5.670374419e-5
"Radiation constant a_r = 4σ/c [erg cm⁻³ K⁻⁴]"
const a_rad = 4 * sigma_SB / c_light
"Gravitational constant [cm³ g⁻¹ s⁻²]"
const G_grav = 6.67430e-8
"Solar mass [g]"
const M_sun = 1.98892e33
"Parsec [cm]"
const parsec = 3.0856775814913673e18
"Astronomical unit [cm]"
const au = 1.495978707e13
"Year [s]"
const year = 3.1556952e7

end # module CGS
