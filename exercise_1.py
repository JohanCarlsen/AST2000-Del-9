import ast2000tools.utils as utils
from ast2000tools.relativity import RelativityExperiments
import ast2000tools.constants as const
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


c = const.c         # m/s
G = const.G         # m^3 / (kg s^2)
solar_mass = const.m_sun        # kg
kg_to_meter = G / c**2      # m
AU_to_m = const.AU          # m
meter_to_sec = 1 / c        # s
from_deg_to_rad = np.pi / 180   # rad

# 1.4 a)
sun_mass = solar_mass * kg_to_meter         # m
sun_radii = const.R_sun         # m
print(f'Sun mass: {solar_mass} [kg] and {sun_mass} [m]')
print(f'Sun radii: {sun_radii} [m]')
M_r = sun_mass / sun_radii
print(f'Ratio sun mass and sun radii: {M_r} [-]')

# 1.4 b)
lambda_SH = 500         # nm
lambda_fa = M_r * lambda_SH + lambda_SH
redshift = M_r
print(f'Redshift: {redshift}')
print(f'lambda_fa: {lambda_fa}nm')

# 1.4 c)
earth_mass = 5.972e27 * kg_to_meter             # m
earth_radii = 6371 * 1000           # m
print(f'Earth mass: {earth_mass / kg_to_meter} [kg] and {earth_mass} [m]')
M_r_earth = earth_mass / earth_radii

# 1.4 d)
light_at_earth = lambda_SH / (M_r_earth + 1)
print(f'Ratio earth mass and earth radii: {M_r_earth} [-]')
print(f'Light at earth: {light_at_earth}nm')
blueshift = (light_at_earth - lambda_SH) / lambda_SH
print(f'Blueshift: {blueshift}')

# 1.5
lambda_a = 2150         # nm
lambda_b = 600          # nm

shift = (lambda_a - lambda_b) / lambda_b        # no dimension
r = 2 / (1 - 1 / (shift + 1)**2)
print(f'The distance r for the quasar from the black hole is {r}M')
print(r * 4e6 * solar_mass * kg_to_meter / AU_to_m)
r = 1 / shift
print(r * 4e6 * solar_mass * kg_to_meter / AU_to_m)
