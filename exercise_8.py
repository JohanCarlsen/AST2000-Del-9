import ast2000tools.utils as utils
from ast2000tools.relativity import RelativityExperiments
import ast2000tools.constants as const
import numpy as np
import matplotlib.pyplot as plt

c = const.c         # m/s
G = const.G         # m^3 / (kg s^2)
solar_mass = const.m_sun        # kg
kg_to_meter = G / c**2      # m
AU_to_m = const.AU          # m


M = 1.43937e7 * solar_mass * kg_to_meter / AU_to_m       # Black hole mass, AU

"""
Potensiale for lys rundt et sort hull
"""

r = np.linspace(2*M, 2, 501)

def V(r):
    return 1/r * np.sqrt(1 - 2*M/r)

plt.plot(r, V(r), color='limegreen', label='Light potential (black hole)', linewidth=2)
plt.xlabel('AU', weight='bold', fontsize=18)
plt.ylabel('v(r)', weight='bold', fontsize=18)
plt.xticks(r[::100])
plt.legend()
plt.show()
