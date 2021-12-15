import ast2000tools.utils as utils
from ast2000tools.relativity import RelativityExperiments
import ast2000tools.constants as const
import numpy as np
import matplotlib.pyplot as plt

seed = utils.get_seed('antonabr')
experiment = RelativityExperiments(seed)

# experiment.black_hole_descent(6, consider_light_travel=True)

"""
Måler både masse og meter i sekunder.
Masse til meter: kg * G/c^2 [kg * m / kg]
Meter til sekunder: m * 1/c [m * 1 / m / s]
=> Masse til sekunder: kg * G/c^3 [kg * s / kg]
"""
print('')
c = const.c         # m/s
G = const.G         # m^3 / (kg s^2)
solar_mass = const.m_sun        # kg

AU_to_m = const.AU          # m
kg_to_meter = G / c**2      # m
meter_to_sec = 1 / c        # s

M = 1.43937e7 * solar_mass * kg_to_meter        # Black hole mass, m
r = 1 * AU_to_m                 # m
v_SH0 = 0.268             # enhetsløs, ved r = 1AU
gamma_SH = 1 / np.sqrt(1 - v_SH0**2)

print(f'Black hole mass: {M}m')

energy_per_mass = np.sqrt(1 - 2*M / r)*gamma_SH     # enhetsløs
print(f'Energy pr. mass: {energy_per_mass} [-] and {energy_per_mass * c**2} [J / kg]')

dtau12 = 27.5415      # s
dt_SH12 = 30.0208       # s, mellom signal 4 og 5

dtau3031 = 27.5415
dt_SH3031 = 1430.99 - 1124.11
r12 = 2*M / (1 - (dtau12 / dt_SH12 * energy_per_mass)**2)        # m
r3031 = 2*M / (1 - (dtau3031 / dt_SH3031 * energy_per_mass)**2)        # m
print(f'Distance between 1 and 2: {r12}m and {r12 / AU_to_m}AU')
print(f'Distance between 30 and 31: {r3031}m and {r3031 / AU_to_m}AU')


print('')

# Part 2

frame_1 = np.loadtxt('black_hole_descent_frame_1.txt')
frame_2 = np.loadtxt('black_hole_descent_frame_2.txt')
frame_1_wlight = np.loadtxt('black_hole_descent_frame_1_with_light_travel.txt')
frame_2_wlight = np.loadtxt('black_hole_descent_frame_2_with_light_travel.txt')

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('Planet-frame', fontsize=16, weight='bold')
ax.set_xticks(frame_1[0, 1::2])
ax.plot(frame_1[0, 1:], np.diff(frame_1[1]), 'r', label='without lightspeed')
ax.plot(frame_1_wlight[0, 1:], np.diff(frame_1_wlight[1]), 'royalblue', label='with lightspeed')
ax.set_xlabel('n', fontsize=18); ax.set_ylabel(r'$\Delta t_{n-1,n}$', fontsize=18);
ax.legend()
fig.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('Rocket-frame', fontsize=16, weight='bold')
ax.plot(frame_2[0, 1:], np.diff(frame_2[1]), 'r', label='without lightspeed')
ax.plot(frame_2_wlight[0, 1:], np.diff(frame_2_wlight[1]), 'royalblue', label='with lightspeed')
ax.legend()
ax.set_xlabel('n', fontsize=18); ax.set_ylabel(r'$\Delta t_{n-1,n}$', fontsize=18);
fig.tight_layout()
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Planet-frame', fontsize=16, weight='bold')
ax2.set_title('Rocket-frame', fontsize=16, weight='bold')
ax1.set_xticks(frame_1[0, 1::2])
ax1.plot(frame_1[0, 1:], np.diff(frame_1[1]), 'r', label='without lightspeed')
ax1.plot(frame_1_wlight[0, 1:], np.diff(frame_1_wlight[1]), 'royalblue', label='with lightspeed')
ax2.plot(frame_2[0, 1:], np.diff(frame_2[1]), 'r', label='without lightspeed')
ax2.plot(frame_2_wlight[0, 1:], np.diff(frame_2_wlight[1]), 'royalblue', label='with lightspeed')
ax1.legend(); ax2.legend()
ax1.set_xlabel('n', fontsize=18); ax1.set_ylabel(r'$\Delta t_{n-1,n}$', fontsize=18);
ax2.set_xlabel('n', fontsize=18); ax2.set_ylabel(r'$\Delta t_{n-1,n}$', fontsize=18);
fig.tight_layout()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Planet-frame ', fontsize=16, weight='bold')
ax2.set_title('Rocket-frame', fontsize=16, weight='bold')
ax1.plot(frame_1[0], frame_1[1], 'r', label='without lightspeed')
ax1.plot(frame_1_wlight[0], frame_1_wlight[1], 'royalblue', label='with lightspeed')
ax2.plot(frame_2[0], (frame_2[1]), 'r', label='without lightspeed')
ax2.plot(frame_2_wlight[0], frame_2_wlight[1], 'royalblue', label='with lightspeed')
ax1.legend(); ax2.legend()
ax1.set_xticks(frame_1[0, ::2])
ax1.set_xlabel('n', fontsize=18); ax1.set_ylabel(r'$t_n$', fontsize=18);
ax2.set_xlabel('n', fontsize=18); ax2.set_ylabel(r'$t_n$', fontsize=18);
fig.tight_layout()
plt.show()
