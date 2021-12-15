'''
EGEN KODE
'''
import ast2000tools.utils as utils
from ast2000tools.relativity import RelativityExperiments
import ast2000tools.constants as const
import numpy as np
import matplotlib.pyplot as plt

seed = utils.get_seed('antonabr')
experiments = RelativityExperiments(seed)

# experiments.gps(6, write_solutions=True)

'''task 1'''

R = 1861743.5943    # m

r_1 = np.array([-3831207, -3148374])  # pos of satelite 1 in m
r_1_norm = np.linalg.norm(r_1)

# print(r_1_norm/1000)
'''
4958.871436196445
'''

h1 = r_1_norm-R # height above surface of satelite 1

# print(h1/1000)
'''
3097.127841896445
'''

r_2 = np.array([-1743749, -4642152])  # pos of satelite 2 in m
r_2_norm = np.linalg.norm(r_2)

# print(r_2_norm/1000)

'''
4958.854279579608
'''

h2 = r_2_norm - R

# print(h2/1000)
'''
3097.1106852796083
'''

# print(f'Absolute error: {abs(h1-h2):e}')
'''
Absolute error: 1.715662e+01
'''


''' task 2'''

M = 1.648794339672e23   # planet mass in kg
G = const.G # gravitational constant in m^2/(kg s^2)

def vel(r):
    '''Function to calculate angular velocity'''
    v = np.sqrt(G * M / r)
    return v

v_1 = vel(r_1_norm)   # m/s

# print(v_1/1000)       # km/s
'''
1.4896858112961426
'''

v_2 = vel(r_2_norm)   # m/s

# print(v_2/1000)       # km/s
'''
1.489688388297313
'''


'''task 3'''

c = const.c_km_pr_s*1000 # light speed m/s
t_planet = 91.5963797   # s

r_1_vec = np.array([1996089, -4539368])   # m
r_1 = np.linalg.norm(r_1_vec)
t_1 = 91.5781213    # s
delta_t_1 = abs(t_planet - t_1)

r_2_vec = np.array([3998348, -2933164])   # m
r_2 = np.linalg.norm(r_2_vec)
t_2 = 91.5755782    # s
delta_t_2 = abs(t_planet - t_2)

# print(f'Delta t_1: {delta_t_1:.3e} s\nDelta t_2: {delta_t_2:.3e} s')
'''
Delta t_1: 1.826e-02 s
Delta t_2: 2.080e-02 s
'''

def theta_angle(r_vec, delta_t):
    '''Function to calculate the angle of my position'''

    r = np.linalg.norm(r_vec)

    theta = np.arctan(r_vec[1] / r_vec[0])
    alpha = np.arccos((r**2 + R**2 - (c * delta_t)**2) / (2 * r * R))

    theta_real_1 = theta + alpha
    theta_real_2 = theta - alpha

    return theta_real_1, theta_real_2

theta_list = np.zeros([1,2,2])
theta_list[:,0,:] = theta_angle(r_1_vec, delta_t_1)
theta_list[:,1,:] = theta_angle(r_2_vec, delta_t_2)

# print(theta_list, f'\nAbsolute error:\t{abs(theta_list[0,0,1] - theta_list[0,1,1]):.3e}')
'''
[[[ 0.51765771 -2.83069176]
  [ 1.56485984 -2.83069646]]]
Absolute error: 4.696e-06
'''

# print(f'Arc length:\n{abs(theta_list[0,0,1] - theta_list[0,1,1]) * R:.3f} m')
'''
Arc length:
8.742 m
'''

theta_real = (theta_list[0,0,1] + theta_list[0,1,1]) / 2

# print(theta_real)
'''
-2.83069410812754
'''

x = R * np.cos(theta_real)  # x coordinate of my position
y = R * np.sin(theta_real)  # y coordinate of my position

r = np.array([x, y])

# print(r/1000)
'''
[-1772.48988692  -569.53385473]
'''


'''task 4'''

def planet_time_gravitational_effects(r_1, r_2, delta_t_2):
    '''Function to calculate time interval'''

    delta_t_1 = np.sqrt((1 - (2*M / r_1)) / (1 - (2*M / r_2))) * delta_t_2
    return delta_t_2

planet_time_1 = planet_time_gravitational_effects(R, r_1, delta_t_1)
planet_time_2 = planet_time_gravitational_effects(R, r_2, delta_t_2)

# print(planet_time_1)
# print(planet_time_2)
'''
0.018258399999993458
0.02080150000000458
'''

def kg_to_m(mass_in_kg):
    '''Convert mass measured in kg to mass measured in meters'''

    M_m = (G / const.c**2) * mass_in_kg

    return M_m

my_weight_in_meters = kg_to_m(83.1)

# print(my_weight_in_meters)
'''
6.17113918363761e-26
'''

planet_mass_in_meters = kg_to_m(M)

# print(f'Our planets mass in meters:\n{planet_mass_in_meters:.3e}')
'''
Our planets mass in meters:
1.224e-04
'''

def planet_time_gravitational_and_relativistic_effects(r_vec, v, delta_t):
    '''
    Calculate time inteval with both
    gravitational and relativistic effects
    '''

    r = np.linalg.norm(r_vec)
    # time = (1 + (planet_mass_in_meters/r) - (planet_mass_in_meters/R) + (v**2/2)) * delta_t
    time = float(np.sqrt((1 - (2*planet_mass_in_meters / R)) / (1 - (2*planet_mass_in_meters / r) - (v/c)**2)) * delta_t)

    return time

delta_t_1_new = planet_time_gravitational_and_relativistic_effects(r_1_vec, v_1, delta_t_1)
delta_t_2_new = planet_time_gravitational_and_relativistic_effects(r_2_vec, v_2, delta_t_2)

# print(delta_t_1_new)
# print(delta_t_2_new)
'''
0.018258399999468895
0.020801499999406954
'''


'''task 5'''

theta_list_new = np.zeros((1,2,2))
theta_list_new[:,0,:] = theta_angle(r_1_vec, delta_t_1_new)
theta_list_new[:,1,:] = theta_angle(r_2_vec, delta_t_2_new)

# print(theta_list_new)
'''
[[[ 0.51765771 -2.83069176]
  [ 1.56485984 -2.83069646]]]
'''

theta_real_new = (theta_list_new[0,0,1] + theta_list_new[0,1,1]) / 2

# print(theta_real_new)
'''
-2.8306941080059467
'''

x_new = R * np.cos(theta_real_new)  # new x coordinate of my position
y_new = R * np.sin(theta_real_new)  # new y coordinate of my position

r_new = np.array([x_new, y_new])

# print(f'New position: {r_new/1000} km\nOld position: {r/1000} km\n\
# Absolute error:\n\
# x: {abs(x-x_new):.3e} m\n\
# y: {abs(y-y_new):.3e} m\n\
# Relative error:\n\
# x: {abs(x-x_new)/abs(x):.3e}\ny: {abs(y-y_new)/abs(y):.3e}')

'''
New position: [-1772.48988685  -569.53385495] km
Old position: [-1772.48988692  -569.53385473] km
Absolute error:
x: 6.925e-05 m
y: 2.155e-04 m
Relative error:
x: 3.907e-11
y: 3.784e-10
'''


'''task 6'''

# late for late in the video

t_p_late = 13006.6859172    # s

r_1_vec_late = np.array([1578461, 4700925]) # m
r_1_late = np.linalg.norm(r_1_vec_late)     # m
t_1_late = 13006.6650847                    # s
delta_t_1_late = abs(t_p_late - t_1_late)   # s

r_2_vec_late = np.array([-983476, 4860351]) # m
r_2_late = np.linalg.norm(r_2_vec_late)     # m
t_2_late = 13006.6676152                    # s
delta_t_2_late = abs(t_p_late - t_2_late)   # s

# print(f'From the end of the video:\nDelta t_1: {delta_t_1_late:.3e} s\n\
# Delta t_2: {delta_t_2_late:.3e} s')
'''
From the end of the video:
Delta t_1: 2.083e-02 s
Delta t_2: 1.830e-02 s
'''

v_1_late = vel(r_1_late)   # m/s

# print(v_1_late/1000)       # km/s
'''
1.4896884001401771
'''

v_2_late = vel(r_2_late)   # m/s

# print(v_2_late/1000)       # km/s
'''
1.4896883713397455
'''

theta_list_late = np.zeros([1,2,2])
theta_list_late[:,0,:] = theta_angle(r_1_vec_late, delta_t_1_late)
theta_list_late[:,1,:] = theta_angle(r_2_vec_late, delta_t_2_late)

# print(theta_list_late)
'''
[[[ 3.4524066  -0.95870986]
  [ 0.3108331  -3.05312378]]]
'''

r_1 = (c/1000) * delta_t_1_late
x_1 = r_1_vec_late[0] / 1000
y_1 = r_1_vec_late[1] / 1000

r_2 = (c/1000) * delta_t_2_late
x_2 = r_2_vec_late[0] / 1000
y_2 = r_2_vec_late[1] / 1000


fig, ax = plt.subplots()
t = np.linspace(0, 2*np.pi, 1001)
ax.plot((R/1000)*np.cos(t), (R/1000)*np.sin(t), label='Surface', color='black', lw=0.5)
ax.plot((r_1*np.cos(t)+x_1), (r_1*np.sin(t)+y_1), label='Sat. 1', color='red')
ax.plot((r_2*np.cos(t)+x_2), (r_2*np.sin(t)+y_2), label='Sat. 2', color='royalblue')
ax.plot(x_1, y_1, 'ro')
ax.plot(x_2, y_2, 'o', color='royalblue')
ax.plot(R/1000*np.cos(theta_list_late[0,0,0]), R/1000*np.sin(theta_list_late[0,0,0]), 'o', label=f'$\\theta={theta_list_late[0,0,0]:.2}$')
ax.plot(R/1000*np.cos(theta_list_late[0,0,1]), R/1000*np.sin(theta_list_late[0,0,1]), 'o', label=f'$\\theta={theta_list_late[0,0,1]:.2}$')
ax.plot(R/1000*np.cos(theta_list_late[0,1,0]), R/1000*np.sin(theta_list_late[0,1,0]), 'o', label=f'$\\theta={theta_list_late[0,1,0]:.2}$')
ax.plot(R/1000*np.cos(theta_list_late[0,1,1]), R/1000*np.sin(theta_list_late[0,1,1]), 'o', label=f'$\\theta={theta_list_late[0,1,1]:.2}$')
plt.axis('equal')
plt.legend()
# plt.show()

theta_real_late = theta_list_late[0,0,0]

x_late = R * np.cos(theta_real_late)  # x coordinate of my position
y_late = R * np.sin(theta_real_late)  # y coordinate of my position

r_late = np.array([x_late, y_late])

# print(f'Position at end of video: {r_late/1000} km\nPosition at start of video: {r/1000} km\n\
# Absolute error:\nx: {abs(r[0]-r_late[0]):.2f} m\ny: {abs(r[1]-r_late[1]):.2f} m\n\
# Relative error:\nx: {abs(r[0]-r_late[0])/abs(r[0]):.3e}, in percent: {(abs(r[0]-r_late[0])/abs(r[0]))*100:.3e} %\n\
# y: {abs(r[1]-r_late[1])/abs(r[1]):.3e}, in percent: {(abs(r[1]-r_late[1])/abs(r[1]))*100:.3e} %')
'''
Position at end of video: [-1772.53806076  -569.38390746] km
Position at start of video: [-1772.48988692  -569.53385473] km
Absolute error:
x: 48.17 m
y: 149.95 m
Relative error:
x: 2.718e-05, in percent: 2.718e-03 %
y: 2.633e-04, in percent: 2.633e-02 %
'''

delta_t_1_new_late = planet_time_gravitational_and_relativistic_effects(r_1_vec_late, v_1_late, delta_t_1_late)
delta_t_2_new_late = planet_time_gravitational_and_relativistic_effects(r_2_vec_late, v_2_late, delta_t_2_late)

# print(delta_t_1_new_late)
# print(delta_t_2_new_late)
'''
0.0208324999997257
0.01830199999987768
'''

last_theta_list = np.zeros((1,2,2))
last_theta_list[:,0,:] = theta_angle(r_1_vec_late, delta_t_1_new_late)
last_theta_list[:,1,:] = theta_angle(r_2_vec_late, delta_t_2_new_late)

# print(last_theta_list)
'''
[[[ 3.4524066  -0.95870986]
  [ 0.3108331  -3.05312378]]]
'''

last_theta = last_theta_list[0,0,0]

last_x = R * np.cos(last_theta) # x-comp
last_y = R * np.sin(last_theta) # y-comp

last_r = np.array([last_x, last_y])

# print(f'Position at end of video with gravitational and relativistic effects: {last_r/1000} km\n\
# Position at start of video: {r/1000} km\n\
# Total time elapsed: {abs(t_p_late-t_planet):.2f} s or {abs(t_p_late-t_planet)/3600:.2f} hours\n\
# Absolute error:\nx: {abs(r[0]-last_r[0]):.2f} m\ny: {abs(r[1]-last_r[1]):.2f} m\n\
# Relative error:\nx: {abs(r[0]-last_r[0])/abs(r[0]):.3e}, in percent: {(abs(r[0]-last_r[0])/abs(r[0]))*100:.3e} %\n\
# y: {abs(r[1]-last_r[1])/abs(r[1]):.3e}, in percent: {(abs(r[1]-last_r[1])/abs(r[1]))*100:.3e} %\n\
# Length of miss: {np.sqrt(abs(r[0]-last_r[0])**2+abs(r[1]-last_r[1])**2):.2f} m')
'''
Position at end of video with gravitational and relativistic effects: [-1772.53806085  -569.38390719] km
Position at start of video: [-1772.48988692  -569.53385473] km
Total time elapsed: 12915.09 s or 3.59 hours
Absolute error:
x: 48.17 m
y: 149.95 m
Relative error:
x: 2.718e-05, in percent: 2.718e-03 %
y: 2.633e-04, in percent: 2.633e-02 %
Length of miss: 157.50 m
'''
