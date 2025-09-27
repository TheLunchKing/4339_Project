
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp


mu = 3.8986 * 10 ** 5 # km^3/s^2
a_iss = 6770  # Semi-major plotis of ISS in meters
n = np.sqrt((mu)/a_iss**3)  # Orbital rate (rad/s)

T_orb = 2 * np.pi / n  # Orbital period

y0 = np.array([10000, 10000, 10000, 0, 0, 0]) #m, m/s

thrust = None
def thrust_func(t, y):
    return np.array([0.0, 0.0, 0.0])

# dy/dt = Ay
def dy_dt(t, y):
    # === Implement here ===
    thrust = thrust_func(t, y)

    A = np.array([
         [0,0,0,1,0,0],
         [0,0,0,0,1,0],
         [0,0,0,0,0,1],
         [(3*(n**2)),0,0,0,2*n,0],
         [0,0,0,-2*n,0,0],
         [0,0,-n**2,0,0,0]
        ])
    
    B = np.array([0, 0, 0, thrust[0], thrust[1], thrust[2]])

    # ======================
    return A @ y + B




# =========== ABM4 ===========
def adams_4(t, y, h: float, f_m: list, dy_dt) -> tuple:
    """Adams-Bashforth-Moulton4 integrator

    Args:
        t (float): time
        y (_type_): state variables
        h (float): step width
        f_m (list): previous steps' derivatives

    Returns:
        tuple: y_p (integrated value), calculation time
    """
    time_start = time.perf_counter()
    # === Implement here ===

    fi = dy_dt(t,y) 

    y_pred = y + (h/24)*(55*fi - 59*f_m[-1] + 37*f_m[-2] - 9*f_m[-3] )

    fi1 = dy_dt(t+h,y_pred) 

    y_p = y + (h/720)*(251*fi1 + 646*fi - 264*f_m[-1] + 106*f_m[-2] - 19*f_m[-3])

    f_m.append(fi)
    if len(f_m) > 4:
        f_m.pop(0)
    
    # ======================
    time_end = time.perf_counter()
    return y_p, time_end - time_start

def abm4_all_step(h: float, times: np.array, dy_dt) -> tuple:
    """ABM4 all step

    Args:
        h (float): step width
        times (np.array): array of time
        dy_dy (function): equation of motion

    Returns:
        tuple: y_abm4 (array of integrated values), calc_times_abm4 (array of calculation time)
    """
    fm_ab4 = []

    y_abm4 = np.zeros((len(times), 6))
    y_abm4_i = y0
    y_abm4[0] = y_abm4_i

    calc_time_abm4 = 0
    calc_times_abm4 = [calc_time_abm4]

    # Initialization of 3 steps for ABM4
    for i in range(3):
        t = times[i]
        fm_ab4.append(dy_dt(t, y_abm4_i))
        y_abm4_i, calc_time_i = runge_kutta_4(t, y_abm4_i, h, dy_dt)

        y_abm4[i + 1] = y_abm4_i

        calc_time_abm4 += calc_time_i
        calc_times_abm4.append(calc_time_abm4)

    # Integration with ABM4
    for i in range(len(times) - 3 - 1):
        t = times[i + 3]

        y_abm4_i, calc_time_i = adams_4(t, y_abm4_i, h, fm_ab4, dy_dt)

        y_abm4[i + 4] = y_abm4_i

        calc_time_abm4 += calc_time_i
        calc_times_abm4.append(calc_time_abm4)

    # output total calculation time
    print("ABM4 time: ", calc_times_abm4[-1])
    return y_abm4, calc_times_abm4
# ===========================




time_start = time.perf_counter()
sol = sp.solve_ivp(
                    fun = dy_dt,
                    t_span = (0,3*T_orb),
                    y0 = y0,
                    method = 'BDF',
                    )
time_end = time.perf_counter()
print(time_end - time_start)

print(sol)

# Extract the results
states = sol.y.T  # Each row is [x, y, z, ux, uy, uz] at a time step
times = sol.t

# Convert positions to km for plotting
x = states[:, 0] / 1000
y = states[:, 1] / 1000
z = states[:, 2] / 1000

# Plot the 3D graph
figure = plt.figure(figsize=(10, 10))
plot = figure.add_subplot(111, projection='3d')
plot.plot(x, y, z, label='Ceres path')
plot.scatter(0, 0, 0, color='red', marker='*', s=50, label='ISS')
plot.set_xlabel('x (km)')
plot.set_ylabel('y (km)')
plot.set_zlabel('z (km)')
plot.set_title('Trajectory of Ceres relative to ISS (no thrust)')
plot.legend()
plt.show()