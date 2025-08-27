
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from two_body import TwoBody

mu = 3.8986 * 10 ** 5 # km^3/s^2

y0 = np.array([10, 10, 10, 0, 0, 0]) #km, km/s
tb = TwoBody(y0[0:3], y0[3:], mu)

thrust = None

# dy/dt = Ay
def dy_dt(t, y, thrust):
    # === Implement here ===
    if thrust == None:
        thrust = np.array([0, 0, 0])

    semimajor = 6783 #km

    n = np.sqrt(mu/(semimajor**3))

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

def dy_dt_w_airdrag(t, y):
    dy = dy_dt(t, y)
    # Air drag parameter
    rho = 7.48e-12 # atmosphere density @ 400km alt. [kg/m3]
    Cd = 1 # drag coefficient
    A_m = 1 # area mass ratio [m2/kg]
    # === Implement here ===

    vmag = np.linalg.norm(dy[0:3])*1000
    vvec = dy[0:3]

    absdrag = 0.5 * rho * (vmag**2) * Cd * A_m

    dragdir = -vvec/vvec 

    a_drag = absdrag*dragdir/1000

    dy[3:6] += a_drag 

    # ======================
    return dy

# return analytical solution of (r, v)
def y_true(t, tb: TwoBody):
    r, v = tb.calc_states(t)
    return np.array([r[0], r[1], r[2], v[0], v[1], v[2]])


# =========== RK4 ===========
def runge_kutta_4(t, y, h: float, dy_dt) -> tuple:
    """RK4 integrator

    Args:
        t (float): time
        y (_type_): state variables
        h (float): step width

    Returns:
        tuple: y_p (integrated value), calculation time
    """
    time_start = time.perf_counter()
    # === Implement here ===
    k1 = dy_dt(t        , y           )
    k2 = dy_dt(t + (h/2), y + h*(k1/2))
    k3 = dy_dt(t + (h/2), y + h*(k2/2))
    k4 = dy_dt(t + h    , y + h*k3    )

    y_p = y + (h/6)*(k1 +2*k2 + 2*k3 + k4)
    # =====================
    time_end = time.perf_counter()
    return y_p, time_end - time_start

def rk4_all_step(h: float, times: np.array, dy_dt) -> tuple:
    """RK4 all step

    Args:
        h (float): step width
        times (np.array): array of time
        dy_dy (function): equation of motion

    Returns:
        tuple: y_rk4 (array of integrated values), calc_times_rk4 (array of calculation time)
    """
    y_rk4 = np.zeros((len(times), 6))
    y_rk4_i = y0

    calc_time_rk4 = 0
    calc_times_rk4 = [calc_time_rk4]
    y_rk4[0] = y_rk4_i

    for i in range(len(times) - 1):
        t = times[i]

        y_rk4_i, calc_time_i = runge_kutta_4(t, y_rk4_i, h, dy_dt)
        y_rk4[i + 1] = y_rk4_i

        calc_time_rk4 += calc_time_i
        calc_times_rk4.append(calc_time_rk4)

    # output total calculation time
    print("RK4 time: ", calc_times_rk4[-1])
    return y_rk4, calc_times_rk4
# ===========================


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


# plot settings
def plot_init(ylabels: list):
    # fontname = "arial"
    fontsize = 15
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # integrated values
    ax1.set_ylabel(ylabels[0], fontsize=fontsize)
    # errors
    ax2.set_ylabel(ylabels[1], fontsize=fontsize)

    ax2.set_xlabel("t", fontsize=fontsize)
    return ax1, ax2

# plot function
def plot_each(ax, x_data, y_data, label: str):
    ax.plot(x_data, y_data, label=label)

"""## Evaluation of computation error and time"""

t_max = 6000 * 3
# Should be less than 1
h = 1

y0 = np.array([3088.0, 4036.0, 4499.0, -6.8, 1.6, 3.2]) #km, km/s
tb = TwoBody(y0[0:3], y0[3:], mu)

times = np.arange(0, t_max, h)
y_analytical = np.array([y_true(t, tb) for t in times])

# =========== RK4 ===========
sol = sp.solve_ivp(dy_dt, (0,t_max), y0, method='RK45' )
y_rk4, calc_times_rk4 = sol[1], sol[0]

# =========== ABM4 ===========
y_abm4, calc_times_abm4 = abm4_all_step(h, times, dy_dt)

# plot trajectory in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(y_rk4.T[0], y_rk4.T[1], y_rk4.T[2])
ax.plot(y_abm4.T[0], y_abm4.T[1], y_abm4.T[2])
ax.plot(y_analytical.T[0], y_analytical.T[1], y_analytical.T[2])

# plot r
ax1, ax2 = plot_init(["$r$", "Error"])
r_yk4 = np.array([np.linalg.norm(r) for r in y_rk4[:, 0:3]])
plot_each(ax1, times, r_yk4, "RK4")
r_abm4 = np.array([np.linalg.norm(r) for r in y_abm4[:, 0:3]])
plot_each(ax1, times, r_abm4, "ABM4")
r_analytic = np.array([np.linalg.norm(r) for r in y_analytical[:, 0:3]])
plot_each(ax1, times, r_analytic, "analytic")
# error
plot_each(ax2, times, np.fabs(r_yk4 - r_analytic), "RK4")
plot_each(ax2, times, np.fabs(r_abm4 - r_analytic), "ABM4")
ax1.legend()
ax2.set_ylim(ymin=0)
ax2.legend()

# plot v
ax1, ax2 = plot_init(["$v$", "Error"])
v_yk4 = np.array([np.linalg.norm(v) for v in y_rk4[:, 3:]])
plot_each(ax1, times, v_yk4, "RK4")
v_abm4 = np.array([np.linalg.norm(v) for v in y_abm4[:, 3:]])
plot_each(ax1, times, v_abm4, "ABM4")
v_analytic = np.array([np.linalg.norm(v) for v in y_analytical[:, 3:]])
plot_each(ax1, times, v_analytic, "analytic")
# error
plot_each(ax2, times, np.fabs(v_yk4 - v_analytic), "RK4")
plot_each(ax2, times, np.fabs(v_abm4 - v_analytic), "ABM4")
ax1.legend()
ax2.set_ylim(ymin=0)
ax2.legend()

# plot the history of calculation time
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(times, calc_times_rk4, label="RK4")
ax.plot(times, calc_times_abm4, label="ABM4")
ax.legend()

"""## Influence of air drag to orbit and computation time"""

t_max = 6000 * 3
# Should be less than 1
h = 1

times = np.arange(0, t_max, h)
y_wo_drag = np.array([y_true(t, tb) for t in times])

# =========== RK4 ===========
y_rk4, calc_times_rk4 = rk4_all_step(h, times, dy_dt_w_airdrag)

# =========== ABM4 ===========
y_abm4, calc_times_abm4 = abm4_all_step(h, times, dy_dt_w_airdrag)


# plot trajectory in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(y_rk4.T[0], y_rk4.T[1], y_rk4.T[2])
ax.plot(y_abm4.T[0], y_abm4.T[1], y_abm4.T[2])
ax.plot(y_wo_drag.T[0], y_wo_drag.T[1], y_wo_drag.T[2])

# plot r
ax1, ax2 = plot_init(["$r$", "diff"])
r_yk4 = np.array([np.linalg.norm(r) for r in y_rk4[:, 0:3]])
plot_each(ax1, times, r_yk4, "RK4")
r_abm4 = np.array([np.linalg.norm(r) for r in y_abm4[:, 0:3]])
plot_each(ax1, times, r_abm4, "ABM4")
r_wo_drag = np.array([np.linalg.norm(r) for r in y_wo_drag[:, 0:3]])
plot_each(ax1, times, r_wo_drag, "w/o drag")
# Effect of drag
plot_each(ax2, times, r_yk4 - r_wo_drag, "RK4")
plot_each(ax2, times, r_abm4 - r_wo_drag, "ABM4")
ax1.legend()
ax2.legend()

# plot the history of calculation time
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(times, calc_times_rk4, label="RK4")
ax.plot(times, calc_times_abm4, label="ABM4")
ax.legend()

