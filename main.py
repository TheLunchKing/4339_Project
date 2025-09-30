
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

def abm4_all_step(h: float, times: np.array, dy_dt, y) -> tuple:
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
    y_abm4_i = y
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




#----------------
# Milestone 1a
#----------------
time_start = time.perf_counter()
sol = sp.solve_ivp(
                    fun = dy_dt,
                    t_span = (0,3*T_orb),
                    y0 = y0,
                    method = 'BDF',
                    )
time_end = time.perf_counter()
print(time_end - time_start)


# Extract the results
states = sol.y.T  # Each row is [x, y, z, ux, uy, uz] at a time step
times = sol.t

# Convert positions to km for plotting
x = states[:, 0] / 1000
y = states[:, 1] / 1000
z = states[:, 2] / 1000

# Plot the 3D graph
# figure = plt.figure(figsize=(10, 10))
# plot = figure.add_subplot(111, projection='3d')
# plot.plot(x, y, z, label='Ceres path')
# plot.scatter(0, 0, 0, color='red', marker='*', s=50, label='ISS')
# plot.set_xlabel('x (km)')
# plot.set_ylabel('y (km)')
# plot.set_zlabel('z (km)')
# plot.set_title('Trajectory of Ceres relative to ISS (no thrust)')
# plot.legend()
# plt.show()



#--------------------------------
# Milestone 1B
#--------------------------------
# Two initial conditions to compare
y0_1 = np.array([10000, 10000, 10000, 0, 0, 0])  
y0_2 = np.array([20000,  5000,  5000, 2, 2, 0]) 
test = enumerate([y0_1, y0_2])   

# Time setup
t0, tf = 0, T_orb
N = 1000
period = np.linspace(t0, tf, N)
h = period[1] - period[0]

fig = plt.figure(figsize=(15, 10))

# Test both initial conditions
for num, y0_test in test:
    sol_ref = sp.solve_ivp(
        fun=dy_dt,
        t_span=(t0, tf),
        y0=y0_test,
        method='BDF',
        t_eval=period,
        rtol=1e-8
    )
    

    y_abm4, _ = abm4_all_step(h, period, dy_dt, y0_test)
    
    # 3D trajectory comparison
    ax1 = fig.add_subplot(2, 3, num*3 + 1, projection='3d')
    ax1.plot(sol_ref.y[0]/1000, sol_ref.y[1]/1000, sol_ref.y[2]/1000, 
             'b-', label='scipy (BDF)', linewidth=2)
    ax1.plot(y_abm4[:, 0]/1000, y_abm4[:, 1]/1000, y_abm4[:, 2]/1000, 
             'r--', label='ABM4', linewidth=1.5)
    ax1.scatter(0, 0, 0, color='red', marker='*', s=100, label='ISS')
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    ax1.set_zlabel('z (km)')
    ax1.set_title(f'IC: {y0_test[:3]/1000} km')
    ax1.legend()
    
    # Position error over time
    ax2 = fig.add_subplot(2, 3, num*3 + 2)
    pos_errors = np.linalg.norm(sol_ref.y[:3].T - y_abm4[:, :3], axis=1)
    ax2.plot(period/T_orb, pos_errors, 'k-', linewidth=2)
    ax2.set_xlabel('Time (orbits)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error vs Time')
    ax2.grid(True)
    
    # Velocity error over time  
    ax3 = fig.add_subplot(2, 3, num*3 + 3)
    vel_errors = np.linalg.norm(sol_ref.y[3:].T - y_abm4[:, 3:], axis=1)
    ax3.plot(period/T_orb, vel_errors, 'r-', linewidth=2)
    ax3.set_xlabel('Time (orbits)')
    ax3.set_ylabel('Velocity Error (m/s)')
    ax3.set_title('Velocity Error vs Time')
    ax3.grid(True)

plt.tight_layout()
plt.show()

