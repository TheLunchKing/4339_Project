
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

# Constants
mu = 3.8986 * 10 ** 5  # km^3/s^2
a_iss = 6770  # Semi-major axis of ISS in km
n = np.sqrt(mu / a_iss**3)  # Orbital rate (rad/s)
T_orb = 2 * np.pi / n  # Orbital period

y0 = np.array([10, 10, 10, 0, 0, 0])  # km, km/s (consistent units)

# Thrust function
def thrust_func(t, y):
    return np.array([0.0, 0.0, 0.0])

# Dynamics function
def dy_dt(t, y):
    thrust = thrust_func(t, y)
    
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
    ])
    
    B = np.array([0, 0, 0, thrust[0], thrust[1], thrust[2]])
    return A @ y + B

# ===================== INTEGRATION METHODS =====================

# ============ RK4 (Baseline) ============
def runge_kutta_4(t, y, h: float, dy_dt) -> tuple:
    time_start = time.perf_counter()
    
    k1 = dy_dt(t, y)
    k2 = dy_dt(t + h/2, y + h*(k1/2))
    k3 = dy_dt(t + h/2, y + h*(k2/2))
    k4 = dy_dt(t + h, y + h*k3)

    y_p = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    time_end = time.perf_counter()
    return y_p, time_end - time_start

def rk4_all_step(h: float, times: np.array, dy_dt, y0) -> tuple:
    y_rk4 = np.zeros((len(times), 6))
    y_rk4_i = y0.copy()
    y_rk4[0] = y_rk4_i

    calc_time_rk4 = 0
    calc_times_rk4 = [calc_time_rk4]

    for i in range(len(times) - 1):
        t = times[i]
        y_rk4_i, calc_time_i = runge_kutta_4(t, y_rk4_i, h, dy_dt)
        y_rk4[i + 1] = y_rk4_i
        calc_time_rk4 += calc_time_i
        calc_times_rk4.append(calc_time_rk4)

    print("RK4 time: ", calc_times_rk4[-1])
    return y_rk4, calc_times_rk4

# ============ ABM4 ============
def adams_4(t, y, h: float, f_m: list, dy_dt) -> tuple:
    time_start = time.perf_counter()
    
    f_n_minus_3, f_n_minus_2, f_n_minus_1, f_n = f_m

    # Adams-Bashforth predictor
    y_pred = y + (h/24) * (55*f_n - 59*f_n_minus_1 + 37*f_n_minus_2 - 9*f_n_minus_3)
    
    # Adams-Moulton corrector
    f_pred = dy_dt(t + h, y_pred)
    y_p = y + (h/720) * (251*f_pred + 646*f_n - 264*f_n_minus_1 + 106*f_n_minus_2 - 19*f_n_minus_3)
    
    f_m_updated = [f_n_minus_2, f_n_minus_1, f_n, f_pred]
    
    time_end = time.perf_counter()
    return y_p, time_end - time_start, f_m_updated

def abm4_all_step(h: float, times: np.array, dy_dt, y0) -> tuple:
    fm_ab4 = []
    y_abm4 = np.zeros((len(times), 6))
    y_abm4_i = y0.copy()
    y_abm4[0] = y_abm4_i

    calc_time_abm4 = 0
    calc_times_abm4 = [calc_time_abm4]

    # Initialisation with RK4
    fm_ab4.append(dy_dt(times[0], y_abm4_i))
    
    for i in range(3):
        t = times[i]
        y_abm4_i, calc_time_i = runge_kutta_4(t, y_abm4_i, h, dy_dt)
        y_abm4[i + 1] = y_abm4_i
        fm_ab4.append(dy_dt(times[i + 1], y_abm4_i))
        calc_time_abm4 += calc_time_i
        calc_times_abm4.append(calc_time_abm4)

    # Main integration
    for i in range(len(times) - 4):
        t = times[i + 3]
        y_abm4_i, calc_time_i, fm_ab4 = adams_4(t, y_abm4_i, h, fm_ab4, dy_dt)
        y_abm4[i + 4] = y_abm4_i
        calc_time_abm4 += calc_time_i
        calc_times_abm4.append(calc_time_abm4)

    # print("ABM4 time: ", calc_times_abm4[-1])
    return y_abm4, calc_times_abm4

# ============ VELOCITY VERLET  ============
def velocity_verlet_step(t, y, h: float, dy_dt) -> tuple:
    """Velocity Verlet - symplectic, excellent for orbital mechanics"""
    time_start = time.perf_counter()
    
    # Get current acceleration
    dy_current = dy_dt(t, y)
    a_current = dy_current[3:6]
    
    # Update position
    y_new = y.copy()
    y_new[0:3] = y[0:3] + h * y[3:6] + 0.5 * h**2 * a_current
    
    # Update velocity with new acceleration
    dy_new = dy_dt(t + h, y_new)
    a_new = dy_new[3:6]
    y_new[3:6] = y[3:6] + 0.5 * h * (a_current + a_new)
    
    time_end = time.perf_counter()
    return y_new, time_end - time_start

def velocity_verlet_all_step(h: float, times: np.array, dy_dt, y0) -> tuple:
    y_vv = np.zeros((len(times), 6))
    y_vv_i = y0.copy()
    y_vv[0] = y_vv_i

    calc_time_vv = 0
    calc_times_vv = [calc_time_vv]

    for i in range(len(times) - 1):
        t = times[i]
        y_vv_i, calc_time_i = velocity_verlet_step(t, y_vv_i, h, dy_dt)
        y_vv[i + 1] = y_vv_i
        calc_time_vv += calc_time_i
        calc_times_vv.append(calc_time_vv)

    print("Velocity Verlet time: ", calc_times_vv[-1])
    return y_vv, calc_times_vv

# ============ HIGH PRECISION RK4 ============
def rk4_high_precision_step(t, y, h: float, dy_dt, micro_steps=4) -> tuple:
    """RK4 with micro-stepping for higher precision"""
    time_start = time.perf_counter()
    
    h_micro = h / micro_steps
    y_temp = y.copy()
    t_temp = t
    
    for _ in range(micro_steps):
        k1 = dy_dt(t_temp, y_temp)
        k2 = dy_dt(t_temp + h_micro/2, y_temp + h_micro*(k1/2))
        k3 = dy_dt(t_temp + h_micro/2, y_temp + h_micro*(k2/2))
        k4 = dy_dt(t_temp + h_micro, y_temp + h_micro*k3)
        y_temp = y_temp + (h_micro/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_temp += h_micro
    
    time_end = time.perf_counter()
    return y_temp, time_end - time_start

def rk4_high_precision_all_step(h: float, times: np.array, dy_dt, y0) -> tuple:
    y_rk4hp = np.zeros((len(times), 6))
    y_rk4hp_i = y0.copy()
    y_rk4hp[0] = y_rk4hp_i

    calc_time_rk4hp = 0
    calc_times_rk4hp = [calc_time_rk4hp]

    for i in range(len(times) - 1):
        t = times[i]
        y_rk4hp_i, calc_time_i = rk4_high_precision_step(t, y_rk4hp_i, h, dy_dt, micro_steps=4)
        y_rk4hp[i + 1] = y_rk4hp_i
        calc_time_rk4hp += calc_time_i
        calc_times_rk4hp.append(calc_time_rk4hp)

    print("High-precision RK4 time: ", calc_times_rk4hp[-1])
    return y_rk4hp, calc_times_rk4hp

# ============ DORMAND-PRINCE 8 ============
def dormand_prince_8_step(t, y, h: float, dy_dt) -> tuple:
    """Dormand-Prince 8 method - very high accuracy"""
    time_start = time.perf_counter()
    
    # Butcher tableau coefficients for Dormand-Prince 8
    # Simplified to not use 13 steps
    c = np.array([0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 1, 1])
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0],
        [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0],
        [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0],
        [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0],
        [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0],
        [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0],
        [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0]
    ])
    b8 = np.array([13451932/455176623, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535, 53011238/667516719, 2/45])
    
    k = np.zeros((13, len(y)))
    k[0] = dy_dt(t, y)
    
    # Compute intermediate stages
    for i in range(1, 10):
        y_stage = y.copy()
        for j in range(i):
            y_stage += h * a[i][j] * k[j]
        k[i] = dy_dt(t + c[i] * h, y_stage)
    
    # 8th order solution
    y8 = y.copy()
    for i in range(10):
        y8 += h * b8[i] * k[i]
    
    time_end = time.perf_counter()
    return y8, time_end - time_start

def dormand_prince_all_step(h: float, times: np.array, dy_dt, y0) -> tuple:
    y_dp = np.zeros((len(times), 6))
    y_dp_i = y0.copy()
    y_dp[0] = y_dp_i

    calc_time_dp = 0
    calc_times_dp = [calc_time_dp]

    for i in range(len(times) - 1):
        t = times[i]
        y_dp_i, calc_time_i = dormand_prince_8_step(t, y_dp_i, h, dy_dt)
        y_dp[i + 1] = y_dp_i
        calc_time_dp += calc_time_i
        calc_times_dp.append(calc_time_dp)

    print("Dormand-Prince 8 time: ", calc_times_dp[-1])
    return y_dp, calc_times_dp


# ============ METHOD SWITCHER ============
def integrate(method: str, h: float, times: np.array, dy_dt, y) -> tuple:
    """Switch between different integration methods"""
    method = method.upper()
    
    if method == 'RK4':
        return rk4_all_step(h, times, dy_dt, y)
    elif method == 'ABM4':
        return abm4_all_step(h, times, dy_dt, y)
    elif method == 'VELOCITY_VERLET' or method == 'VV':
        return velocity_verlet_all_step(h, times, dy_dt, y)
    elif method == 'RK4_HP' or method == 'HIGH_PRECISION':
        return rk4_high_precision_all_step(h, times, dy_dt, y)
    elif method == 'DORMAND_PRINCE' or method == 'DP8':
        return dormand_prince_all_step(h, times, dy_dt, y)
    else:
        raise ValueError(f"Unknown method")

# ===================== TESTING =====================

def compare_methods():
    """Compare all integration methods"""
    # Test setup
    y0_test = np.array([10, 10, 10, 0, 0, 0])  # km, km/s
    t0, tf = 0, T_orb
    N = 6000
    period = np.linspace(t0, tf, N)
    h = period[1] - period[0]
    
    # Reference solution (highest accuracy)
    print("Computing reference solution...")
    sol_ref = sp.solve_ivp(
        fun=dy_dt,
        t_span=(t0, tf),
        y0=y0_test,
        method='DOP853',
        t_eval=period,
        rtol=1e-13,
        atol=1e-15
    )
    
    # Methods to test
    methods = ['RK4', 'ABM4', 'VELOCITY_VERLET', 'RK4_HP']
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    fig = plt.figure(figsize=(20, 10))
    
    for idx, method in enumerate(methods):
        print(f"\n=== Testing {method} ===")
        
        # Integrate
        y_method, _ = integrate(method, h, period, dy_dt, y0_test)
        
        # Calculate errors
        pos_errors = np.linalg.norm(sol_ref.y[:3].T - y_method[:, :3], axis=1)
        vel_errors = np.linalg.norm(sol_ref.y[3:].T - y_method[:, 3:], axis=1)
        
        print(f"Max position error: {np.max(pos_errors):.6e} km")
        print(f"Max velocity error: {np.max(vel_errors):.6e} km/s")
        
        # Plot trajectory
        ax1 = fig.add_subplot(2, len(methods), idx + 1, projection='3d')
        ax1.plot(sol_ref.y[0], sol_ref.y[1], sol_ref.y[2], 
                'k-', label='Reference', linewidth=2, alpha=0.3)
        ax1.plot(y_method[:, 0], y_method[:, 1], y_method[:, 2], 
                color=colors[idx], linestyle='--', label=method, linewidth=1.5)
        ax1.scatter(0, 0, 0, color='red', marker='*', s=100, label='ISS')
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('y (km)')
        ax1.set_zlabel('z (km)')
        ax1.set_title(f'{method}\nMax Pos Error: {np.max(pos_errors):.2e} km')
        ax1.legend()
        
        # Plot errors
        ax2 = fig.add_subplot(2, len(methods), idx + len(methods) + 1)
        ax2.semilogy(period/T_orb, pos_errors, color=colors[idx], linewidth=2, label='Position Error')
        ax2.semilogy(period/T_orb, vel_errors, color=colors[idx], linestyle='--', linewidth=2, label='Velocity Error')
        ax2.set_xlabel('Time (orbits)')
        ax2.set_ylabel('Error (km or km/s)')
        ax2.set_title(f'{method} Errors')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_single_method(method_name='VELOCITY_VERLET'):
    """Test a single method in detail"""
    # Two initial conditions to compare
    y0_1 = np.array([10, 10, 10, 0, 0, 0])  
    y0_2 = np.array([20, 5, 5, 0.002, 0.002, 0])  # km, km/s
    
    # Time setup
    t0, tf = 0, T_orb
    N = 6000
    period = np.linspace(t0, tf, N)
    h = period[1] - period[0]
    
    fig = plt.figure(figsize=(15, 10))
    
    for num, y0_test in enumerate([y0_1, y0_2]):
        print(f"\n=== Testing {method_name} with IC {num+1} ===")
        
        # Reference solution
        sol_ref = sp.solve_ivp(
            fun=dy_dt,
            t_span=(t0, tf),
            y0=y0_test,
            method='DOP853',
            t_eval=period,
            rtol=1e-12,
            atol=1e-14
        )
        
        # Method solution
        y_method, _ = integrate(method_name, h, period, dy_dt, y0_test)
        
        # Calculate errors
        pos_errors = np.linalg.norm(sol_ref.y[:3].T - y_method[:, :3], axis=1)
        vel_errors = np.linalg.norm(sol_ref.y[3:].T - y_method[:, 3:], axis=1)
        
        print(f"Max position error: {np.max(pos_errors):.6e} km")
        print(f"Max velocity error: {np.max(vel_errors):.6e} km/s")
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 3, num*3 + 1, projection='3d')
        ax1.plot(sol_ref.y[0], sol_ref.y[1], sol_ref.y[2], 
                'b-', label='Reference', linewidth=2, alpha=0.7)
        ax1.plot(y_method[:, 0], y_method[:, 1], y_method[:, 2], 
                'r--', label=method_name, linewidth=1.5)
        ax1.scatter(0, 0, 0, color='red', marker='*', s=100, label='ISS')
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('y (km)')
        ax1.set_zlabel('z (km)')
        ax1.set_title(f'IC: {y0_test[:3]} km - {method_name}')
        ax1.legend()
        
        # Position error
        ax2 = fig.add_subplot(2, 3, num*3 + 2)
        ax2.plot(period/60, pos_errors*1000, 'k-', linewidth=2)
        ax2.set_xlabel('Time (mins)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title(f'Position Error - {method_name}')
        ax2.grid(True)
        
        # Velocity error
        ax3 = fig.add_subplot(2, 3, num*3 + 3)
        ax3.plot(period/60, vel_errors*1000, 'r-', linewidth=2)
        ax3.set_xlabel('Time (mins)')
        ax3.set_ylabel('Velocity Error (m/s)')
        ax3.set_title(f'Velocity Error - {method_name}')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("Orbital Mechanics Integration Methods Comparison")
    print("=" * 50)
    print(f"Orbital period: {T_orb:.2f} s")
    print(f"Orbital rate: {n:.6f} rad/s")
    
    # Compare all methods
    # compare_methods()
    
    # Test a specific method
    test_single_method('ABM4')  




# Available methods:
# - 'RK4' - Standard 4th order Runge-Kutta
# - 'ABM4' - Adams-Bashforth-Moulton 4th order  
# - 'VELOCITY_VERLET' - Symplectic (best for orbits)
# - 'RK4_HP' - High-precision RK4 with micro-stepping
# - 'DORMAND_PRINCE' - 8th order method






































# =========== RK4 ===========
# def runge_kutta_4(t, y, h: float, dy_dt) -> tuple:
#     """RK4 integrator

#     Args:
#         t (float): time
#         y (_type_): state variables
#         h (float): step width

#     Returns:
#         tuple: y_p (integrated value), calculation time
#     """
#     time_start = time.perf_counter()
#     # === Implement here ===
#     k1 = dy_dt(t        , y           )
#     k2 = dy_dt(t + (h/2), y + h*(k1/2))
#     k3 = dy_dt(t + (h/2), y + h*(k2/2))
#     k4 = dy_dt(t + h    , y + h*k3    )

#     y_p = y + (h/6)*(k1 +2*k2 + 2*k3 + k4)
#     # =====================
#     time_end = time.perf_counter()
#     return y_p, time_end - time_start

# def rk4_all_step(h: float, times: np.array, dy_dt) -> tuple:
#     """RK4 all step

#     Args:
#         h (float): step width
#         times (np.array): array of time
#         dy_dy (function): equation of motion

#     Returns:
#         tuple: y_rk4 (array of integrated values), calc_times_rk4 (array of calculation time)
#     """
#     y_rk4 = np.zeros((len(times), 6))
#     y_rk4_i = y0

#     calc_time_rk4 = 0
#     calc_times_rk4 = [calc_time_rk4]
#     y_rk4[0] = y_rk4_i

#     for i in range(len(times) - 1):
#         t = times[i]

#         y_rk4_i, calc_time_i = runge_kutta_4(t, y_rk4_i, h, dy_dt)
#         y_rk4[i + 1] = y_rk4_i

#         calc_time_rk4 += calc_time_i
#         calc_times_rk4.append(calc_time_rk4)

#     # output total calculation time
#     print("RK4 time: ", calc_times_rk4[-1])
#     return y_rk4, calc_times_rk4
# # ===========================


# # =========== ABM4 ===========
# def adams_4(t, y, h: float, f_m: list, dy_dt) -> tuple:
#     """Adams-Bashforth-Moulton4 integrator

#     Args:
#         t (float): time
#         y (_type_): state variables
#         h (float): step width
#         f_m (list): previous steps' derivatives

#     Returns:
#         tuple: y_p (integrated value), calculation time
#     """
#     time_start = time.perf_counter()
#     # === Implement here ===

#     fi = dy_dt(t,y) 

#     f_n_minus_3, f_n_minus_2, f_n_minus_1, f_n = f_m

#     y_pred = y + (h/24) * (55*f_n - 59*f_n_minus_1 + 37*f_n_minus_2 - 9*f_n_minus_3)

#     f_pred = dy_dt(t + h, y_pred)

#     y_p = y + (h/720) * (251*f_pred + 646*f_n - 264*f_n_minus_1 + 106*f_n_minus_2 - 19*f_n_minus_3)

#     f_m_updated = [f_n_minus_2, f_n_minus_1, f_n, f_pred]
    
#     # ======================
#     time_end = time.perf_counter()
#     return y_p, time_end - time_start, f_m_updated

# def abm4_all_step(h: float, times: np.array, dy_dt, y) -> tuple:
#     """ABM4 all step

#     Args:
#         h (float): step width
#         times (np.array): array of time
#         dy_dy (function): equation of motion

#     Returns:
#         tuple: y_abm4 (array of integrated values), calc_times_abm4 (array of calculation time)
#     """
#     fm_ab4 = []

#     y_abm4 = np.zeros((len(times), 6))
#     y_abm4_i = y
#     y_abm4[0] = y_abm4_i

#     calc_time_abm4 = 0
#     calc_times_abm4 = [calc_time_abm4]

#     # Initialisation of 3 steps for ABM4
#     for i in range(3):
#         t = times[i]
#         # Store current derivative before stepping (for first step only)
#         if i == 0:
#             fm_ab4.append(dy_dt(t, y_abm4_i))

#         # fm_ab4.append(dy_dt(t, y_abm4_i))
        
#         # Take RK4 step
#         y_abm4_i, calc_time_i = runge_kutta_4(t, y_abm4_i, h, dy_dt)
#         y_abm4[i + 1] = y_abm4_i

#         # Store derivative at new point
#         fm_ab4.append(dy_dt(times[i + 1], y_abm4_i))

#         calc_time_abm4 += calc_time_i
#         calc_times_abm4.append(calc_time_abm4)

#     # Verify we have 4 derivatives for ABM4 [f0, f1, f2, f3]
#     if len(fm_ab4) != 4:
#         raise ValueError(f"Initialisation failed: expected 4 derivatives, got {len(fm_ab4)}")

#     # Integration with ABM4
#     for i in range(len(times) - 4):
#         t = times[i + 3]

#         y_abm4_i, calc_time_i, fm_ab4_updated = adams_4(t, y_abm4_i, h, fm_ab4, dy_dt)
#         y_abm4[i + 4] = y_abm4_i

#         # Update derivative history for next step
#         fm_ab4 = fm_ab4_updated

#         calc_time_abm4 += calc_time_i
#         calc_times_abm4.append(calc_time_abm4)

#     # output total calculation time
#     print("ABM4 time: ", calc_times_abm4[-1])
#     return y_abm4, calc_times_abm4
# ===========================


