
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize


mu = 3.986e14 # m^3/s^2
a_iss = 6770e3  # Semi-major plotis of ISS in meters
n = np.sqrt((mu)/a_iss**3)  # Orbital rate (rad/s)

T_orb = 2 * np.pi / n  # Orbital period

y0 = np.array([10000, 10000, 10000, 0, 0, 0]) #m, m/s


# PD Controller parameters
kp = 5e-5  # Position gain
kd = 1e-2  # Velocity gain
max_thrust = 0.07  # m/s²
# Simulation parameters
t0, tf = 0, (2/3)*T_orb  # Simulate for 2 orbits
N = 2000
times_docking = np.linspace(t0, tf, N)
h_docking = times_docking[1] - times_docking[0]


# System matrices for CW equations
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [3*n**2, 0, 0, 0, 2*n, 0],
    [0, 0, 0, -2*n, 0, 0],
    [0, 0, -n**2, 0, 0, 0]
])

B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# ============================================================================
# METHOD 1: PD CONTROLLER
# ============================================================================
def thrust_func_pd(t, y):
    """PD Controller - Simple but effective"""
    kp = 5e-5  # Position gain
    kd = 1e-2  # Velocity gain
    
    target_pos = np.array([0.0, 0.0, 0.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    
    current_pos = y[:3]
    current_vel = y[3:]
    
    pos_error = current_pos - target_pos
    vel_error = current_vel - target_vel
    
    u = -kp * pos_error - kd * vel_error
    
    # Saturate thrust
    u_norm = np.linalg.norm(u)
    if u_norm > max_thrust:
        u = u * (max_thrust / u_norm)
        
    return u

# ============================================================================
# METHOD 2: LQR CONTROLLER  
# ============================================================================
def thrust_func_lqr(t, y):
    """LQR Controller - Optimal for linear systems"""
    # Moderate weights - balance between fuel and performance
    Q = np.diag([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])  
    R = np.diag([1e3, 1e3, 1e3])
    
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    
    u = -K @ y
    
    # Saturate thrust
    u_norm = np.linalg.norm(u)
    if u_norm > max_thrust:
        u = u * (max_thrust / u_norm)
        
    return u

# ============================================================================
# METHOD 3: FUEL-OPTIMIZED LQR
# ============================================================================
def thrust_func_lqr_fuel_optimized(t, y):
    """LQR with heavy control penalty to minimize fuel usage"""
    # Light state penalties, heavy control penalties
    Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])  
    R = np.diag([1e4, 1e4, 1e4])  # Very high control penalty
    
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    
    u = -K @ y
    
    # Saturate thrust
    u_norm = np.linalg.norm(u)
    if u_norm > max_thrust:
        u = u * (max_thrust / u_norm)
        
    return u

# ============================================================================
# METHOD 4: TIME-VARYING LQR (TVLQR)
# ============================================================================
def thrust_func_tvlqr(t, y):
    """Time-varying LQR - accounts for changing dynamics"""
    # Time-varying weights - more aggressive near docking
    time_factor = min(1.0, t / (0.8 * tf))  # Ramp up over 80% of trajectory
    
    Q = np.diag([1e-3 * (1 + 9 * time_factor), 
                 1e-3 * (1 + 9 * time_factor), 
                 1e-3 * (1 + 9 * time_factor),
                 1e-1 * (1 + 9 * time_factor),
                 1e-1 * (1 + 9 * time_factor), 
                 1e-1 * (1 + 9 * time_factor)])
    
    R = np.diag([1e3, 1e3, 1e3])
    
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    
    u = -K @ y
    
    # Saturate thrust
    u_norm = np.linalg.norm(u)
    if u_norm > max_thrust:
        u = u * (max_thrust / u_norm)
        
    return u

# ============================================================================
# METHOD 5: SLIDING MODE CONTROL (SMC)
# ============================================================================
def thrust_func_smc(t, y):
    """Sliding Mode Control - Robust to uncertainties"""
    lambda_smc = 0.1  # Sliding surface parameter
    
    # Sliding surface: s = λ*pos_error + vel_error
    pos_error = y[:3]
    vel_error = y[3:]
    s = lambda_smc * pos_error + vel_error
    
    # SMC control law
    k_smc = 0.01
    u = -lambda_smc * vel_error - k_smc * np.sign(s)
    
    # Add CW dynamics compensation
    x, y_pos, z, vx, vy, vz = y
    u[0] -= (3*n**2*x + 2*n*vy)
    u[1] -= (-2*n*vx) 
    u[2] -= (-n**2*z)
    
    # Saturate thrust
    u_norm = np.linalg.norm(u)
    if u_norm > max_thrust:
        u = u * (max_thrust / u_norm)
        
    return u

# ============================================================================
# METHOD 6: OPTIMAL CONTROL VIA DIRECT COLLOCATION (MOST FUEL EFFICIENT)
# ============================================================================
class OptimalControlDocking:
    def __init__(self, times, y0, max_thrust=0.07):
        self.times = times
        self.y0 = y0
        self.max_thrust = max_thrust
        self.N = len(times)
        self.dt = times[1] - times[0]
        self.control_profile = None
        self.optimized = False
        
        # System matrices
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1],
            [3*n**2, 0, 0, 0, 2*n, 0],
            [0, 0, 0, -2*n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        
        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    
    def dynamics(self, y, u):
        """CW dynamics with control input"""
        return self.A @ y + self.B @ u
    
    def simulate_trajectory(self, U_flat):
        """Simulate trajectory given control sequence"""
        U = U_flat.reshape(3, self.N)
        Y = np.zeros((6, self.N))
        Y[:, 0] = self.y0
        
        for k in range(self.N-1):
            # RK4 integration
            k1 = self.dynamics(Y[:, k], U[:, k])
            k2 = self.dynamics(Y[:, k] + 0.5*self.dt*k1, U[:, k])
            k3 = self.dynamics(Y[:, k] + 0.5*self.dt*k2, U[:, k]) 
            k4 = self.dynamics(Y[:, k] + self.dt*k3, U[:, k])
            Y[:, k+1] = Y[:, k] + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return Y
    
    def objective(self, U_flat):
        """Performance index J = ½∫uᵀu dt"""
        U = U_flat.reshape(3, self.N)
        return 0.5 * np.sum(U**2) * self.dt
    
    def constraints(self, U_flat):
        """Docking constraints"""
        Y = self.simulate_trajectory(U_flat)
        final_pos = Y[:3, -1]
        final_vel = Y[3:, -1]
        
        # Position < 10m, velocity < 0.5 m/s
        pos_constraint = 10 - np.linalg.norm(final_pos)
        vel_constraint = 0.5 - np.linalg.norm(final_vel)
        
        return np.array([pos_constraint, vel_constraint])
    
    def optimize_control(self):
        """Solve the optimal control problem"""
        print("Solving optimal control problem...")
        
        # Initial guess: zero control
        U0_flat = np.zeros(3 * self.N)
        
        # Bounds: control constraints
        bounds = [(-self.max_thrust, self.max_thrust) for _ in range(3 * self.N)]
        
        # Constraints
        constraints = {
            'type': 'ineq',
            'fun': self.constraints
        }
        
        # Options for better convergence
        options = {
            'maxiter': 1000,
            'ftol': 1e-6,
            'disp': True
        }
        
        # Solve optimization
        result = minimize(
            self.objective, 
            U0_flat, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if result.success:
            self.control_profile = result.x.reshape(3, self.N)
            self.optimized = True
            
            # Verify solution
            Y = self.simulate_trajectory(result.x)
            final_pos = np.linalg.norm(Y[:3, -1])
            final_vel = np.linalg.norm(Y[3:, -1])
            J_value = result.fun
            
            print(f"Optimal control solved successfully!")
            print(f"Final position: {final_pos:.3f} m")
            print(f"Final velocity: {final_vel:.3f} m/s") 
            print(f"Performance index J: {J_value:.6f}")
        else:
            print(f"Optimization failed: {result.message}")
            # Fallback to LQR
            self._setup_lqr_fallback()
            
        return self.control_profile
    
    def _setup_lqr_fallback(self):
        """Fallback to LQR if optimization fails"""
        print("Using LQR fallback...")
        Q = np.diag([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])
        R = np.diag([1e3, 1e3, 1e3])
        P = solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ self.B.T @ P
        
        # Generate LQR control sequence
        self.control_profile = np.zeros((3, self.N))
        Y = np.zeros((6, self.N))
        Y[:, 0] = self.y0
        
        for k in range(self.N):
            u = -K @ Y[:, k]
            u_norm = np.linalg.norm(u)
            if u_norm > self.max_thrust:
                u = u * (self.max_thrust / u_norm)
            self.control_profile[:, k] = u
            
            if k < self.N-1:
                # Simple Euler integration for fallback
                Y[:, k+1] = Y[:, k] + self.dt * self.dynamics(Y[:, k], u)
        
        self.optimized = True
    
    def get_control(self, t):
        """Get optimal control at time t"""
        if not self.optimized:
            return np.zeros(3)
            
        # Find nearest time index
        idx = min(int(t / self.dt), self.N-1)
        return self.control_profile[:, idx]

# Initialize and optimize:
optimal_controller = OptimalControlDocking(times_docking, y0, max_thrust)
optimal_controller.optimize_control()

# Usage:
def thrust_func_optimal(t, y):
    """Optimal control wrapper function"""
    return optimal_controller.get_control(t)

# ===========================================================================
# SELECT ACTIVE CONTROLLER HERE
# ===========================================================================

# thrust_func = thrust_func_pd                    # Method 1: PD Controller
# thrust_func = thrust_func_lqr                    # Method 2: Standard LQR  
# thrust_func = thrust_func_lqr_fuel_optimized    # Method 3: Fuel-optimized LQR
# thrust_func = thrust_func_tvlqr                 # Method 4: Time-varying LQR
thrust_func = thrust_func_smc                   # Method 5: Sliding Mode Control
# thrust_func = thrust_func_optimal               # Method 6: Optimal Control

# ===========================================================================
# DYNAMICS AND SIMULATION (UNCHANGED)
# ===========================================================================

def dy_dt(t, y):
    thrust = thrust_func(t, y)
    
    # Direct implementation of CW equations
    x, y_pos, z, vx, vy, vz = y
    
    dxdt = vx
    dydt = vy  
    dzdt = vz
    dvxdt = 3*n**2*x + 2*n*vy + thrust[0]
    dvydt = -2*n*vx + thrust[1]
    dvzdt = -n**2*z + thrust[2]
    
    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])




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




# Pre-compute optimal control if using that method
if thrust_func == thrust_func_optimal:
    optimal_controller.optimize_control(y0, times_docking)

print("Starting docking simulation...")
print(f"Using controller: {thrust_func.__name__}")

# Simulate docking using ABM4
y_abm4_docking, _ = abm4_all_step(h_docking, times_docking, dy_dt, y0)

# Extract results
positions = y_abm4_docking[:, :3]
velocities = y_abm4_docking[:, 3:]

# Calculate thrust profile and performance metrics
thrust_profile = np.zeros((len(times_docking), 3))
distances = np.zeros(len(times_docking))
rel_velocities = np.zeros(len(times_docking))
performance_index = np.zeros(len(times_docking))

total_J = 0
for i in range(len(times_docking)):
    thrust_profile[i] = thrust_func(times_docking[i], y_abm4_docking[i])
    distances[i] = np.linalg.norm(positions[i])
    rel_velocities[i] = np.linalg.norm(velocities[i])
    
    # Trapezoidal integration for performance index
    if i > 0:
        dt = times_docking[i] - times_docking[i-1]
        u_prev = thrust_profile[i-1]
        u_curr = thrust_profile[i]
        total_J += 0.5 * dt * (np.dot(u_prev, u_prev) + np.dot(u_curr, u_curr))
    performance_index[i] = total_J

# Check docking success
final_distance = distances[-1]
final_velocity = rel_velocities[-1]
docking_success = (final_distance < 10) and (final_velocity < 0.5)

print(f"\nDocking Results:")
print(f"Final distance: {final_distance:.3f} m")
print(f"Final velocity: {final_velocity:.3f} m/s")
print(f"Docking successful: {'YES' if docking_success else 'NO'}")
print(f"Total performance index J: {performance_index[-1]:.6f}")

# Create all required plots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Applied acceleration vs time
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(times_docking/60, thrust_profile[:, 0], label='$u_x$')
ax1.plot(times_docking/60, thrust_profile[:, 1], label='$u_y$')
ax1.plot(times_docking/60, thrust_profile[:, 2], label='$u_z$')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.set_title('Applied Acceleration vs Time')
ax1.legend()
ax1.grid(True)

# Plot 2: Distance vs time
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(times_docking/T_orb, distances)
ax2.axhline(y=10, color='r', linestyle='--', label='Docking threshold (10m)')
ax2.set_xlabel('Time (orbits)')
ax2.set_ylabel('Distance (m)')
ax2.set_title('Distance to ISS vs Time')
ax2.grid(True)
ax2.legend()

# Plot 3: Relative velocity vs time
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(times_docking/T_orb, rel_velocities)
ax3.axhline(y=0.5, color='r', linestyle='--', label='Docking threshold (0.5 m/s)')
ax3.set_xlabel('Time (orbits)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Relative Velocity vs Time')
ax3.grid(True)
ax3.legend()

# Plot 4: Performance index vs time
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(times_docking/T_orb, performance_index)
ax4.set_xlabel('Time (orbits)')
ax4.set_ylabel('Performance Index J')
ax4.set_title('Performance Index vs Time')
ax4.grid(True)

# 3D trajectory plot
ax5 = fig.add_subplot(2, 3, (5, 6), projection='3d')
ax5.plot(positions[:, 0]/1000, positions[:, 1]/1000, positions[:, 2]/1000, 
         'b-', linewidth=2, label='Ceres trajectory')
ax5.scatter(0, 0, 0, color='red', marker='*', s=200, label='ISS')
# Plot the final approach (last 100 points) in red to show docking
final_points = min(100, len(positions))
ax5.plot(positions[-final_points:, 0]/1000, positions[-final_points:, 1]/1000, 
         positions[-final_points:, 2]/1000, 'r-', linewidth=2, label='Final approach')
ax5.set_xlabel('x (km)')
ax5.set_ylabel('y (km)')
ax5.set_zlabel('z (km)')
ax5.set_title('3D Docking Trajectory')
ax5.legend()

plt.tight_layout()
plt.show()