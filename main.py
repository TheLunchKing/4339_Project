
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp


mu = 3.8986 * 10 ** 5 # km^3/s^2
a_iss = 6770  # Semi-major plotis of ISS in meters
n = np.sqrt((mu)/a_iss**3)  # Orbital rate (rad/s)

T_orb = 2 * np.pi / n  # Orbital period

y0 = np.array([10000, 10000, 10000, 0, 0, 0]) #km, km/s

thrust = None
def thrust_func(t, y):
    return np.array([10, 0.0, 0.0])

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



sol = sp.solve_ivp(
                    fun = dy_dt,
                    t_span = (0,3*T_orb),
                    y0 = y0,
                    method = 'BDF'
                    )

print(sol)

# Extract the results
states = sol.y.T  # Each row is [x, y, z, ux, uy, uz] at a time step
times = sol.t

# Convert positions to km for plotting
x_km = states[:, 0] / 1000
y_km = states[:, 1] / 1000
z_km = states[:, 2] / 1000

# Plot the 3D graph
figure = plt.figure(figsize=(10, 10))
plot = figure.add_subplot(111, projection='3d')
plot.plot(x_km, y_km, z_km, label='Ceres path')
plot.scatter(0, 0, 0, color='red', marker='*', s=50, label='ISS')
plot.set_xlabel('x (km)')
plot.set_ylabel('y (km)')
plot.set_zlabel('z (km)')
plot.set_title('Milestone 1A: Trajectory of Ceres relative to ISS (no thrust)')
plot.legend()
plt.show()