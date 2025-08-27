
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from two_body import TwoBody

mu = 3.8986 * 10 ** 5 # km^3/s^2

y0 = np.array([10, 10, 10, 0, 0, 0]) #km, km/s

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

print(dy_dt(0, y0, thrust))