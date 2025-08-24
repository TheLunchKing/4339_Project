import numpy as np


class TwoBody:
    # should input departure velocity
    def __init__(self, r: np.array, v: np.array, mu: float) -> None:
        self.mu = mu
        self.__calc_orbit_information(r=r, v=v)

    # private function
    def __calc_orbit_information(self, r: np.array, v: np.array) -> None:
        self.h = np.cross(r, v)
        h_norm = np.linalg.norm(self.h)
        self.orbit_normal = self.h / h_norm
        self.inclination = np.arccos(self.orbit_normal[2])
        self.p = h_norm**2 / self.mu
        self.r_norm = np.linalg.norm(r)
        self.v_norm = np.linalg.norm(v)
        self.r_v_dir_diff = np.arcsin(h_norm / (self.r_norm * self.v_norm))
        self.energy = self.v_norm**2 - 2 * self.mu / self.r_norm
        self.f = np.cross(v, self.h) - self.mu / self.r_norm * r
        # self.e = np.sqrt(1 + (h_norm / self.mu) ** 2 * self.energy)
        self.e = np.linalg.norm(self.f) / self.mu
        if self.e == 0:  # circle orbit
            self.a = self.r_norm / 2
            self.rp = self.a
            self.ra = self.rp
            self.period = 2 * np.pi * self.r_norm / self.v_norm
        else:
            self.a = self.p / (1 - self.e**2)
            self.rp = self.a * (1 - self.e)
            self.ra = self.a * (1 + self.e)
            self.period = 2 * np.pi * np.sqrt(self.a**3 / self.mu)  # period [s]

        self.mean_motion = 2 * np.pi / self.period
        cos_theta = (self.p - self.r_norm) / (self.e * self.r_norm)
        E = np.arccos((self.e + cos_theta) / (1 + self.e * cos_theta))
        self.p_vec = self.f / np.linalg.norm(self.f)
        self.w_vec = self.orbit_normal
        self.q_vec = np.cross(self.w_vec, self.p_vec)
        # Check the direction
        if (np.dot(r, self.q_vec) < 0):
            E = 2 * np.pi - E

        self.t0_tp = (E - self.e * np.sin(E)) / self.mean_motion
        # print("initial E: ", E)

    def calc_states(self, t: float) -> tuple:
        M = (t + self.t0_tp) * self.mean_motion
        if (M >= 2* np.pi):
            M -= 2 * np.pi * (M // (2 * np.pi))
        # Solve Kepler equation
        E = np.pi  # initial value
        while (np.fabs(self.KeplerEq(E, M)) > 1e-15):
            E = E - self.KeplerEq(E, M) / self.KeplerEqDot(E)

        theta = np.arccos((np.cos(E) - self.e) / (1 - self.e * np.cos(E)))
        if (M > np.pi):
            theta = 2 * np.pi - theta
        r = self.a * (1 - self.e * np.cos(E))
        pos_inplane = np.array([r * np.cos(theta), r * np.sin(theta), 0])
        v = np.sqrt(self.mu * (2.0 / r - 1.0 / self.a))
        sin_gamma = np.linalg.norm(self.h) / (r * v)
        # for numerical error
        if (sin_gamma > 1):
            sin_gamma = 1
        elif (sin_gamma < -1):
            sin_gamma = -1
        gamma = np.arcsin(sin_gamma)  # angle between r and v vector
        if (theta > np.pi):
            gamma = np.pi - gamma
        vel_inplane = np.array([v * np.cos(theta + gamma), v * np.sin(theta + gamma), 0])
        # if (t == 3500):
        #     print("pos, vel: ", pos_inplane, vel_inplane)

        p = self.f / np.linalg.norm(self.f)
        w = self.orbit_normal
        q = np.cross(w, p)
        # q /= np.linalg.norm(q)
        dcm_inplane_to_frame = np.array([p, q, w]).T

        pos = dcm_inplane_to_frame @ pos_inplane
        vel = dcm_inplane_to_frame @ vel_inplane
        return pos, vel

    def KeplerEq(self, E: float, M: float)-> float:
        return E - self.e * np.sin(E) - M

    def KeplerEqDot(self, E: float)-> float:
        return 1 - self.e * np.cos(E)

    # def __calc_additional_orbital_elements(self) -> None:
    #     # lan_sc: vector to ascending node of sc orbit in ECLIP
    #     z_ECLIP = np.array([0, 0, 1])
    #     lan_sc = np.cross(z_ECLIP, self.orbit_normal)
    #     self.an_dir = lan_sc / np.linalg.norm(lan_sc)

    #     orthogonal_lan_in_trj_plane = np.cross(self.orbit_normal, self.an_dir)
    #     self.trans_ECLIP_to_orbital_plane_along_AN = np.array(
    #         [self.an_dir, orthogonal_lan_in_trj_plane, self.orbit_normal]
    #     )
    #     f_dep_in_trj_plane = self.trans_ECLIP_to_orbital_plane_along_AN @ self.f

    #     cos_omega = np.dot(f_dep_in_trj_plane, np.array([1, 0, 0])) / np.linalg.norm(
    #         self.f
    #     )
    #     # to calculate argument of periapsis, should consider sign.
    #     if f_dep_in_trj_plane[1] > 0:
    #         self.omega = np.arccos(cos_omega)
    #     else:
    #         self.omega = 2 * np.pi - np.arccos(cos_omega)

    #     self.r_AN = self.p / (1 + self.e * np.cos(-self.omega))
    #     self.r_DN = self.p / (1 + self.e * np.cos(np.pi - self.omega))

    #     self.rp_dir = self.f / np.linalg.norm(self.f)
    #     orthogonal_rp_in_trj_plane = np.cross(self.orbit_normal, self.rp_dir)
    #     self.trans_ECLIP_to_orbital_plane_along_rp = np.array(
    #         [self.rp_dir, orthogonal_rp_in_trj_plane, self.orbit_normal]
    #     )
