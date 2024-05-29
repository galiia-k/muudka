import numpy as np
from pyquaternion import Quaternion


class Atmosphere:

    def __init__(self, p: float, r_c: np.ndarray, S: float, e: float, U_div_V0: float, m: float, n: np.ndarray):
        self.p = p  # atmosphere's density
        self.r_c = r_c  # ACS: difference between mass center and geometric center (vector)
        self.S = S  # plate area
        self.e = e  # reflection coefficient
        self.U_div_V0 = U_div_V0  # flow to satellite velocity ratio
        self.m = m  # satellite's mass
        self.n = n  # normal vector

    def acceleration(self, x: np.ndarray) -> np.ndarray:
        Q = Quaternion(x[9:13])
        v_acs = Q.conjugate.rotate(x[3:6])

        Ev = v_acs / np.linalg.norm(v_acs)
        Ev_n = abs(np.dot(Ev, self.n))

        I1, I2, I3 = Ev_n * Ev, Ev_n ** 2 * self.n, Ev_n * self.n

        a_acs = -self.p * np.dot(v_acs, v_acs) * self.S / self.m * (
                (1 - self.e) * (I1 + self.U_div_V0 * I3) + 2 * self.e * I2)
        return Q.rotate(a_acs)  # back to ICS

    def momentum(self, x: np.ndarray) -> np.ndarray:
        Q = Quaternion(x[9:13])
        v_acs = Q.conjugate.rotate(x[3:6])

        Ev = v_acs / np.linalg.norm(v_acs)
        Ev_n = abs(np.dot(Ev, self.n))

        J1, J2, J3 = Ev_n * np.cross(Ev, self.r_c), Ev_n ** 2 * np.cross(self.n, self.r_c), \
                     Ev_n * np.cross(self.n, self.r_c)

        return -self.p * np.dot(v_acs, v_acs) * self.S * (
                (1 - self.e) * (J1 + self.U_div_V0 * J3) + 2 * self.e * J2)  # in ACS
