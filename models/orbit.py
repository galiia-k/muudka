import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, acos, atan2
from pyquaternion import Quaternion


class Orbit:
    result_t = None
    result_x = result_y = result_z = result_dotx = result_doty = result_dotz = (
        result_Wx
    ) = result_Wy = result_Wz = None
    result_a = result_e = result_i = result_w = result_omega = result_nu = None

    def __init__(
        self, a: float, e: float, i: float, w: float, omega: float, nu: float, mu: float
    ):
        self.a = a
        self.e = e
        self.i = i
        self.w = w
        self.omega = omega
        self.nu = nu
        self.mu = mu

    def kepler2rv(
        self, a: float, e: float, i: float, w: float, omega: float, nu: float
    ) -> np.ndarray:
        u = w + nu
        alpha = cos(omega) * cos(u) - sin(omega) * sin(u) * cos(i)
        beta = sin(omega) * cos(u) + cos(omega) * sin(u) * cos(i)
        gamma = sin(u) * sin(i)
        r_norm = a * (1 - e**2) / (1 + e * cos(nu))

        alpha_nu = -cos(omega) * sin(u) - sin(omega) * cos(u) * cos(i)
        beta_nu = -sin(omega) * sin(u) + cos(omega) * cos(u) * cos(i)
        gamma_nu = cos(u) * sin(i)
        p = a * (1 - e**2)
        coef1 = (self.mu / p) ** 0.5 * e * sin(nu)
        coef2 = (self.mu / p) ** 0.5 * (1 + e * cos(nu))
        return np.array(
            [
                alpha * r_norm,
                beta * r_norm,
                gamma * r_norm,
                coef1 * alpha + coef2 * alpha_nu,
                coef1 * beta + coef2 * beta_nu,
                coef1 * gamma + coef2 * gamma_nu,
            ]
        )

    def rv2kepler(self, rv: np.ndarray) -> np.ndarray:
        r = rv[:3]
        v = rv[3:6]
        c = np.cross(r, v)
        i = acos(np.dot(c, np.array([0, 0, 1])) / norm(c))
        l = np.cross(np.array([0, 0, 1]), c) / norm(np.cross(np.array([0, 0, 1]), c))
        big_omega = atan2(
            np.dot(l, np.array([0, 1, 0])), np.dot(l, np.array([1, 0, 0]))
        )
        n = np.cross(c, l) / norm(np.cross(c, l))
        f = np.cross(v, c) - self.mu / norm(r) * r
        omega = atan2(np.dot(f, n), np.dot(f, l))
        u = atan2(np.dot(r, n), np.dot(r, l))
        nu = u - omega
        p = norm(c) ** 2 / self.mu
        e = norm(f) / self.mu
        a = p / (1 - e**2)

        return np.array([a, e, i, omega, big_omega, nu])

    def add_results(self, t: np.ndarray, y: np.ndarray):
        self.result_t = t

        (
            self.result_x,
            self.result_y,
            self.result_z,
            self.result_dotx,
            self.result_doty,
            self.result_dotz,
        ) = y.T

        (
            self.result_a,
            self.result_e,
            self.result_i,
            self.result_w,
            self.result_omega,
            self.result_nu,
        ) = np.array(list(map(self.rv2kepler, y))).T
