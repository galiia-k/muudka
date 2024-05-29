import numpy as np
from pyquaternion import Quaternion


class First_integrals:
    result_c = result_E = result_f = result_h = result_t = None

    def __init__(self, mu: float, J: np.ndarray):
        self.mu = mu
        self.J = J

    def calculate(self, t: np.ndarray, x: np.ndarray):
        """
        :param t: время
        :param x: массив векторов состояния (r, v, omega, q), r, v - ИСК, q, omega - ССК
        :return: первые интегралы E, c, f, интеграл Якоби
        """
        self.result_t = t
        r = x[:, :3]
        v = x[:, 3:6]
        omega = x[:, 6:9]

        self.result_c = np.cross(r, v)
        self.result_E = np.linalg.norm(v, axis=1) ** 2 / 2 - self.mu / np.linalg.norm(
            r, axis=1
        )
        self.result_f = (
            np.cross(v, self.result_c)
            - self.mu * r / np.linalg.norm(r, axis=1)[:, np.newaxis]
        )

        self.result_h = np.zeros(len(r))

        for i in range(len(r)):
            Q = Quaternion(x[i][9:])
            r_acs = Q.conjugate.rotate(r[i])
            v_acs = Q.conjugate.rotate(v[i])
            omega0_acs = np.cross(r_acs, v_acs) / (np.linalg.norm(r_acs) ** 2)
            self.result_h[i] = (
                np.dot(omega[i], self.J @ omega[i]) / 2
                + 3
                * self.mu
                * np.dot(r_acs, self.J @ r_acs)
                / (2 * np.linalg.norm(r_acs) ** 5)
                - np.dot(omega[i], self.J @ omega0_acs)
            )
