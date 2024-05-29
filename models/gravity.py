import numpy as np
import os
from datetime import datetime, timedelta
import ephem
from pyquaternion import Quaternion


class Gravity_J2:
    R = 6378 * 1e3
    J2 = 1082.8 * 1e-6
    mu = 3.986 * 1e14

    def __init__(self, J: np.ndarray):
        self.J = J

    def acceleration(self, x: np.ndarray) -> np.ndarray:
        WJ2 = (
            3
            / 2
            * self.J2
            * self.mu
            * self.R**2
            / (np.linalg.norm(x[0:3])) ** 7
            * np.array(
                [
                    -x[0] * (x[0] ** 2 + x[1] ** 2 - 4 * x[2] ** 2),
                    -x[1] * (x[0] ** 2 + x[1] ** 2 - 4 * x[2] ** 2),
                    -3 * x[2] * (x[0] ** 2 + x[1] ** 2) + 2 * x[2] ** 3,
                ]
            )
        )
        result = -self.mu * x[0:3] / (np.linalg.norm(x[0:3])) ** 3 + WJ2
        return result

    def momentum(self, x: np.ndarray) -> np.ndarray:
        r_acs = Quaternion(x[9:13]).conjugate.rotate(x[:3])
        return (
            3 * self.mu / (np.linalg.norm(r_acs) ** 5) * np.cross(r_acs, self.J @ r_acs)
        )
