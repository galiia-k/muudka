import numpy as np
import ctypes
import os
from datetime import datetime, timedelta
import pymap3d as pm
from pyquaternion import Quaternion


class Straight_dipole:
    mu_e = 7.812 * 1e15  # км^3 * кг * с^-2 * А^-1 - постоянная земного магнетизма
    k_vec = np.array([0, 0, -1])

    def __init__(self):
        "init"

    def get_B(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        r_norm = np.linalg.norm(r)
        B_ics = -self.mu_e / r_norm ** 5 * (r_norm ** 2 * self.k_vec - 3 * np.dot(self.k_vec, r) * r)
        return Quaternion(x[9:13]).conjugate.rotate(B_ics)  # in attached coordinate system


class Inclined_dipole:
    mu_e = 7.812 * 1e15

    def __init__(self, start_time: datetime, delta: float, lmbd: float):
        '''
        :param start_time: starting modeling time
        :param delta: altitude of magnetic pole
        :param lmbd: longitude of magnetic pole
        '''
        self.start_time = start_time
        self.k_vec_ecef = np.array([np.sin(delta) * np.cos(lmbd), np.sin(delta) * np.sin(lmbd), np.cos(delta)])

    def get_B(self, x: np.ndarray, t: float) -> np.ndarray:
        current_datetime = self.start_datetime + timedelta(seconds=t)

        # Находим вектор k в ИСК
        k_vec_isk = np.concatenate(
            pm.ecef2eci(x=self.k_vec_ecef[0], y=self.k_vec_ecef[1], z=self.k_vec_ecef[2], time=current_datetime))
        r = x[0:3]
        nr = np.linalg.norm(r)
        Q = Quaternion(x[9:13])

        B_ics = -self.mu_e / nr ** 5 * (nr ** 2 * k_vec_isk - 3 * np.dot(k_vec_isk, r) * r)
        return Q.conjugate.rotate(B_ics)  # B in attached coordinate system
