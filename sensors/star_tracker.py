import numpy as np
from pyquaternion import Quaternion
from typing import List
import sys


class Star_tracker:
    Q_measured_last = None
    t_last = 0

    def __init__(
        self, nominal_quat: Quaternion, covariance_matrix: np.ndarray, delta_time: float
    ):
        """
        :param nominal_quat: кватернион перехода из ССК в СК звездного датчика
        :param covariance_matrix: ковариационная матрица ЗД
        :param delta_time: шаг измерений ЗД по времени
        """

        self.nominal_quat = nominal_quat

        noise = np.random.multivariate_normal(np.zeros(3), covariance_matrix)

        delta_quat = Quaternion(scalar=(1 - np.dot(noise, noise)) ** 0.5, vector=noise)
        self.quat_real = nominal_quat * delta_quat

        self.covariance_matrix = covariance_matrix
        self.delta_time = delta_time

    def Q_measured(self, t: float, Q_real: Quaternion) -> np.ndarray:
        """
        :param t: текущее время
        :param Q_real: истинное значение кватерниона ориентации (ИСК -> ССК)
        :return: измеренный кватернион (ИСК -> ССК)
        """

        if t - self.t_last >= self.delta_time:
            noise = np.random.multivariate_normal(np.zeros(3), self.covariance_matrix)

            delta_quat = Quaternion(
                scalar=(1 - np.dot(noise, noise)) ** 0.5, vector=noise
            )

            Q_measured = (
                Q_real * self.quat_real * delta_quat * self.nominal_quat.conjugate
            )

            self.Q_measured_last = Q_measured
            self.t_last = t

            return Q_measured

        else:
            return self.Q_measured_last
