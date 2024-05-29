import numpy as np
import sys


class Angular_velocity_sensor:
    omega_measured_last = np.zeros(3)
    t_last = 0

    def __init__(
        self,
        nominal_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        std_e: float,
        delta_time: float,
    ):
        """
        :param nominal_matrix: Номинальная матрица ДУС
        :param covariance_matrix: матрица ковариаций ДУС
        :param std_e: Стандартное отклонение шума в измерениях датчика
        :param delta_time: шаг измерений ДУС по времени
        """

        if np.linalg.matrix_rank(nominal_matrix) < 3:
            print("Номинальная матрица ДУС задана неверно")
            sys.exit()

        self.nominal_matrix = nominal_matrix

        real_matrix = np.array(
            [nominal_matrix[i] + np.random.normal(0, std_e, size=3) for i in range(3)]
        )

        self.matrix_nom2real = np.linalg.inv(nominal_matrix) @ real_matrix

        self.covariance_matrix = covariance_matrix
        self.delta_time = delta_time

    def omega_measured(self, t: float, omega_real: np.ndarray) -> np.ndarray:
        """
        :param t: Текущее время
        :param omega_real: Реальная угловая скорость в ССК
        :return: Измерения угловой скорости в ССК
        """

        if t - self.t_last >= self.delta_time:
            noise = np.random.multivariate_normal(np.zeros(3), self.covariance_matrix)

            omega_measured = self.matrix_nom_real @ (omega_real + noise)

            self.omega_measured_last = omega_measured
            self.t_last = t

            return omega_measured

        else:
            return self.omega_measured_last
