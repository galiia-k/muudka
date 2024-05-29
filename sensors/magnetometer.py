import numpy as np
import sys
from typing import Tuple


class Magnetometer:
    B_meas_last = np.zeros(3)  # Last measured magnetic field
    t_last = 0  # Time of last measurement

    def __init__(
        self,
        nominal_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        std_e: float,
        delta_time: float,
    ):
        """
        :param nominal_matrix: Номинальная матрица магнитометра
        :param covariance_matrix: Ковариационная матрица шумов
        :param std_e: Стандартное отклонение шума в измерениях датчика
        :param delta_time: Промежуток между измерениями магнитометра
        """

        if (
            np.linalg.matrix_rank(nominal_matrix) < 3
        ):  # Проверка того, что оси не компланарны
            print("Номинальная матрица магнитометра задана неверно")
            sys.exit()

        self.nominal_matrix = nominal_matrix

        matrix_real = np.array(
            [nominal_matrix[i] + np.random.normal(0, std_e, size=3) for i in range(3)]
        )

        self.matrix_nom_real = (
            np.linalg.inv(nominal_matrix) @ matrix_real
        )  # матрица перехода ССК-магнитометр-ССК

        self.covariance_matrix = covariance_matrix
        self.delta_time = delta_time

    def B_measured(self, t: float, B_real: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Возвращает измеренную величину магнитного поля Земли и время измерения
        :param t: текущее время
        :param B_real: Реальное значение магнитного поля Земли в ССК
        """

        if t - self.t_last >= self.delta_time:
            noise = np.random.multivariate_normal(np.zeros(3), self.covariance_matrix)

            B_measured = self.matrix_nom_real @ (B_real + noise)

            self.B_measured_last = B_measured
            self.t_last = t

            return self.B_measured_last, t

        else:
            return self.B_measured_last, self.t_last
