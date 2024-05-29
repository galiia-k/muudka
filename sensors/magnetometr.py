import numpy as np
import sys
from typing import Tuple

class Magnetometr:
    B_meas_last = np.zeros(3)  # Last measured magnetic field
    t_last = 0  # Time of last measurement

    def __init__(self, nominal_matrix: np.ndarray, covariance_matrix: np.ndarray, std_e: float, delta_time: float):
        '''
        :param alpha1, alpha2, alpha3: Коэффициенты усиления по осям e1, e2, e3
        :param e1, e2, e3: Координаты осей датчика в ССК
        :param covariance_matrix: Ковариационная матрица шумов
        :param std_e: Стандартное отклонение шума в измерениях датчика
        :param delta_time: Промежуток между измерениями магнитометра
        '''

        matrix_nom = np.array([alpha1 * e1, alpha2 * e2, alpha3 * e3])
        if np.linalg.matrix_rank(matrix_nom) < 3:  # Проверка того, что оси не компланарны
            sys.exit()

        matrix_real = np.array([matrix_nom[i] + np.random.normal(0, std_e, size=3) for i in range(3)])

        self.matrix_nom_real = np.linalg.inv(matrix_nom) @ matrix_real  # матрица перехода ССК-магнитометр-ССК

        self.covariance_matrix = covariance_matrix
        self.delta_time = delta_time


    def B_measurement(self, t: float, B_real: np.ndarray) -> Tuple[np.ndarray, float]:
        '''
        Возвращает измеренную величину магнитного поля Земли и время измерения
        :param t: текущее время
        :param B_real: Реальное значение магнитного поля Земли в ССК
        '''

        if t - self.t_last >= self.delta_time:
            noise = np.random.multivariate_normal([0, 0, 0], self.covariance_matrix)

            B_measured = self.matrix_nom_real @ (B_real + noise)

            self.B_measured_last = B_measured
            self.t_last = t

            return self.B_measured_last, t

        else:
            return self.B_measured_last, self.t_last
