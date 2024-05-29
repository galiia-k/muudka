import numpy as np
from numpy.linalg import matrix_rank, pinv
from typing import List


class Flywheel:
    H = t = omega = dot_omega = None

    def __init__(
        self,
        H0: np.ndarray,
        max_dotH: np.ndarray,
        list_J: List[np.ndarray],
        list_directions: List[np.ndarray],
    ):
        """
        :param H0: начальный момент маховиков в ССК
        :param max_dotH: максимально возможное изменение момента в ССК
        :param list_J: список тензоров инерции маховиков относительно центра масс каждого
            маховика в ССК.
        :param list_directions: список направлений угловой скорости маховиков
        """

        self.H0 = H0
        self.max_dotH = max_dotH
        self.list_J = list_J

        if matrix_rank(list_directions) == 3:
            self.list_directions_rotation = list_directions
        else:
            raise ValueError("Направления вращения маховиков указаны неверно")

    def momentum(self, dot_H: np.ndarray) -> np.ndarray:
        """
        :param dot_H: Запрашиваемый момент
        :return: Реальный момент (если запрашиваемый больше максимально возможного, возвращает максимально возможный)
        """
        return np.where(
            np.abs(dot_H) > self.max_dotH, np.sign(dot_H) * self.max_dotH, dot_H
        )

    def add_results(self, t: np.ndarray, H: np.ndarray):
        self.t = t
        self.H = H

    def calculate_rotation(self):
        H = self.H

        # создаем хранилище для угловой скорости и углового ускорения маховиков на i шаге
        self.omega = np.zeros((len(H), len(self.list_directions_rotation)))
        self.dot_omega = np.zeros((len(H), len(self.list_directions_rotation)))
        # Инициализируем пустой список для хранения столбцов матрицы A
        A_columns = []

        # Создаем матрицу A циклом
        for J, e in zip(self.list_J, self.list_directions_rotation):
            A_columns.append(J @ e)

        # Преобразуем список в матрицу и транспонируем ее,
        # после этого находим псевдообратную матрицу A+
        A_pseudo = pinv(np.vstack(A_columns).T)

        # Решаем систему уравнений X = A+ * H и добавляем результат в список self.omega
        for i in range(len(H)):
            self.omega[i] = np.dot(A_pseudo, H[i])

        # считаем ускорение
        for i in range(len(self.omega) - 1):
            self.dot_omega[i + 1] = (self.omega[i + 1] - self.omega[i]) / (
                self.t[i + 1] - self.t[i]
            )
