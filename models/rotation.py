import numpy as np
from numpy.linalg import norm
from pyquaternion import Quaternion


class Rotation:

    Wabs = Wx = Wy = Wz = None
    A_last = 0
    Q = None
    t = None

    W_rel = []
    A = []

    def __init__(self, J: np.ndarray, Q0: Quaternion, W0: np.ndarray):
        """
        Конструктор класса, который инициализирует объект ориентации для углвого движения с заданными параметрами

            :param J: Полный Тензор инерции всего спутника в ССК, если есть солнечная панель или маховики
            :param Q0: Ориентация ССK относительно ИСК в нач момент времени.
            :param W0: Угловая абсолютная начальная скорость в ССk.

        """
        self.J = J
        self.Q0 = Q0
        self.W0 = W0

    def add_calc_results(self, t: np.ndarray, y: np.ndarray, type_control: str, B_inertial: Quaternion | None = None):
        """
        Добавляет результаты расчетов для вращения

            :param B_inertial: Кватернион перехода из инерциальной системы координат в опорную систему координат,
                      параметр обязательный для type_control = inertial
            :param type_control: str , тип управления, который используется "orbit" | "inertial" | ""
            :param t: NumPy массив, t[i] - Время (секунды).
            :param y: NumPy массив состояний, содержащий значения вектора r иск (y[i][0:3]) и
            вектора v иск (y[i][3:6]), w сск (y[i][6:9]), Q из иск в сск (y[i][9:13]).
        """

        self.t = t
        self.Wx, self.Wy, self.Wz = y[:, 6:9].T
        self.Wabs = norm(y[:, 6:9], axis=1)

        self.Q = y[:, 9:13]

        self.W_rel = np.zeros((len(t), 3))
        self.A = np.zeros((len(t), 4))

        if type_control == "orbit":
            for i in range(len(t)):
                r = y[i][0:3]
                v = y[i][3:6]
                W_abs_cck = y[i][6:9]
                Q = Quaternion(y[i][9:13])

                W_ref_ock = np.array([0, norm(np.cross(r, v) / (norm(r) ** 2)), 0])

                # Ищем кватернион B (из ИСК в ОСК):
                e3 = r / norm(r)
                e2 = np.cross(r, v) / norm(np.cross(r, v))
                e1 = np.cross(e2, e3)
                K_matrix = np.array([e1, e2, e3])
                B_conj = Quaternion(matrix=K_matrix)

                # Кватернион A (из ОСК в ССК) и Wrel в ssk:
                A = B_conj * Q
                if A.scalar < 0:
                    A = -A
                W_rel_cck = W_abs_cck - A.conjugate.rotate(W_ref_ock)

                self.W_rel[i] = W_rel_cck
                self.A[i] = A.elements

        elif type_control == "inertial":
            for i in range(len(t)):
                W_abs_cck = y[i][6:9]
                Q = Quaternion(y[i][9:13])

                W_ref_ock = np.array([0, 0, 0])

                # Кватернион A (из ОСК в ССК) и Wrel в ssk:
                A = B_inertial.conjugate * Q
                if A.scalar < 0:
                    A = -A
                W_rel_cck = W_abs_cck - A.conjugate.rotate(W_ref_ock)

                self.W_rel[i] = W_rel_cck
                self.A[i] = A.elements












