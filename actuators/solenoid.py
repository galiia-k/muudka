import numpy as np
from typing import List
import sys


class Solenoid:
    result_m = result_t = result_I = None

    def __init__(self, max_m: np.ndarray, list_directions: List[np.ndarray]):
        '''
        :param max_m: максимально возможный магнитный момент в ССК
        :param list_directions: направления расположения катушки (??? хз че это такое)
        '''
        self.max_m = max_m

        if np.linalg.matrix_rank(list_directions) != 3:
            print("Направления катушек указаны неверно")
            sys.exit()

        else:
            self.list_directions = list_directions


    def momentum(self, m: np.ndarray) -> np.ndarray:
        '''
        :param m: Запрашиваемый момент для управления
        :return: Реальный момент (если запрашиваемый больше максимально возможного, возвращает максимально возможный)
        '''
        return np.where(np.abs(m) > self.max_m, np.sign(m) * self.max_m, m)


    def add_results(self, t: np.ndarray, m: np.ndarray):
        self.result_m = m
        self.result_t = t


    def required_I(self):
        '''
        :return: Требуемое значение силы тока для катушек
        '''
        m = self.result_m

        self.result_I = np.zeros((len(m), len(self.list_directions)))
        A_columns = self.list_directions

        A = np.vstack(A_columns).T
        A_pseudo = np.linalg.pinv(A)

        for i in range(len(m)):
            self.result_I[i] = np.dot(A_pseudo, m[i])
