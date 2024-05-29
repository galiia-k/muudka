import numpy as np
from typing import List
import sys


class Flywheel:
    result_H = result_t = result_omega = result_dot_omega = None

    def __init__(self, H0: np.ndarray, max_dotH: np.ndarray, list_J: List[np.ndarray],
                 list_directions: List[np.ndarray]):
        '''
        :param H0: начальный момент маховиков в ССК
        :param max_dotH: максимально возможное изменение момента в ССК
        :param list_J: список тензоров инерции маховиков относительно центра масс каждого
            маховика в ССК.
        :param list_directions: список направлений угловой скорости маховиков
        '''
        self.H0 = H0
        self.max_dotH = max_dotH
        self.list_J = list_J

        if np.linalg.matrix_rank(list_directions) == 3:
            self.list_directions_rotation = list_directions
        else:
            print("Направления вращения маховиков указаны неверно")
            sys.exit()


    def momentum(self, dot_H: np.ndarray) -> np.ndarray:
        '''
        :param dot_H: Запрашиваемый момент
        :return: Реальный момент (если запрашиваемый больше максимально возможного, возвращает максимально возможный)
        '''
        return np.where(np.abs(dot_H) > self.max_dotH, np.sign(dot_H) * self.max_dotH, dot_H)


    def add_results(self, t: np.ndarray, H: np.ndarray):
        self.result_t = t
        self.result_H = H


    #def required_rotation(self):


