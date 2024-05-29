import numpy as np
from typing import List


class Satellite:
    M_total = M_solar_panel = 0.0
    list_flywheel = []
    J_cm = J_cm_total = J_cm_solar_panel = J0_solar_panel = np.zeros((3, 3))
    coord0_solar_panel = coord_cm = np.zeros(3)

    def __init__(self, J0: np.ndarray, M0: float):
        """
        Инициализирует объект спутника.
            Параметры:
                J0 (np.ndarray): Тензор инерции спутника без маховиков и солнечной панели в ССК.
                M0 (float): Масса спутника без маховиков и солнечной панели.
        """
        self.J0 = J0
        self.M0 = M0

    def add_solar_panel(
        self,
        J0_solar_panel: np.ndarray,
        M_solar_panel: float,
        coord0_solar_panel: np.ndarray,
    ):
        """
        Добавляет солнечную панель к спутнику.
            Параметры:
                J0_solar_panel (np.ndarray): Тензор инерции солнечной панели относительно своего центра масс в ССК.
                M_solar_panel (float): Масса солнечной панели.
                coord0_solar_panel (np.ndarray): Координаты центра масс солнечной панели относительно центра масс
                                                 спутника без маховиков и солнечной панели.
        """
        self.J0_solar_panel = J0_solar_panel
        self.M_solar_panel = M_solar_panel
        self.coord0_solar_panel = coord0_solar_panel

    def add_flywheel(
        self, J0_flywheel: np.ndarray, M_flywheel: float, coord0_flywheel: np.ndarray
    ):
        """
        Добавляет маховик к спутнику.
            Параметры:
                   J0_flywheel (np.ndarray): Тензор инерции маховика относительно своего центра масс в ССК.
                   M_flywheel (float): Масса маховика.
                   coord0_flywheel (np.ndarray): Координаты центра масс маховика относительно центра масс спутника без
                                                 маховиков и солнечной панели.
        """
        self.list_flywheel.append(
            {
                "J0_flywheel": J0_flywheel,
                "M_flywheel": M_flywheel,
                "coord0_flywheel": coord0_flywheel,
            }
        )

    def calculate_cm_and_J(self):
        """
        Вычисляет центр масс и тензор инерции спутника со всеми компонентами.
        """
        # считаем обущю массу и координату центра масс
        self.M_total = self.M0 + self.M_solar_panel
        self.coord_cm = self.coord0_solar_panel * self.M_solar_panel / self.M_total
        for i in range(len(self.list_flywheel)):
            self.coord_cm = (
                self.coord_cm * self.M_total
                + self.list_flywheel[i]["coord0_flywheel"]
                * self.list_flywheel[i]["M_flywheel"]
            ) / (self.M_total + self.list_flywheel[i]["M_flywheel"])
            self.M_total += self.list_flywheel[i]["M_flywheel"]

        # пересчитываем тензоры инерции для каждой компоненты спутника
        r = self.coord_cm
        diff = lambda x: np.dot(x, x) * np.eye(3) - np.outer(x, x)
        self.J_cm = self.J0 + self.M0 * diff(r)
        self.J_cm_total = self.J_cm

        r = self.coord_cm - self.coord0_solar_panel
        self.J_cm_solar_panel = self.J0_solar_panel + self.M_solar_panel * diff(r)
        self.J_cm_total = self.J_cm_total + self.J_cm_solar_panel

        for i in range(len(self.list_flywheel)):
            r = self.coord_cm - self.list_flywheel[i]["coord0_flywheel"]
            self.list_flywheel[i]["J_cm_flywheel"] = self.list_flywheel[i][
                "J0_flywheel"
            ] + self.list_flywheel[i]["M_flywheel"] * diff(r)

            self.J_cm_total = self.J_cm_total + self.list_flywheel[i]["J_cm_flywheel"]

    def get_full_inertia_tensor(self) -> np.ndarray:
        return self.J_cm_total

    def get_total_mass(self) -> float:
        return self.M_total

    def get_list_J_cm_flywheel(self) -> List:
        """
        Возвращает список тензоров инерции маховиков в ССК относительно центра масс аппарата.
        """
        list_J_cm_flywheel = []
        for i in range(len(self.list_flywheel)):
            list_J_cm_flywheel.append(self.list_flywheel[i]["J_cm_flywheel"])

        return list_J_cm_flywheel

    def get_list_J0_flywheel(self) -> List:
        """
        Возвращает список тензоров инерции маховиков в ССК относительно центра масс каждого маховика.
        """
        list_J0_flywheel = []
        for i in range(len(self.list_flywheel)):
            list_J0_flywheel.append(self.list_flywheel[i]["J0_flywheel"])
        return list_J0_flywheel
