import sys

import numpy as np
from numpy.linalg import inv, norm
from pyquaternion import Quaternion

from models.orbit import Orbit
from models.rotation import Rotation
from models.gravity import Gravity_ordinary, Gravity_J2, Gravity_harmonics
from models.geomagnetism import Direct_dipole, Inclined_dipole, Geomagnetic_harmonics
from models.atmosphere import Aerodynamics
from models.control import Control_orbit, Control_inertial, Control_bdot
from actuators.flywheel import Flywheel
from actuators.solenoid import Solenoids
from models.first_integrals import First_integrals


class Simulation:
    t0 = 0
    y0 = None

    def __init__(
        self,
        orbit: Orbit,
        rotation: Rotation,
        gravity: Gravity_ordinary | Gravity_J2 | Gravity_harmonics,
        aerodynamics: Aerodynamics | None,
        geomagnetic_field: (
            Direct_dipole | Inclined_dipole | Geomagnetic_harmonics | None
        ),
        control: Control_orbit | Control_inertial | None,
        m_bdot: Control_bdot | None,
        flywheel: Flywheel | None,
        solenoids: Solenoids | None,
        first_integrals: First_integrals | None,
    ):
        """
        Класс `Simulation` используется для моделирования движения космического объекта в пространстве, учитывая
        параметры, такие как орбитальные элементы, гравитация, аэродинамика, управление и маховики. После инициализации
        параметров симуляции, метод `start` запускает симуляцию, решая дифференциальные уравнения движения с помощью
        численных методов. Класс обрабатывает различные аспекты движения объекта, такие как вычисление управления,
        вращение, моменты сил и интегрирование первых интегралов, предоставляя средство для анализа поведения объекта в
        различных условиях и с различными параметрами.
            :param orbit: объект класса `Orbit`, описывающий орбиту, по которой происходит движение.
            :param rotation: объект класса `Rotation`, определяющий вращение объекта.
            :param gravity: объект класса `Gravity_ordinary`, `Gravity_J2`, или `Gravity_harmonics`, представляющий
                            гравитационную модель для симуляции.
            :param geomagnetic_field: Модель для получения магнитной индукции. Может быть одной из типов:
                                      Direct_dipole, Inclined_dipole, Geomagnetic_harmonics или None
            :param aerodynamics: объект класса `Aerodynamics` или `None`, описывающий аэродинамическую модель если нужно
            :param control: объект класса `Control_orbit`, `Control_inertial` или `None`, представляющий управления
            :param m_bdot: объект класса `Control_bdot` или `None`, представляющий модель управления Bdot
            :param flywheel: объект класса `Flywheel` или `None`, представляющий маховик.
            :param solenoids: объект класса `Solenoids` или `None`, представляющий систему катушек.
            :param first_integrals: объект класса `First_integrals` или `None`, содержащий первые интегралы
        """
        self.orbit = orbit
        self.rotation = rotation
        self.gravity = gravity
        self.aerodynamics = aerodynamics
        self.geomagnetic_field = geomagnetic_field
        self.control = control
        self.m_bdot = m_bdot

        if m_bdot is not None and control is not None:
            raise ValueError("m_bdot и control переданы одновременно")

        if m_bdot is not None and geomagnetic_field is None:
            raise ValueError("m_bdot и control переданы одновременно")

        self.flywheel = flywheel
        self.solenoids = solenoids
        self.first_integrals = first_integrals

        self.inv_J = inv(rotation.J)

        # добавляем начальные данные
        y0 = self.orbit.kepler2rv(
            a=orbit.a, e=orbit.e, i=orbit.i, w=orbit.w, omega=orbit.omega, nu=orbit.nu
        )
        self.y0 = (
            np.concatenate((y0, rotation.W0, rotation.Q0.elements))
            if self.flywheel is None
            else np.concatenate((y0, rotation.W0, rotation.Q0.elements, flywheel.H0))
        )

    def vector_function_right_parts(
        self, t: float, y: np.ndarray, U: np.ndarray, m_bdot: np.ndarray
    ) -> np.ndarray:
        """
        Рассчитывает правые части векторной функции для задачи двух тел и вращения КА

            :param m_bdot: NumPy массив, магнитного момента катушек в ССК
            :param U: NumPy массив, требуемое управление в ССК
            :param t: Время (секунды).
            :param y: NumPy массив, содержащий текущие значения вектора r (y[0:3]) и вектора v (y[3:6]) в ИСК,
             w (y[6:9]), Q (y[9:13]) и H (y[13:16] - если нужно) в ССК.

        :return: NumPy массив, содержащий правые части дифференциальных уравнений задачи двух тел
        """
        right_parts = np.zeros(len(y))

        # Первые 6 уравнений на координаты и скорости
        right_parts[0:3] = y[3:6]
        right_parts[3:6] = self.gravity.get_acceleration(y=y, t=t) + (
            self.aerodynamics.get_acceleration(y=y, t=t)
            if self.aerodynamics is not None
            else 0
        )

        # уравнение на Кватернион
        right_parts[9:13] = np.array(
            (0.5 * Quaternion(y[9:13]) * Quaternion(vector=y[6:9])).elements
        )

        # Ищем момент от внешних сил
        M_ext = self.gravity.get_moment(y=y, t=t) + (
            self.aerodynamics.get_moment(y=y, t=t)
            if self.aerodynamics is not None
            else 0
        )

        J_dotW = -np.cross(y[6:9], self.rotation.J @ y[6:9])
        mag_moment = (
            np.cross(m_bdot, self.geomagnetic_field.get_B(t=t, y=y))
            if self.m_bdot is not None
            else 0
        )
        if self.flywheel is not None:
            right_parts[13:16] = self.flywheel.checking_max_dotH(
                t=t, dot_H=-np.cross(y[6:9], y[13:16]) - U
            )
            U = -right_parts[13:16] - np.cross(y[6:9], y[13:16])

        right_parts[6:9] = self.inv_J @ (J_dotW + M_ext + U + mag_moment)
        return right_parts

    def runge_kutta_4(self, f, t0: float, y0: np.ndarray, h, n: int):
        """
        Метод Рунге-Кутта 4-го порядка для решения системы дифференциальных уравнений

        Аргументы:
        f: функция, описывающая систему дифференциальных уравнений
           Принимает аргументы t и y и возвращает массив значений производных dy/dx
        t0: начальное значение t
        y0: начальное значение y (массив значений)
        h: шаг интегрирования
        n: количество шагов
        """

        # Создаем массивы для хранения результатов
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))
        U = np.zeros((n + 1, 3))
        m_bdot = np.zeros((n + 1, 3))

        # Записываем начальные значения
        t[0] = t0
        y[0] = y0

        # Кэшируем проверки на None
        control_get = (
            self.control.get if self.control is not None else lambda y, t: np.zeros(3)
        )
        m_bdot_get = (
            self.m_bdot.get if self.m_bdot is not None else lambda y, t: np.zeros(3)
        )
        check_B = (
            self.solenoids.checking_max_m
            if self.solenoids is not None
            else lambda m, t: m
        )

        # Итерационный процесс метода Рунге-Кутта 4-го порядка
        for i in range(n):
            # получаем управление
            U[i] = control_get(y=y[i], t=t[i])
            m_bdot[i] = check_B(m=m_bdot_get(y=y[i], t=t[i]), t=t[i])
            k1 = f(t[i], y[i], U[i], m_bdot[i])
            k2 = f(t[i] + h / 2, y[i] + h / 2 * k1, U[i], m_bdot[i])
            k3 = f(t[i] + h / 2, y[i] + h / 2 * k2, U[i], m_bdot[i])
            k4 = f(t[i] + h, y[i] + h * k3, U[i], m_bdot[i])

            t[i + 1] = t[i] + h
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6

            # нормировка кватерниона в виде
            y[i + 1][9:13] = np.array(Quaternion(y[i + 1][9:13]).normalised.elements)

            if i % 12000 == 0:  # Если i делится на 10 без остатка
                print(norm(y[i][6:9]))

        return t, y, U, m_bdot

    def start(self, h: float, n: int):
        t, y, U, m_bdot = self.runge_kutta_4(
            f=self.vector_function_right_parts, t0=self.t0, y0=self.y0, h=h, n=n
        )

        # добавляем результаты орбитального движения
        self.orbit.add_results(t=t, y=y[:, 0:6])

        # добавляем результаты углового движения в зависимости от управления
        if self.control is None:
            self.rotation.add_results(
                t=t, y=y, type_control="orbit"
            )  # смотрим относительно орбитальной с-мы координат
        elif isinstance(self.control, Control_orbit):
            self.rotation.add_results(t=t, y=y, type_control="orbit")
        elif isinstance(self.control, Control_inertial):
            self.rotation.add_results(
                t=t, y=y, type_control="inertial", B_inertial=self.control.B
            )

        # Сохраняем управление
        if self.control is not None:
            self.control.add_results(t=t, U=U)
        if self.m_bdot is not None:
            self.m_bdot.add_results(t=t, m=m_bdot)

        #  добавляем результаты кин моментов и считаем требуемое значение угловой скорости для маховиков
        if self.flywheel is not None:
            self.flywheel.add_results(H=y[:, 13:16], t=t)
            self.flywheel.calculate_required_rotation()

        #  добавляем результаты магнитного момента катушек и рассчитываем силу тока на них
        if self.solenoids is not None:
            self.solenoids.add_results(m=m_bdot, t=t)
            self.solenoids.calculate_required_I()

        # Считаем первые интегралы
        if self.first_integrals is not None:
            self.first_integrals.calculate(t=t, y=y)
