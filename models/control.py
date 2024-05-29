import numpy as np
from pyquaternion import Quaternion

from sensors.angular_velocity import Angular_velocity_sensor
from sensors.star_tracker import Star_tracker
from sensors.magnetometer import Magnetometer
from gravity import Gravity_J2
from atmosphere import Atmosphere
from geomagnetism import Straight_dipole, Inclined_dipole


class Control_inertial:
    omega_last = omega_now = 0
    t_last = t_now = 0
    U = np.zeros(3)
    res_U = res_t = None

    def __init__(
        self,
        Kw: float,
        Kq: float,
        B: Quaternion,
        J: np.ndarray,
        mu: float,
        gravity: Gravity_J2,
        atmoshpere: Atmosphere,
        av_sensor: Angular_velocity_sensor | None,
        star_tracker: Star_tracker | None,
    ):
        self.Kw = Kw
        self.Kq = Kq
        self.B = B  # ИСК в опорную СК
        self.J = J
        self.mu = mu
        self.gravity = gravity
        self.atmoshpere = atmoshpere
        self.av_sensor = av_sensor
        self.star_tracker = star_tracker

    def calculate_control(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        :return: управление на ССК
        """
        r = x[:3]
        v = x[3:6]
        omega_acs = (
            self.av_sensor.omega_measured(t, x[6:9])
            if self.av_sensor is not None
            else x[6:9]
        )
        Q = (
            self.star_tracker.Q_measured(t, x[9:])
            if self.star_tracker is not None
            else Quaternion(x[9:])
        )

        M_ext = self.gravity.momentum(x)
        if self.atmoshpere is not None:
            M_ext += self.atmoshpere.momentum(x)

        omega_rcs = np.zeros(3)

        # Кватернион A (из ОСК в ССК) и Wrel в ssk:
        A = self.B.conjugate * Q
        if A.scalar < 0:
            A = -A
        omega_acs_rel = omega_acs - A.conjugate.rotate(omega_rcs)

        # считаем численно производную требуемой угловой скорости в сск
        if t - self.t_last == 0:  # это для нулевого управления
            self.omega_last = np.cross(r, v) / (np.linalg.norm(r) ** 2)
            return np.zeros(3)
        self.omega_now = np.cross(r, v) / (np.linalg.norm(r) ** 2)
        self.t_now = t
        omega_dot_ics = (self.omega_now - self.omega_last) / (self.t_now - self.t_last)
        omega_dot_acs = Q.conjugate.rotate(omega_dot_ics)
        self.t_last = t
        self.omega_last = self.omega_now

        q = A.vector

        self.U = (
            -M_ext
            + np.cross(omega_acs, self.J @ omega_acs)
            - self.J @ np.cross(omega_acs_rel, A.conjugate.rotate(omega_rcs))
            + self.J @ omega_dot_acs
            - self.Kw * omega_acs_rel
            - self.Kq * q
        )

        return self.U

    def add_results(self, t: np.ndarray, U: np.ndarray):
        """
        Добавляет результаты расчетов управления
            :param t: NumPy массив, t[i] - Время (секунды).
            :param U: NumPy массив состояний, содержащий значения момента управления в сск (U[i])
        """
        self.result_t = t
        self.result_U = U

    class Control_orbit:
        W_ref_isk_last = W_ref_isk_now = 0
        t_last = t_now = 0
        U = np.array([0, 0, 0])
        result_U = None
        result_t = None

        def __init__(
            self,
            Kw: float,
            Kq: float,
            J: np.ndarray,
            mu: float,
            gravity: Gravity_J2,
            atmosphere: Atmosphere | None,
            angular_velocity_sensor: Angular_velocity_sensor | None,
            star_sensor: Star_tracker | None,
        ):
            self.Kw = Kw
            self.Kq = Kq
            self.J = J
            self.mu = mu
            self.gravity = gravity
            self.atmosphere = atmosphere
            self.angular_velocity_sensor = angular_velocity_sensor
            self.star_sensor = star_sensor

        def get(self, t: float, y: np.ndarray) -> np.ndarray:
            """
            Рассчитывает управление

                :param t: Время (секунды).
                :param y: NumPy массив, содержащий текущие значения вектора r иск (y[0:3]) и вектора v иск (y[3:6]),
                 w сск (y[6:9]), Q из иск в сск (y[9:13]).
                :param Mext: NumPy массив, содержащий внешний момент в сск.

            :return: NumPy массив, содержащий управления на Оси ССК
            """

            r = y[0:3]
            v = y[3:6]
            if self.angular_velocity_sensor is not None:
                # получаем измерение угловой скорости от ДУС
                W_abs_cck = self.angular_velocity_sensor.get_measured_w(
                    t=t, w_real=y[6:9]
                )
            else:
                W_abs_cck = y[6:9]
            if self.star_sensor is not None:
                # получаем измерение Q от звездного датчика
                Q = self.star_sensor.get_measured_Q(t=t, Q_real=y[9:13])
            else:
                Q = Quaternion(y[9:13])

            # Ищем момент от внешних сил
            M_ext = self.gravity.get_moment(y=y, t=t)
            if self.atmosphere is not None:
                M_ext += self.atmosphere.get_moment(y=y, t=t)

            W_ref_ock = np.array(
                [0, np.linalg.norm(np.cross(r, v) / (np.linalg.norm(r) ** 2)), 0]
            )

            # Ищем кватернион B (из ИСК в ОСК):
            e3 = r / np.linalg.norm(r)
            e2 = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
            e1 = np.cross(e2, e3)
            K_matrix = np.array([e1, e2, e3])
            B_conj = Quaternion(matrix=K_matrix)

            # Кватернион A (из ОСК в ССК) и Wrel в ssk:
            A = B_conj * Q
            if A.scalar < 0:
                A = -A
            W_rel_cck = W_abs_cck - A.conjugate.rotate(W_ref_ock)

            # считаем численно производную требуемой углвой скорости в сск
            if t - self.t_last == 0:  # это для нулевого управления
                self.W_ref_isk_last = np.cross(r, v) / (np.linalg.norm(r) ** 2)
                return np.array([0, 0, 0])
            self.W_ref_isk_now = np.cross(r, v) / (np.linalg.norm(r) ** 2)
            self.t_now = t
            dotW_ref_ick = (self.W_ref_isk_now - self.W_ref_isk_last) / (
                self.t_now - self.t_last
            )
            dotW_ref_cck = Q.conjugate.rotate(dotW_ref_ick)
            self.t_last = t
            self.W_ref_isk_last = self.W_ref_isk_now

            q = A.vector

            self.U = (
                -M_ext
                + np.cross(W_abs_cck, self.J @ W_abs_cck)
                - self.J @ np.cross(W_rel_cck, A.conjugate.rotate(W_ref_ock))
                + self.J @ dotW_ref_cck
                - self.Kw * W_rel_cck
                - self.Kq * q
            )

            return self.U

        def add_results(self, t: np.ndarray, U: np.ndarray):
            self.result_t = t
            self.result_U = U

    class Control_bdot:
        t_last = 0
        B_last = np.array([0, 0, 0])
        m_last = np.array([0, 0, 0])
        result_m = None
        result_t = None

        def __init__(
            self,
            k: float,
            geomagnetic_field: Straight_dipole | Inclined_dipole,
            magnetometer: Magnetometer | None,
        ):
            self.k = k
            self.geomagnetic_field = geomagnetic_field
            self.magnetometer = magnetometer

        def get(self, t: float, y: np.ndarray) -> np.ndarray:
            # получаем магнитную индукцию
            B = self.geomagnetic_field.get_B(y=y, t=t)

            # получаем измерение с магнитометра и проверяем на дискретность по времени
            if self.magnetometer is not None:
                B, t_measured = self.magnetometer.get_measured_B(B_real=B, t=t)
                if t_measured != t:
                    return self.m_last

            if t - self.t_last == 0:  # это для нулевого управления
                self.B_last = B
                return np.array([0, 0, 0])

            m = -self.k * (B - self.B_last) / (t - self.t_last)
            self.B_last = B
            self.t_last = t
            self.m_last = m

            return self.m_last

        def add_results(self, t: np.ndarray, m: np.ndarray):
            self.result_t = t
            self.result_m = m
