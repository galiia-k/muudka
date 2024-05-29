import numpy as np
from pyquaternion import Quaternion

from sensors.angular_velocity import Angular_velocity_sensor
from sensors.star_tracker import Star_tracker
from sensors.magnetometer import Magnetometer
from .gravity import Gravity_J2
from .atmosphere import Atmosphere
from .geomagnetism import Straight_dipole, Inclined_dipole


class Control_inertial:
    omega_last = omega_now = 0
    t_last = t_now = 0
    U = np.zeros(3)
    res_U = res_t = None

    def __init__(self, Kw: float, Kq: float, B: Quaternion, J: np.ndarray, mu: float,
                 gravity: Gravity_J2, atmoshpere: Atmosphere,
                 av_sensor: Angular_velocity_sensor | None, star_tracker: Star_tracker | None):
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
        '''
        :return: управление на ССК
        '''
        r = x[:3]
        v = x[3:6]
        omega_acs = self.av_sensor.omega_measured(t, x[6:9]) if self.av_sensor is not None else omega_acs = x[6:9]
        Q = self.star_tracker.Q_measured(t, x[9:]) if self.star_tracker is not None else Q = Quaternion(x[9:])

        M_ext = self.gravity.momentum(x)
        if self.atmoshpere is not None:
            M_ext += self.atmoshpere.momentum(x)

