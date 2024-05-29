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

    def __init__(
        self,
        Kw: float,
        Kq: float,
        B: Quaternion,
        J: np.ndarray,
        mu: float,
        gravity: Gravity_J2,
        atmoshpere: Atmosphere,
        angular_velocity: Angular_velocity_sensor | None,
        star_tracker: Star_tracker | None,
    ):
        self.Kw = Kw
        self.Kq = Kq
        self.B = B  # ИСК в опорную СК
        self.J = J
        self.mu = mu
