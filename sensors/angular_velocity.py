import numpy as np
import sys


class Angular_velocity_sensor:
    omega_measured_last = np.zeros(3)
    t_last = 0

    def __init__(self):