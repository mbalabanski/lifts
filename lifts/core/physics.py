import numpy as np
from numpy.typing import NDArray


def motor_actuator_to_thrust_angular_velocities(motor_actuator_values: NDArray):
    raise NotImplementedError()

def thrust_angular_velocities_to_motor_actuator(thrust_vel: NDArray, max_actuator=1.0):
    '''
    Converts thrust and angular velocities to motor actuator values
    '''

    raise NotImplementedError()