from lifts.filters.filter import Filter

import numpy as np

class MotorNoise(Filter):
    action_filter = True

    def __init__(self, stddev):
        self.stddev = stddev

    def apply(self, input):
        return np.random.normal(input, self.stddev, input.shape)