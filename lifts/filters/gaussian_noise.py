from filter import Filter

import numpy as np

class GaussianNoise(Filter):

    state_filter = True

    def __init__(self, stddev):
        super().__init__()

        self.stddev = stddev

    def apply(self, input):
        return np.random.normal(input, np.ones_like(input) * self.stddev)