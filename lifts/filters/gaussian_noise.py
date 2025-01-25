from lifts.filters.filter import Filter

import numpy as np
from numpy.typing import NDArray
from typing import Dict

class GaussianNoise(Filter):

    state_filter = True

    def __init__(self, stddev):
        super().__init__()

        self.stddev = stddev

    def apply(self, input: Dict[str, NDArray]):

        for key in input.keys():
            input[key] = np.random.normal(input[key], np.ones_like(input[key]) * self.stddev)

        return input