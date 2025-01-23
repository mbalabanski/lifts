from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class Filter(ABC):
    '''
    Class for filters to be applied to the simulator.

    A filter should indicate if it is a state filter, action filter or both by setting 
    self.state_filter to True and self.action_filter to True respectively.
    '''

    state_filter = False
    action_filter = False

    @abstractmethod
    def apply(state_input: NDArray) -> NDArray:
        '''
        Applies a filter to the desired space.
        '''
        raise NotImplementedError("This method should be implemented in a subclass.")

