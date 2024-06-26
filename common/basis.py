import numpy as np


class Basis(object):
    """
    Trivial basis
    """

    def __init__(self, nvars):
        self.num_terms = nvars
        self._shrink = np.ones((self.num_terms,))

    def get_num_basis_functions(self):
        return self.num_terms

    def compute_features(self, state):
        return state

    def get_shrink(self):
        return self._shrink

class ScaledBasis(Basis):
    """
    Scales variables in the range [0,1]
    """

    def __init__(self, nvars, low, high, bias_unit=False):
        super().__init__(nvars)
        self.low = low
        self.high = high
        self.range = self.high-self.low
        self._bias_unit = bias_unit
        if self._bias_unit:
            self.num_terms += 1

    def scale_state(self, state):
        return (state - self.low)/self.range

    def compute_features(self, state):
        scaled_state = self.scale_state(state)
        if self._bias_unit:
            scaled_state = np.concatenate(([1.], scaled_state))
        return scaled_state