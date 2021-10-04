#
import logging

import numpy as np


class Cue:
    """ Single cue, an array with values [-1, 0, 1], of length l """

    def __init__(self, length):
        self.vals = np.random.choice([-1, 1], size=length)

    def apply_learning_rate(self, learning_rate=1.):
        """ With probability (1-learning_rate), set to zero.
        If learning_rate is high (close to 1), then prob is close to 0"""
        num_indices = int(self.vals.size * (1. - learning_rate))
        indices = np.random.choice(np.arange(self.vals.size), replace=False,
                                   size=num_indices)
        self.vals[indices] = 0

    def set_values(self, vals):
        """ For testing, explicitly set to certain values"""
        self.vals = vals

    @classmethod
    def activation(cls, a, b) -> float:
        """Measure the activation between this cue and another.   The
        rule is it is sum of a-values times b-values, when either
        a or b is not zero, cubed.  """
        len = a.vals.size
        if len != b.vals.size:
            logging.warning("Unequal size vectors")

        x = 0.
        count = 0.
        for ii in range(len):
            va = a.vals[ii]
            vb = b.vals[ii]
            if va != 0 or vb != 0:
                count += 1
                x += va * vb
        x /= count
        x = x * x * x
        return x

    def __str__(self):
        printstring = "[ " + ", ".join('{0:2d}'.format(val) for val in self.vals) + " ]"
        return printstring
