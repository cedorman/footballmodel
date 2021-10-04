#
import logging
from typing import List

import numpy as np


class Cue:
    """ Single cue, an array with values [-1, 0, 1], of length l """

    def __init__(self, init_values: List[int] = None):
        if init_values is not None:
            self.vals = np.array(init_values)
        self.act = 0.

    @classmethod
    def random(cls, length: int):
        return cls([np.random.choice([-1, 1]) for ii in range(0, length)])

    def apply_learning_rate(self, learning_rate=1.):
        """ With probability (1-learning_rate), set to zero.
        If learning_rate is high (close to 1), then prob is close to 0"""
        num_indices = int(self.vals.size * (1. - learning_rate))
        indices = np.random.choice(np.arange(self.vals.size), replace=False,
                                   size=num_indices)
        self.vals[indices] = 0

    def set_values(self, vals):
        """ For testing, explicitly set to certain values"""
        if type(vals) is np.ndarray:
            self.vals = vals
        else:
            self.vals = np.array(vals)

    @classmethod
    def activation(cls, probe, cue) -> float:
        """Measure the activation between this cue and another.   The
        rule is it is sum of a-values times b-values, when either
        a or b is not zero, cubed.  """
        len = probe.vals.size
        if len != cue.vals.size:
            logging.warning("Unequal size vectors")

        x = 0.
        count = 0.
        for ii in range(len):
            va = probe.vals[ii]
            vb = cue.vals[ii]
            if va != 0 or vb != 0:
                count += 1
                x += va * vb
        x /= count
        x = x * x * x

        cue.set_activation(x)

        return x

    def set_activation(self, new_act):
        self.act = new_act

    def get_activation(self):
        return self.act

    def __str__(self):
        printstring = "[ " + ", ".join('{0:2d}'.format(val) for val in self.vals) + " ]"
        return printstring
