#
import logging
from typing import List

import numpy as np


class Cue:
    """ Single cue, an array with values [-1, 0, 1], of length l """

    def __init__(self, init_val: List[float], init_hypo: List[float], event: int):
        self.set_values(init_val)
        self.set_hypo(init_hypo)
        self.act = 0.
        self.activated = False
        self.event = event

    @classmethod
    def with_np(cls, init_val: np.ndarray, init_hypo: np.ndarray, event: int):
        acue = cls([], [], event)
        acue.set_values(init_val)
        acue.set_hypo(init_hypo)
        return acue

    @classmethod
    def random(cls, length: int, event: int = 0):
        vals = [np.random.choice([-1., 1.]) for ii in range(0, length)]
        hypo = [np.random.choice([-1., 1.]) for ii in range(0, length)]
        acue = cls(vals, hypo, event)
        return acue

    @classmethod
    def zeros(cls, length: int, event: int = 0):
        acue = cls([], [], event)
        acue.set_values(np.zeros(length))
        acue.set_hypo(np.zeros(length))
        return acue

    @classmethod
    def probe(cls, vals: np.ndarray):
        """Special ctor that has vals, but not hypotheses or event yet"""
        acue = cls([], [], -1)
        acue.set_values(vals)
        return acue

    def set_values(self, vals):
        """ For testing, explicitly set to certain values"""
        if type(vals) is np.ndarray:
            self.vals = vals
        else:
            self.vals = np.array(vals)

    def set_hypo(self, vals):
        """ For testing, explicitly set to certain values"""
        if type(vals) is np.ndarray:
            self.hypo = vals
        else:
            self.hypo = np.array(vals)

    def apply_learning_rate(self, learning_rate=1.):
        """ With probability (1-learning_rate), set to zero.
        If learning_rate is high (close to 1), then prob is close to 0
        and will keep more data."""
        if learning_rate == 1.0:
            return
            
        num_indices = int(self.vals.size * (1. - learning_rate))
        indices = np.random.choice(np.arange(self.vals.size), replace=False,
                                   size=num_indices)
        self.vals[indices] = 0

    def set_event(self, new_event):
        self.event = new_event

    @staticmethod
    def compute_act(a, b) -> float:
        
        length = a.size
        if length != b.size:
            logging.warning("Unequal size vectors")

        x = 0.
        count = 0.
        for ii in range(length):
            va = a[ii]
            vb = b[ii]
            if va != 0 or vb != 0:
                count += 1
                x += va * vb
        x /= count
        x = x * x * x
        logging.debug(f"activation: {a} {b} -> {x}")
        return x

    @classmethod
    def compute_activation(cls, probe, cue) -> float:
        """Measure the activation between this cue and another based
        on the data component.   The rule is it is sum of a-values times
        b-values, when either a or b is not zero, cubed.  """
        act_val = Cue.compute_act(probe.vals, cue.vals)
        cue.set_activation(act_val)
        return act_val

    @classmethod
    def compute_semantic_activation(cls, probe, cue) -> float:
        """Measure the activation between this cue and probe, based
        on both the data and hypotheses component."""
        fulla = np.append(probe.vals, probe.hypo)
        fullb = np.append(cue.vals, cue.hypo)
        act_val = Cue.compute_act(fulla, fullb)
        cue.set_activation(act_val)
        return act_val

    def set_activation(self, new_act):
        self.act = new_act

    def get_activation(self):
        return self.act

    def add_vals(self, vals_to_be_added):
        """When computing the content vector, we need to sum the
        weighted data components.  this makes it easier"""
        self.vals = self.vals + vals_to_be_added

    def add_hypos(self, hypos_to_be_added):
        self.hypo = self.hypo + hypos_to_be_added

    def __str__(self):
        printstring = "[ " + ", ".join('{0:2.2f}'.format(val) for val in self.vals) + " ]"
        return printstring

    def normalize(self):
        """Normalize to range [-1:1] this Cue by dividing by greatest value"""
        max_val = max(np.max(self.vals), np.max(self.hypo))
        self.vals /= max_val
        self.hypo /= max_val
