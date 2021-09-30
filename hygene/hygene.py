#
# Implementation of the HyGene model from original paper
#
import logging

import numpy as np

np.random.seed(1324)


class Symptom:
    """ Single symptom, [-1, 0, 1] with length l """

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
        """Measure the activation between this symptom and another.   The
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


class Symptoms:
    """Represents a set of symptoms or cues"""

    def __init__(self):
        self.symptoms = None

    def set_symptoms(self, symptoms):
        self.symptoms = symptoms

    def distance_to_symptoms(self, comparison_symptoms):
        """Passed in a set of symptoms, we return the distance to our symptoms"""


class Hygene:
    """HyGene model.


    """

    def __init__(self, t_max: float, act_min: float):
        self.symptoms = None
        self.T_max = t_max
        self.ACT_MIN = act_min
        self.num_semantic_failures = 0

    def add_episode(self, symptoms: Symptoms, result: float):
        """Add an episodic event to memory along with the result of that
        episodic event."""
        pass

    def add_semantic_memory(self, symptoms: Symptoms, result: float):
        """Add information to semantic memory"""
        pass

    def receive_symptoms(self, symptoms: Symptoms):
        self.symptoms = symptoms
        self.activate_episodic_traces()
        self.compare_with_semantic()
        return self.generate_probability_judgement()

    def activate_episodic_traces(self):
        pass

    def compare_with_semantic(self):
        while self.num_semantic_failures < self.T_max:
            score, semantic_index = self.get_highest_semantic()
            if score > self.ACT_MIN:
                self.add_semantic_to_working_memory(semantic_index)

    def generate_probability_judgement(self):
        pass

    def get_highest_semantic(self):
        return 0., 0

    def add_semantic_to_working_memory(self, semantic_index):
        pass
