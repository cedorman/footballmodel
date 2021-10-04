#
# Implementation of the HyGene model from original paper
#
import logging

import numpy as np

np.random.seed(1324)


class Cues:
    """Represents a set of cues"""

    def __init__(self):
        self.cues = None

    def set_cues(self, cues):
        self.cues = cues

    def distance_to_cues(self, comparison_cues):
        """Passed in a set of cues, we return the distance to our cues"""


class Hygene:
    """HyGene model.


    """

    def __init__(self, t_max: float, act_min: float):
        self.cues = None
        self.T_max = t_max
        self.ACT_MIN = act_min
        self.num_semantic_failures = 0

    def add_episode(self, cues: Cues, result: float):
        """Add an episodic event to memory along with the result of that
        episodic event."""
        pass

    def add_semantic_memory(self, cues: Cues, result: float):
        """Add information to semantic memory"""
        pass

    def receive_cues(self, cues: Cues):
        self.cues = cues
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
