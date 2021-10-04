#
# Implementation of the HyGene model from original paper
#
import logging
from typing import List

import numpy as np

from hygene.Cue import Cue

np.random.seed(1324)


#
# class Cue:
#     """Represents a set of cue"""
#
#     def __init__(self):
#         self.cue = None
#
#     def set_cue(self, cue):
#         self.cue = cue
#
#     def distance_to_cue(self, comparison_cue):
#         """Passed in a set of cue, we return the distance to our cue"""
#

class Hygene:
    """HyGene model.


    """

    def __init__(self, t_max: float, act_min: float):
        self.cues = None  # type: List[Cue]
        self.T_max = t_max
        self.ACT_MIN = act_min
        self.num_semantic_failures = 0

        self.probe = None  # type: Cue

    def set_probe(self, probe: Cue):
        self.probe = probe

    def set_cues(self, new_cues: List[Cue]):
        self.cues = new_cues

    def compute_activations(self):
        if self.probe is None:
            logging.warning("No probe")

        if self.cues is None:
            logging.warning("No cues")

        [Cue.activation(self.probe, cue) for cue in self.cues]

    def get_activation(self, index):
        return self.cues[index].get_activation()

    def receive_cue(self, cue: Cue):
        self.cue = cue
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
