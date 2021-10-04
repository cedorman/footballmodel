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
    """HyGene model
    """

    def __init__(self, t_max: float = 10, act_min: float = 0.0):

        self.T_max = t_max
        self.act_min = act_min

        # Data to be set
        self.probe = None  # type: Cue
        self.cues = None  # type: List[Cue]
        self.hypos = None  # type: List[Cue]
        self.semantic = None  # type: List[Cue]

        # Data Calculated
        self.content_hypo = None
        self.content = None
        # Note:  This is un-normed until you call norm on it
        self.semantic_activation = None

    def set_probe(self, probe: Cue):
        self.probe = probe

    def set_cues(self, new_cues: List[Cue]):
        self.cues = new_cues

    def set_hypos(self, new_hypos: List[Cue]):
        self.hypos = new_hypos

    def set_semantic_memory(self, new_sem: List[Cue]):
        self.semantic = new_sem

    def compute_activations(self):
        if self.probe is None:
            logging.warning("No probe")

        if self.cues is None:
            logging.warning("No cues")

        [Cue.activation(self.probe, cue) for cue in self.cues]

    def get_activation(self, index):
        return self.cues[index].get_activation()

    def calculate_content_vectors(self):
        """Create the content vector based on the cues and the activations,
        but only the activations above a certain level"""

        if self.cues is None:
            logging.warning("No cues")

        # The data component
        self.content = Cue.zeros(len(self.cues[0].vals))
        for cue in self.cues:
            act = cue.get_activation()
            if act >= self.act_min:
                cue.activated = True
                weighted_value = act * cue.vals
                self.content.add_vals(weighted_value)

        # The hypothesis component
        self.content_hypo = Cue.zeros(len(self.cues[0].vals))
        for ii, cue in enumerate(self.hypos):
            act = self.cues[ii].get_activation()
            if act >= self.act_min:
                weighted_value = act * cue.vals
                self.content_hypo.add_vals(weighted_value)

        return self.content, self.content_hypo

    def get_unspecified_probe(self):
        p = np.append(self.content.vals, self.content_hypo.vals)
        p /= np.max(p)
        self.unspecified_probe = Cue(p)
        return self.unspecified_probe

    def get_semantic_activations(self):
        acts = [Cue.activation(self.unspecified_probe, cue) for cue in self.semantic]
        sumval = sum(acts)
        for cue in self.semantic:
            cue.act /= sumval
            if cue.act > 0:
                cue.activated = True
        return [cue.act for cue in self.semantic]
