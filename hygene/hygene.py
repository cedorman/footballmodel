#
# Implementation of the HyGene model from original paper
#
import logging
from typing import List

import numpy as np

from hygene.cue import Cue

np.random.seed(1324)


class Hygene:
    """HyGene model
    """

    def __init__(self, t_max: float = 10, act_thresh: float = 0.0):

        self.T_max = t_max
        self.act_thresh = act_thresh
        self.act_min_h = 0.  # Hypothesis min activation starts at 0
        self.retrieval_failure_limit = 1
        self.retrieval_falures = 0

        # Data to be set
        self.probe: Cue = None
        self.cues: List[Cue] = None
        self.hypos: List[Cue] = None
        self.semantic: List[Cue] = None

        # Data Calculated
        self.content_hypo = None
        self.content = None
        self.semantic_activations = None

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

        [Cue.compute_activation(self.probe, cue) for cue in self.cues]

    def get_activation(self, index):
        '''Return the activation for a particular cue.
        NOTE:  must have called compute_activation first.'''
        return self.cues[index].get_activation()

    def calculate_content_vectors(self):
        """Step 1: Create the content vector based on the cues and the
        activations, but only the activations above a certain level"""

        if self.cues is None:
            logging.warning("No cues")

        # The data component
        self.content = Cue.zeros(len(self.cues[0].vals))
        for cue in self.cues:
            act = cue.get_activation()
            if act >= self.act_thresh:
                cue.activated = True
                weighted_value = act * cue.vals
                self.content.add_vals(weighted_value)

        # The hypothesis component
        self.content_hypo = Cue.zeros(len(self.cues[0].vals))
        for ii, cue in enumerate(self.hypos):
            act = self.cues[ii].get_activation()
            if act >= self.act_thresh:
                weighted_value = act * cue.vals
                self.content_hypo.add_vals(weighted_value)

        return self.content, self.content_hypo

    def get_unspecified_probe(self):
        '''Step 2:  Extraction of unspecified probe.
        Combine content vector and hypothesis content vector
        and normalize. '''
        p = np.append(self.content.vals, self.content_hypo.vals)
        p /= np.max(p)
        self.unspecified_probe = Cue(p)
        return self.unspecified_probe

    def get_semantic_activations(self):
        '''Step 3:  compute _semantic_ activations.  To turn them
         into probabilities, sum the activations and normalize '''
        acts = [Cue.compute_activation(self.unspecified_probe, cue) for cue in self.semantic]
        sumval = sum(acts)
        for cue in self.semantic:
            cue.act /= sumval
            if cue.act > 0:
                cue.activated = True
            else:
                cue.act = 0.
                cue.act = False
        self.semantic_activations = [cue.act for cue in self.semantic]
        return self.semantic_activations

    def sample_hypotheses(self):
        """Step 4: Sample from the activated hypotheses until successive
        number of failures == T_MAX, where:
        - Retrieval of a semantic memory that does not exceed act_min_h is
          a failure.
        - Sampling from the same semantic memory is a failure. """

        # Act min h starts at zero.  Successful adding to SOC resets
        self.act_min_h = 0.

        # Set of Leading Contenders (SOC)
        self.soc = []
        while self.retrieval_falures < self.retrieval_failure_limit:
            hypothesis = self.pick_hypothesis()
            if hypothesis.act > self.act_min_h and hypothesis not in self.soc:
                self.soc.append(hypothesis)
                self.act_min_h = max(self.act_min_h, hypothesis.act)
                self.retrieval_falures = 0
            else:
                self.retrieval_falures += 1
        return self.soc

    def pick_hypothesis(self):
        hypo = np.random.choice(self.semantic, p=self.semantic_activations)
        return hypo
