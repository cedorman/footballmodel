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
        self.act_thresh = act_thresh  # In the paper, this is A_c,
        self.act_min_h = 0.  # Hypothesis min activation starts at 0
        self.retrieval_failure_limit = 1
        self.retrieval_falures = 0

        # Data to be set
        self.probe: Cue = None
        self.traces: List[Cue] = []
        self.semantic: List[Cue] = []

        # Data Calculated
        self.content: Cue = None
        self.semantic_activations: List[float] = []
        self.soc: List[Cue] = []
        self.I_c = List[float]

    def set_probe(self, probe: Cue):
        self.probe = probe

    def set_cues(self, new_cues: List[Cue]):
        self.traces = new_cues

    def set_semantic_memory(self, new_sem: List[Cue]):
        self.semantic = new_sem

    def set_soc(self, new_soc_indices: List[int]):
        '''For debugging and testing, we want to be able to set
        the set of leading contenders to particular semantic memories.'''
        self.soc = []
        for ii in new_soc_indices:
            self.soc.append(self.semantic[ii])

    def get_semantic_hypothesis(self, index: int):
        return self.semantic[index]

    def compute_activations(self):
        [Cue.compute_activation(self.probe, trace) for trace in self.traces]

    def get_activation(self, index: int):
        '''Return the activation for a particular cue.
        NOTE:  must have called compute_activation first.'''
        return self.traces[index].get_activation()

    def calculate_content_vectors(self):
        """Step 1: Create the content vector based on the cues and the
        activations, but only the activations above a certain level"""

        if self.traces is None:
            logging.warning("No traces")

        # The data component
        self.content = Cue.zeros(len(self.traces[0].vals), -1)
        for cue in self.traces:
            act = cue.get_activation()
            if act >= self.act_thresh:
                cue.activated = True

                weighted_value = act * cue.vals
                self.content.add_vals(weighted_value)

                weighted_value = act * cue.hypo
                self.content.add_hypos(weighted_value)

        return self.content

    def get_unspecified_probe(self):
        '''Step 2:  Extraction of unspecified probe.
        Combine content vector and hypothesis content vector
        and normalize. '''
        self.unspecified_probe = self.content
        self.unspecified_probe.normalize()
        return self.unspecified_probe

    def get_semantic_activations(self):
        '''Step 3:  compute _semantic_ memory activations.  To turn them
         into probabilities, sum the activations and normalize '''
        acts = [Cue.compute_semantic_activation(self.unspecified_probe, cue)
                for cue in self.semantic]
        acts = [max(0, act) for act in acts]

        # Now, normalize them and activate the positive ones
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

    def get_echo_intensities(self):
        '''Step 5:  For each hypothesis in SOC, use the
        traces (above A_c, activation threshold) to compute
        I_c.  '''
        self.I_c: List[float] = []  # Array of floats, length of SOC
        for soc_contender in self.soc:
            count = 0
            hypothesis_I_c = 0
            for trace in self.traces:
                if trace.event == soc_contender.event and trace.activated:
                    hypothesis_I_c += Cue.compute_act(soc_contender.hypo, trace.hypo)
                    logging.debug(f"for trace {trace.hypo} and soc {soc_contender.hypo}  Sum_I_c = {hypothesis_I_c}")
                    count += 1
            if count > 0:
                hypothesis_I_c /= count
            self.I_c.append(hypothesis_I_c)

        return self.I_c

    def get_probabilities(self):
        '''Step 6:  Normalize over the intensities of SOC to
        make them probabilities'''
        self.probabilities = []
        asum = sum(self.I_c)
        for val in self.I_c:
            self.probabilities.append(val / asum)
        return self.probabilities
