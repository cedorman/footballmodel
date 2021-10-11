import logging
from unittest import TestCase

import numpy as np

from hygene.cue import Cue
from hygene.hygene import Hygene
from tests.original_data import TEST_ACTIVATION_THRESHOLD, TEST_PROBE, TEST_DATA, \
    TEST_ACTIVATION, TEST_CONTENT_VECTOR, TEST_HYPO, TEST_CONTENT_HYPO_VECTOR, \
    TEST_SEMANTIC_ACTIVATION_NORMED, \
    TEST_EVENT, TEST_UNSPEC_PROBE_DATA, TEST_UNSPEC_PROBE_HYPO, TEST_SEMANTIC_MEMORY_DATA, TEST_SEMANTIC_MEMORY_HYPO, \
    TEST_SEMANTIC_MEMORY_EVENT

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestHygene(TestCase):

    def test_hygene(self):
        hy = Hygene(0, TEST_ACTIVATION_THRESHOLD)
        hy.set_probe(Cue.probe(TEST_PROBE))
        hy.set_cues([Cue(TEST_DATA[ii], TEST_HYPO[ii], TEST_EVENT[ii]) for ii in range(len(TEST_DATA))])
        hy.compute_activations()

        # --------------------
        # For each Cue, make sure that the activation is correct
        # --------------------
        for ii in range(0, len(TEST_ACTIVATION)):
            act = hy.get_activation(ii)
            np.testing.assert_almost_equal(act, TEST_ACTIVATION[ii], decimal=4)

        # --------------------
        # Get the content vector and make sure that it is correct
        # --------------------
        content = hy.calculate_content_vectors()

        logging.warning(f"Content vector: {content.vals}")
        for ii in range(0, len(content.vals)):
            np.testing.assert_almost_equal(content.vals[ii], TEST_CONTENT_VECTOR[ii], decimal=1)

        logging.warning(f"Content hypo vector: {content.hypo}")
        for ii in range(0, len(content.hypo)):
            np.testing.assert_almost_equal(content.hypo[ii], TEST_CONTENT_HYPO_VECTOR[ii], decimal=1)

        # --------------------
        # Get unspecified probe
        # --------------------
        unspec_probe = hy.get_unspecified_probe()
        logging.warning(f"Content unspec_probe: {unspec_probe}")
        for ii in range(0, len(TEST_UNSPEC_PROBE_DATA)):
            np.testing.assert_almost_equal(unspec_probe.vals[ii], TEST_UNSPEC_PROBE_DATA[ii], decimal=1)
        for ii in range(0, len(TEST_UNSPEC_PROBE_HYPO)):
            np.testing.assert_almost_equal(unspec_probe.hypo[ii], TEST_UNSPEC_PROBE_HYPO[ii], decimal=1)

        # --------------------
        # Calc relevant hypotheses
        # --------------------
        hy.set_semantic_memory(
            [Cue(TEST_SEMANTIC_MEMORY_DATA[ii],
                 TEST_SEMANTIC_MEMORY_HYPO[ii],
                 TEST_SEMANTIC_MEMORY_EVENT[ii])
             for ii in range(len(TEST_SEMANTIC_MEMORY_DATA))])
        semantic_hypothesis_activations = hy.get_semantic_activations()
        logging.warning(f"Hypothesis activations: {semantic_hypothesis_activations}")
        for ii, semantic_cue in enumerate(hy.semantic):
            act = semantic_cue.get_activation()
            np.testing.assert_almost_equal(act, TEST_SEMANTIC_ACTIVATION_NORMED[ii], decimal=2)

        # --------------------
        # Sample
        # --------------------
        hy.sample_hypotheses()

        # --------------------
        # Calc probabilities
        # --------------------
        hy.set_soc([0])
        echo_intensities= hy.get_echo_intensities()
        logging.warning(f"Echo intensity for the first semantic memory component: {echo_intensities}")