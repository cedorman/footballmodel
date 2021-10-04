import logging
from unittest import TestCase

import numpy as np

from hygene.Cue import Cue
from hygene.hygene import Hygene
from tests.original_data import TEST_ACTIVATION_THRESHOLD, TEST_PROBE, TEST_DATA, \
    TEST_ACTIVATION, TEST_CONTENT_VECTOR, TEST_HYPO, TEST_CONTENT_HYPO_VECTOR, \
    TEST_UNSPEC_PROBE, TEST_SEMANTIC_MEMORY, TEST_SEMANTIC_ACTIVATION_NORMED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestHygene(TestCase):

    def test_hygene(self):
        hy = Hygene(0, TEST_ACTIVATION_THRESHOLD)
        hy.set_probe(Cue(TEST_PROBE))
        hy.set_cues([Cue(x) for x in TEST_DATA])
        hy.set_hypos([Cue(x) for x in TEST_HYPO])
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
        content_data, content_hypo = hy.calculate_content_vectors()

        logging.warning(f"Content vector: {content_data}")
        for ii in range(0, len(content_data.vals)):
            np.testing.assert_almost_equal(TEST_CONTENT_VECTOR[ii], content_data.vals[ii], decimal=1)

        logging.warning(f"Content hypo vector: {content_hypo}")
        for ii in range(0, len(content_hypo.vals)):
            np.testing.assert_almost_equal(TEST_CONTENT_HYPO_VECTOR[ii], content_hypo.vals[ii], decimal=1)

        # --------------------
        # Get unspecified probe
        # --------------------
        unspec_probe = hy.get_unspecified_probe()
        logging.warning(f"Content unspec_probe: {unspec_probe}")
        for ii in range(0, len(TEST_UNSPEC_PROBE)):
            np.testing.assert_almost_equal(TEST_UNSPEC_PROBE[ii], unspec_probe.vals[ii], decimal=1)

        # --------------------
        # Calc relevant hypotheses
        # --------------------
        hy.set_semantic_memory([Cue(x) for x in TEST_SEMANTIC_MEMORY])
        semantic_hypothesis_activations = hy.get_semantic_activations()
        logging.warning(f"Hypothesis activations: {semantic_hypothesis_activations}")
        for ii, semantic_cue in enumerate(hy.semantic):
            act = semantic_cue.get_activation()
            np.testing.assert_almost_equal(TEST_SEMANTIC_ACTIVATION_NORMED[ii], act, decimal=2)
