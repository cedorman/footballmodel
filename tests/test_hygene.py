import logging
from unittest import TestCase

import numpy as np

from hygene.Cue import Cue
from hygene.hygene import Hygene
from tests.original_data import TEST_ACTIVATION_THRESHOLD, TEST_PROBE, TEST_DATA, TEST_ACTIVATION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestHygene(TestCase):

    def test_hygene(self):
        hy = Hygene(0, TEST_ACTIVATION_THRESHOLD)
        hy.set_probe(Cue(TEST_PROBE))
        hy.set_cues([Cue(x) for x in TEST_DATA])
        hy.compute_activations()

        for ii in range(0, len(TEST_ACTIVATION)):
            act = hy.get_activation(ii)
            np.testing.assert_almost_equal(act, TEST_ACTIVATION[ii], decimal=4)
