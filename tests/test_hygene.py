import logging
from unittest import TestCase

import numpy as np

from hygene.Cue import Cue

# This data is from the Original Paper
TEST_CUE = [0, 1, -1, 1, 0, 1, -1, 1, 0]

TESTDATA = [
    [0, 0, -1, 1, 0, 1, -1, 0, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 1, -1, 0, 0, 1, -1, 1, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 0, -1, 1, 0, 1, -1, 1, 0],
    [-1, 0, -1, 0, -1, 1, 1, 0, 0],
    [-1, 0, -1, 1, -1, 1, 1, 0, 0],
    [-1, 0, 0, 1, -1, 1, 1, 1, 0],
    [1, 0, -1, -1, 1, 1, -1, 0, 0]
]

TEST_ACTIVATION = [0.2963, 1., 0.5787, 1, 1, 0.5787, 0.002, 0.0156, 0.0156, 0.0156]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestCue(TestCase):

    def test_apply_learning_rate(self):
        symp = Cue(10)
        symp.apply_learning_rate(.9)
        logging.warning(f"Resulting cue: {symp}")


class TestCue(TestCase):
    def test_activation(self):
        # Test on some values from the original paper
        a = Cue(1)
        a.set_values(np.array(TEST_CUE))
        logging.warning(f"Probe   : {a}")

        for ii in range(0, len(TESTDATA)):
            b = Cue(1)
            b.set_values(np.array(TESTDATA[ii]))

            d = Cue.activation(a, b)
            logging.warning(f"Trace {ii} : {b}   activation: {d}")
            np.testing.assert_almost_equal(d, TEST_ACTIVATION[ii], decimal=4)
