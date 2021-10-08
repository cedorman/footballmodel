import logging
from unittest import TestCase

import numpy as np

from hygene.cue import Cue
from tests.original_data import TEST_PROBE, TEST_DATA, TEST_ACTIVATION


class TestCue(TestCase):

    def test_ctors(self):
        with_array = Cue([1,1,1,1,-1,-1])
        with_len_rand = Cue.random(7)
        with_len_zero = Cue.zeros(8)
        logging.warning(f"Resulting with_array: {with_array}")
        logging.warning(f"Resulting with_len_rand: {with_len_rand}")
        logging.warning(f"Resulting with_len_zero: {with_len_zero}")

    def test_apply_learning_rate(self):
        symp = Cue.random(10)
        symp.apply_learning_rate(.3)
        logging.warning(f"Resulting cue: {symp}")

    def test_activation(self):
        # Test on some values from the original paper
        a = Cue()
        a.set_values(np.array(TEST_PROBE))
        logging.warning(f"Probe   : {a}")

        for ii in range(0, len(TEST_DATA)):
            b = Cue(1)
            b.set_values(np.array(TEST_DATA[ii]))

            d = Cue.compute_activation(a, b)
            logging.warning(f"Trace {ii} : {b}   activation: {d}")
            np.testing.assert_almost_equal(d, TEST_ACTIVATION[ii], decimal=4)
