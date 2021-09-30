import logging
from unittest import TestCase

import numpy as np

from hygene.hygene import Symptom

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestSymptom(TestCase):

    def test_apply_learning_rate(self):
        symp = Symptom(10)
        symp.apply_learning_rate(.9)
        logging.warning(f"Resulting symptom: {symp}")


class TestSymptom(TestCase):
    def test_activation(self):
        a = Symptom(1)
        a.set_values(np.array([0, 1, -1, 1, 0, 1, -1, 1, 0]))
        b = Symptom(1)
        b.set_values(np.array([0, 0, -1, 1, 0, 1, -1, 0, 0]))
        logging.warning(f"a : {a}")
        logging.warning(f"b : {b}")
        d = Symptom.activation(a, b)
        logging.warning(f"activation: {d}")
        np.testing.assert_almost_equal(d, 0.2963, decimal=4)
