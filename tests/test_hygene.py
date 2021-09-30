import logging
from unittest import TestCase

from hygene.hygene import Symptom

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestSymptom(TestCase):

    def test_apply_learning_rate(self):
        symp = Symptom(10)
        symp.apply_learning_rate(.9)
        logging.warning(f"Resulting symptom: {symp}")
