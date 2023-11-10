# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import unittest

from instance_approximator import ConnectedComponentsInstanceApproximator, CCABackend
from instance_matcher import NaiveOneToOneMatching
from instance_evaluator import evaluate_matched_instance
import numpy as np
from result import PanopticaResult
from utils.datatypes import SemanticPair, UnmatchedInstancePair, MatchedInstancePair, _ProcessingPair
from evaluator import Panoptic_Evaluator


class Test_Panoptic_Evaluator(unittest.TestCase):
    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=int)
        b = a.copy()
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        sample = SemanticPair(b, a)

        evaluator = Panoptic_Evaluator(
            expected_input=SemanticPair,
            instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=CCABackend.cc3d),
            instance_matcher=NaiveOneToOneMatching(),
        )

        result, debug_data = evaluator.evaluate(sample)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.sq, 0.75)
        self.assertEqual(result.pq, 0.75)
