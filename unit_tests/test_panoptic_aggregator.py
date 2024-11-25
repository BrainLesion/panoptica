# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

import numpy as np

from panoptica import InputType, Panoptica_Aggregator, Panoptica_Statistic
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import MaximizeMergeMatching, NaiveThresholdMatching
from panoptica.metrics import Metric
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.panoptica_result import MetricCouldNotBeComputedException
from panoptica.utils.processing_pair import SemanticPair
from panoptica.utils.segmentation_class import SegmentationClassGroups
import sys
from pathlib import Path


output_test_dir = Path(__file__).parent.joinpath("unittest_tmp_file.tsv")


class Test_Example_Scripts(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_example_scripts_future(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        print(directory)
        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_statistics import main

        main("future")

    def test_example_scripts_pool(self):
        directory = Path(__file__).parent.parent.joinpath("examples")

        print(directory)
        if not directory.exists():
            self.skipTest(f"directory {directory} does not exist")

        sys.path.append(str(directory))

        from examples.example_spine_statistics import main

        main("pool")


class Test_Panoptica_Aggregator(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_evaluation(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")
        os.remove(str(output_test_dir))

    def test_simple_evaluation_then_statistic(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")

        statistic_obj = aggregator.make_statistic()
        statistic_obj.print_summary()

        os.remove(str(output_test_dir))
