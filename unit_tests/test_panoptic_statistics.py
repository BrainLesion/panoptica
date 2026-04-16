# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
from panoptica import (
    InputType,
    Panoptica_Aggregator,
    Panoptica_Statistic,
    FloatDistribution,
)
from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
from panoptica.instance_matcher import NaiveThresholdMatching
from panoptica.panoptica_evaluator import Panoptica_Evaluator
from pathlib import Path

output_test_dir = Path(__file__).parent.joinpath("unittest_tmp_file.tsv")

input_test_file = Path(__file__).parent.joinpath("test_unittest_file.tsv")


class Test_Panoptica_Statistics(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_simple_statistic(self):
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

        statistic_obj = Panoptica_Statistic.from_file(output_test_dir)

        statistic_obj.print_summary()

        self.assertEqual(statistic_obj.get("ungrouped", "tp"), [1.0])
        self.assertEqual(statistic_obj.get("ungrouped", "sq"), [0.75])
        self.assertEqual(statistic_obj.get("ungrouped", "sq_rvd"), [-0.25])

        tp_values = statistic_obj.get("ungrouped", "tp")
        sq_values = statistic_obj.get("ungrouped", "sq")
        self.assertEqual(FloatDistribution(tp_values).avg, 1.0)
        self.assertEqual(FloatDistribution(sq_values).avg, 0.75)

        os.remove(str(output_test_dir))

    def test_multiple_samples_statistic(self):
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        c = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:35, 10:20] = 2
        c[20:40, 10:20] = 5
        c[0:10, 0:10] = 3

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        aggregator = Panoptica_Aggregator(evaluator, output_file=output_test_dir)

        aggregator.evaluate(b, a, "test")
        aggregator.evaluate(a, c, "test2")

        statistic_obj = Panoptica_Statistic.from_file(output_test_dir)

        statistic_obj.print_summary()

        self.assertEqual(
            FloatDistribution(statistic_obj.get("ungrouped", "tp")).avg, 1.0
        )
        self.assertEqual(
            FloatDistribution(statistic_obj.get("ungrouped", "sq")).avg, 0.875
        )
        self.assertEqual(
            FloatDistribution(statistic_obj.get("ungrouped", "fn")).avg, 0.5
        )
        self.assertEqual(
            FloatDistribution(statistic_obj.get("ungrouped", "rec")).avg, 0.75
        )
        self.assertEqual(
            FloatDistribution(statistic_obj.get("ungrouped", "rec")).std, 0.25
        )

        os.remove(str(output_test_dir))

    def test_statistics_from_file(self):
        statistic_obj = Panoptica_Statistic.from_file(input_test_file)
        #
        test2 = statistic_obj.get_one_subject("test2")  # get one subject
        print()
        print("test2", test2)
        self.assertEqual(test2["ungrouped"]["n_ref_instances"], 2)

        all_n_ref_instances = statistic_obj.get_across_groups("n_ref_instances")
        print()
        print("all_n_ref_instances", all_n_ref_instances)
        self.assertEqual(len(all_n_ref_instances), 2)
        self.assertEqual(sum(all_n_ref_instances), 3)

        groupwise_summary = statistic_obj.get_summary_across_groups()
        print()
        print(groupwise_summary)
        self.assertEqual(groupwise_summary["n_ref_instances"].avg, 1.5)

    def test_autc_statistic_write_read(self):
        """
        Tests the Panoptica_Aggregator's ability to correctly build headers,
        evaluate AUTC, write to a TSV, and read back via Panoptica_Statistic.
        """
        a = np.zeros([50, 50], dtype=np.uint16)
        b = a.copy().astype(a.dtype)
        a[20:40, 10:20] = 1
        b[20:40, 10:20] = 1

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        # thresholds will be [0.5, 1.0]
        aggregator = Panoptica_Aggregator(
            panoptica_evaluator=evaluator,
            output_file=output_test_dir,
            is_autc=True,
            threshold_step_size=0.5,
        )

        aggregator.evaluate(b, a, "test_autc_subject")

        statistic_obj = Panoptica_Statistic.from_file(output_test_dir, verbose=False)

        metrics = statistic_obj.metricnames

        self.assertIn("autc_pq", metrics)
        self.assertIn("autc_sq", metrics)

        self.assertIn("t0.5_pq", metrics)
        self.assertIn("t1_pq", metrics)

        autc_pq_vals = statistic_obj.get("ungrouped", "autc_pq")
        self.assertEqual(len(autc_pq_vals), 1)
        self.assertEqual(autc_pq_vals[0], 1.0)

        t05_pq_vals = statistic_obj.get("ungrouped", "t0.5_pq")
        self.assertEqual(t05_pq_vals[0], 1.0)

        t1_pq_vals = statistic_obj.get("ungrouped", "t1_pq")
        self.assertEqual(t1_pq_vals[0], 1.0)

        if output_test_dir.exists():
            os.remove(str(output_test_dir))

    def test_mismatch_regular_and_autc_aggregator_header(self):
        """
        Tests that an AssertionError is raised if an AUTC aggregator tries
        to append to a TSV file created by a regular (non-AUTC) aggregator.
        """
        a = np.zeros([50, 50], dtype=np.uint16)
        a[20:40, 10:20] = 1

        evaluator_regular = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        agg_regular = Panoptica_Aggregator(
            panoptica_evaluator=evaluator_regular,
            output_file=output_test_dir,
            is_autc=False,
        )
        agg_regular.evaluate(a, a, "test_regular")

        # Try to resume/append to the same file but with an AUTC setup
        with self.assertRaisesRegex(AssertionError, "Hash of header not the same"):
            agg_autc = Panoptica_Aggregator(
                panoptica_evaluator=evaluator_regular,
                output_file=output_test_dir,
                is_autc=True,
                threshold_step_size=0.1,
            )

        if output_test_dir.exists():
            os.remove(str(output_test_dir))

    def test_make_autc_plots_graceful_on_regular_stats(self):
        """
        Tests that make_autc_plots does not crash when provided with
        a regular Panoptica_Statistic (without threshold columns),
        but gracefully skips and returns a base figure without traces.
        """
        from panoptica.panoptica_statistics import make_autc_plots

        a = np.zeros([50, 50], dtype=np.uint16)
        a[20:40, 10:20] = 1

        evaluator = Panoptica_Evaluator(
            expected_input=InputType.SEMANTIC,
            instance_approximator=ConnectedComponentsInstanceApproximator(),
            instance_matcher=NaiveThresholdMatching(),
        )

        agg = Panoptica_Aggregator(
            panoptica_evaluator=evaluator, output_file=output_test_dir, is_autc=False
        )
        agg.evaluate(a, a, "test_reg")

        stat = Panoptica_Statistic.from_file(output_test_dir, verbose=False)

        # Attempt to plot an AUTC curve from a regular stat object
        # It should print a warning internally, but not crash.
        fig = make_autc_plots(statistics_dict={"Regular": stat}, metric="pq")

        # Verify the figure was created but has 0 traces plotted
        self.assertEqual(len(fig.data), 0)

        if output_test_dir.exists():
            os.remove(str(output_test_dir))
