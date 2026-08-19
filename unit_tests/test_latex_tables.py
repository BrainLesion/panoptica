# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
from pathlib import Path

from panoptica import Panoptica_Statistic
from panoptica.panoptica_statistics import make_latex_table_over_setups
from panoptica.utils.citation_reminder import disable_citation_reminder
from panoptica.utils.latex_utils import (
    LATEX_TABLE_REQUIREMENTS,
    base_metric_name,
    escape_latex,
    format_mean_std,
    metric_arrow,
    metric_direction,
    prettify_group_name,
    prettify_metric_name,
    render_latex_table,
    select_best_indices,
)

output_test_file = Path(__file__).parent.joinpath("unittest_tmp_table.tex")


def _make_statistic(
    pq_vertebra=(0.8, 0.9),
    pq_ivd=(0.6, 0.5),
    assd_vertebra=(2.0, 3.0),
    assd_ivd=(4.0, 5.0),
) -> Panoptica_Statistic:
    """Two groups, two metrics, two subjects; built without running an evaluator."""
    return Panoptica_Statistic(
        subj_names=["s1", "s2"],
        value_dict={
            "vertebra": {"pq": list(pq_vertebra), "sq_assd": list(assd_vertebra)},
            "ivd": {"pq": list(pq_ivd), "sq_assd": list(assd_ivd)},
        },
    )


def _body_lines(table: str) -> list[str]:
    """The cell-carrying lines of a rendered table (header row included)."""
    return [
        line
        for line in table.splitlines()
        if line.endswith(r"\\")
        and not line.lstrip().startswith("%")
        # The spanning group header intentionally has fewer cells.
        and r"\multicolumn" not in line
    ]


class Test_Latex_Utils(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
        return super().setUp()

    def test_escape_latex(self):
        self.assertEqual(escape_latex("sq_dsc"), r"sq\_dsc")
        self.assertEqual(escape_latex("a & b"), r"a \& b")
        self.assertEqual(escape_latex("100%"), r"100\%")
        # The backslash must be replaced first, or it would escape its own escapes.
        self.assertEqual(escape_latex("\\"), r"\textbackslash{}")

    def test_prettify_names(self):
        self.assertEqual(prettify_metric_name("sq_dsc"), "SQ DSC")
        self.assertEqual(prettify_metric_name("pq"), "PQ")
        self.assertEqual(prettify_metric_name("n_ref_instances"), "N Ref Instances")
        self.assertEqual(prettify_metric_name("global_bin_dsc"), "Global Bin DSC")
        self.assertEqual(prettify_group_name("ungrouped"), "Ungrouped")
        self.assertEqual(prettify_group_name("spinal_cord"), "Spinal Cord")

    def test_base_metric_name(self):
        self.assertEqual(base_metric_name("sq_assd"), "assd")
        self.assertEqual(base_metric_name("pq_dsc"), "dsc")
        self.assertEqual(base_metric_name("global_bin_hd95"), "hd95")
        self.assertEqual(base_metric_name("t0.5_pq"), "pq")
        self.assertEqual(base_metric_name("autc_pq"), "pq")
        self.assertEqual(base_metric_name("pq"), "pq")

    def test_metric_direction(self):
        self.assertEqual(metric_direction("pq"), "increasing")
        self.assertEqual(metric_direction("sq_dsc"), "increasing")
        self.assertEqual(metric_direction("rec"), "increasing")
        self.assertEqual(metric_direction("sq_assd"), "decreasing")
        self.assertEqual(metric_direction("sq_hd95"), "decreasing")
        self.assertEqual(metric_direction("fp"), "decreasing")
        self.assertEqual(metric_direction("t0.5_pq"), "increasing")
        # A standard deviation has no "better" direction.
        self.assertIsNone(metric_direction("sq_dsc_std"))
        self.assertIsNone(metric_direction("made_up_metric"))

    def test_metric_arrow(self):
        self.assertEqual(metric_arrow("pq"), r"$\uparrow$")
        self.assertEqual(metric_arrow("sq_assd"), r"$\downarrow$")
        self.assertEqual(metric_arrow("made_up_metric"), "")

    def test_format_mean_std(self):
        self.assertEqual(format_mean_std(0.8421, 0.1134), r"$0.842 \pm 0.113$")
        self.assertEqual(format_mean_std(0.8421, 0.1134, ndigits=1), r"$0.8 \pm 0.1$")
        self.assertEqual(format_mean_std(0.8421, 0.1134, show_std=False), "$0.842$")
        self.assertEqual(
            format_mean_std(0.8421, 0.1134, bold=True),
            r"$\mathbf{0.842} \pm 0.113$",
        )
        self.assertEqual(format_mean_std(None, 0.1), "--")
        self.assertEqual(format_mean_std(float("nan"), 0.1), "--")

    def test_select_best_indices(self):
        self.assertEqual(select_best_indices([0.1, 0.9, 0.5], "increasing"), {1})
        self.assertEqual(select_best_indices([0.1, 0.9, 0.5], "decreasing"), {0})
        # Unknown direction -> nothing is marked.
        self.assertEqual(select_best_indices([0.1, 0.9], None), set())
        # Ties are all returned.
        self.assertEqual(select_best_indices([0.9, 0.9, 0.1], "increasing"), {0, 1})
        # Missing values are ignored, all-missing selects nothing.
        self.assertEqual(select_best_indices([None, 0.5], "increasing"), {1})
        self.assertEqual(select_best_indices([None, None], "increasing"), set())

    def test_render_latex_table_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            render_latex_table(["A", "B"], [("row", ["1"])])
        with self.assertRaises(ValueError):
            render_latex_table(
                ["A", "B"], [("row", ["1", "2"])], column_groups=[("G", 3)]
            )

    def test_render_latex_table_column_groups(self):
        table = render_latex_table(
            ["A", "B", "A", "B"],
            [("row", ["1", "2", "3", "4"])],
            column_groups=[("G1", 2), ("G2", 2)],
        )
        self.assertIn(r"\multicolumn{2}{c}{G1}", table)
        self.assertIn(r"\multicolumn{2}{c}{G2}", table)
        self.assertIn(r"\cmidrule(lr){2-3}", table)
        self.assertIn(r"\cmidrule(lr){4-5}", table)


class Test_Latex_Tables(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
        return super().setUp()

    def tearDown(self) -> None:
        if output_test_file.exists():
            os.remove(str(output_test_file))
        return super().tearDown()

    def test_default_table_uses_booktabs_and_tabularx(self):
        table = _make_statistic().get_latex_table()
        self.assertIn(LATEX_TABLE_REQUIREMENTS, table)
        self.assertIn(r"\begin{tabularx}{\textwidth}", table)
        self.assertIn(r"\toprule", table)
        self.assertIn(r"\midrule", table)
        self.assertIn(r"\bottomrule", table)
        self.assertNotIn(r"\hline", table)

    def test_basic_table_uses_plain_tabular(self):
        table = _make_statistic().get_latex_table(basic=True)
        self.assertIn(r"\begin{tabular}{", table)
        self.assertIn(r"\hline", table)
        self.assertNotIn(r"\toprule", table)
        self.assertNotIn("tabularx", table)
        # A plain tabular needs no extra packages, so no requirement comment.
        self.assertNotIn(LATEX_TABLE_REQUIREMENTS, table)

    def test_column_counts_are_consistent(self):
        for table in (
            _make_statistic().get_latex_table(),
            _make_statistic().get_latex_table(rows="metrics"),
            _make_statistic().get_latex_table(include_across_groups=True),
            make_latex_table_over_setups({"a": _make_statistic()}),
            make_latex_table_over_setups(
                {"a": _make_statistic(), "b": _make_statistic()}
            ),
        ):
            counts = {line.count("&") for line in _body_lines(table)}
            self.assertEqual(len(counts), 1, msg=table)

    def test_caption_and_label(self):
        table = _make_statistic().get_latex_table(caption="My caption", label="tab:x")
        self.assertIn(r"\caption{My caption}", table)
        self.assertIn(r"\label{tab:x}", table)

    def test_bold_best_respects_metric_direction(self):
        # vertebra has the higher pq AND the lower assd -> best in both columns.
        table = _make_statistic().get_latex_table()
        vertebra_row = next(
            line for line in table.splitlines() if line.startswith("Vertebra")
        )
        ivd_row = next(line for line in table.splitlines() if line.startswith("Ivd"))
        self.assertEqual(vertebra_row.count(r"\mathbf"), 2)
        self.assertEqual(ivd_row.count(r"\mathbf"), 0)

    def test_bold_best_can_be_disabled(self):
        table = _make_statistic().get_latex_table(bold_best=False)
        self.assertNotIn(r"\mathbf", table)

    def test_unknown_direction_is_never_bolded(self):
        stat = Panoptica_Statistic(
            subj_names=["s1", "s2"],
            value_dict={
                "a": {"made_up_metric": [1.0, 1.0]},
                "b": {"made_up_metric": [2.0, 2.0]},
            },
        )
        table = stat.get_latex_table()
        self.assertNotIn(r"\mathbf", table)
        self.assertNotIn(r"\uparrow", table)
        self.assertNotIn(r"\downarrow", table)

    def test_ties_bold_every_winner(self):
        stat = _make_statistic(pq_vertebra=(0.6, 0.5), pq_ivd=(0.6, 0.5))
        table = stat.get_latex_table(metrics="pq")
        self.assertEqual(table.count(r"\mathbf"), 2)

    def test_arrows(self):
        table = _make_statistic().get_latex_table()
        self.assertIn(r"PQ $\uparrow$", table)
        self.assertIn(r"SQ ASSD $\downarrow$", table)
        table_no_arrows = _make_statistic().get_latex_table(show_arrows=False)
        self.assertNotIn(r"\uparrow", table_no_arrows)
        self.assertNotIn(r"\downarrow", table_no_arrows)

    def test_show_std(self):
        self.assertIn(r"\pm", _make_statistic().get_latex_table())
        self.assertNotIn(r"\pm", _make_statistic().get_latex_table(show_std=False))

    def test_rows_metrics_transposes(self):
        table = _make_statistic().get_latex_table(rows="metrics")
        header = _body_lines(table)[0]
        self.assertIn("Vertebra", header)
        self.assertIn("Ivd", header)
        self.assertTrue(
            any(line.startswith("PQ ") for line in table.splitlines()), msg=table
        )

    def test_rows_invalid_value(self):
        with self.assertRaises(ValueError):
            _make_statistic().get_latex_table(rows="nonsense")

    def test_selection_and_alternate_names(self):
        table = _make_statistic().get_latex_table(
            metrics="pq",
            groups=["vertebra"],
            alternate_metricnames=["Panoptic Quality"],
            alternate_groupnames="Vertebrae",
        )
        self.assertIn("Panoptic Quality", table)
        self.assertIn("Vertebrae", table)
        self.assertNotIn("ASSD", table)

    def test_alternate_names_length_mismatch(self):
        with self.assertRaises(ValueError):
            _make_statistic().get_latex_table(alternate_groupnames=["only one"])

    def test_unknown_group_or_metric_raises(self):
        with self.assertRaises(KeyError):
            _make_statistic().get_latex_table(groups="nope")
        with self.assertRaises(KeyError):
            _make_statistic().get_latex_table(metrics="nope")

    def test_across_groups_row_is_added_but_not_compared(self):
        table = _make_statistic().get_latex_table(include_across_groups=True)
        across_row = next(
            line for line in table.splitlines() if line.startswith("Across groups")
        )
        self.assertIn(r"\pm", across_row)
        self.assertNotIn(r"\mathbf", across_row)

    def test_missing_values_render_placeholder(self):
        stat = Panoptica_Statistic(
            subj_names=["s1", "s2"],
            value_dict={"a": {"pq": [None, None]}, "b": {"pq": [0.5, 0.7]}},
        )
        table = stat.get_latex_table()
        self.assertIn("--", table)
        # Only the group with values can win.
        self.assertEqual(table.count(r"\mathbf"), 1)

    def test_std_columns_are_excluded_by_default(self):
        stat = Panoptica_Statistic(
            subj_names=["s1"],
            value_dict={"a": {"pq": [0.5], "sq_dsc": [0.6], "sq_dsc_std": [0.1]}},
        )
        table = stat.get_latex_table()
        self.assertIn("SQ DSC", table)
        self.assertNotIn("SQ DSC Std", table)
        # ... but can still be requested explicitly.
        self.assertIn("SQ DSC Std", stat.get_latex_table(metrics=["sq_dsc_std"]))

    def test_threshold_columns_are_excluded_by_default(self):
        stat = Panoptica_Statistic(
            subj_names=["s1"],
            value_dict={"a": {"pq": [0.5], "t0.5_pq": [0.4]}},
        )
        header = _body_lines(stat.get_latex_table())[0]
        self.assertEqual(header.count("&"), 1)

    def test_output_file_matches_return_value(self):
        table = _make_statistic().get_latex_table(output_file=output_test_file)
        self.assertTrue(output_test_file.exists())
        self.assertEqual(output_test_file.read_text(encoding="utf-8"), table)


class Test_Latex_Table_Over_Setups(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
        self.stat_a = _make_statistic()
        self.stat_b = _make_statistic(
            pq_vertebra=(0.7, 0.75),
            pq_ivd=(0.65, 0.7),
            assd_vertebra=(1.0, 1.5),
            assd_ivd=(3.0, 3.5),
        )
        self.setups = {"baseline": self.stat_a, "ours": self.stat_b}
        return super().setUp()

    def tearDown(self) -> None:
        if output_test_file.exists():
            os.remove(str(output_test_file))
        return super().tearDown()

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            make_latex_table_over_setups({})

    def test_row_order_follows_dict_order(self):
        table = make_latex_table_over_setups(self.setups)
        rows = [line for line in table.splitlines() if line.endswith(r"\\")]
        body = [line for line in rows if line.startswith(("baseline", "ours"))]
        self.assertEqual(len(body), 2)
        self.assertTrue(body[0].startswith("baseline"))
        self.assertTrue(body[1].startswith("ours"))

    def test_multiple_groups_use_spanning_header(self):
        table = make_latex_table_over_setups(self.setups)
        self.assertIn(r"\multicolumn{2}{c}{Vertebra}", table)
        self.assertIn(r"\multicolumn{2}{c}{Ivd}", table)
        self.assertIn(r"\cmidrule(lr){2-3}", table)

    def test_single_group_has_no_spanning_header(self):
        table = make_latex_table_over_setups(self.setups, groups="vertebra")
        self.assertNotIn(r"\multicolumn", table)
        self.assertNotIn(r"\cmidrule", table)

    def test_basic_flattens_group_and_metric_labels(self):
        table = make_latex_table_over_setups(self.setups, basic=True)
        self.assertNotIn(r"\multicolumn", table)
        self.assertIn(r"Vertebra / PQ $\uparrow$", table)

    def test_bold_marks_the_best_setup_per_metric(self):
        table = make_latex_table_over_setups(self.setups, groups="vertebra")
        baseline_row = next(
            line for line in table.splitlines() if line.startswith("baseline")
        )
        ours_row = next(line for line in table.splitlines() if line.startswith("ours"))
        # baseline wins pq (0.85 > 0.725), ours wins assd (1.25 < 2.5).
        self.assertEqual(baseline_row.count(r"\mathbf"), 1)
        self.assertEqual(ours_row.count(r"\mathbf"), 1)
        self.assertIn(r"$\mathbf{0.850}", baseline_row)
        self.assertIn(r"$\mathbf{1.250}", ours_row)

    def test_rows_metrics_puts_setups_in_the_header(self):
        table = make_latex_table_over_setups(self.setups, rows="metrics")
        header = _body_lines(table)[-3]
        self.assertIn("baseline", header)
        self.assertIn("ours", header)
        self.assertTrue(any(line.startswith("PQ ") for line in table.splitlines()))

    def test_rows_invalid_value(self):
        with self.assertRaises(ValueError):
            make_latex_table_over_setups(self.setups, rows="nonsense")

    def test_missing_metric_or_group_raises(self):
        with self.assertRaises(ValueError):
            make_latex_table_over_setups(self.setups, metrics="nope")
        with self.assertRaises(ValueError):
            make_latex_table_over_setups(self.setups, groups="nope")

    def test_default_metrics_are_intersected_over_setups(self):
        extra = Panoptica_Statistic(
            subj_names=["s1", "s2"],
            value_dict={
                "vertebra": {"pq": [0.1, 0.2]},
                "ivd": {"pq": [0.1, 0.2]},
            },
        )
        table = make_latex_table_over_setups({"a": self.stat_a, "b": extra})
        self.assertIn("PQ", table)
        self.assertNotIn("ASSD", table)

    def test_output_file_matches_return_value(self):
        table = make_latex_table_over_setups(self.setups, output_file=output_test_file)
        self.assertEqual(output_test_file.read_text(encoding="utf-8"), table)


if __name__ == "__main__":
    unittest.main()
