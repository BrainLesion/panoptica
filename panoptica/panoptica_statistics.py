import csv
import numpy as np
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    print(e)
    print("OPTIONAL PACKAGE MISSING")


class ValueSummary:
    def __init__(self, value_list: list[float]) -> None:
        self.__value_list = value_list
        if len(value_list) == 0:
            self.__avg = np.nan
            self.__std = np.nan
            self.__min = np.nan
            self.__max = np.nan
        else:
            self.__avg = float(np.average(value_list))
            self.__std = float(np.std(value_list))
            self.__min = min(value_list)
            self.__max = max(value_list)

    @property
    def values(self) -> list[float]:
        return self.__value_list

    @property
    def avg(self) -> float:
        return self.__avg

    @property
    def std(self) -> float:
        return self.__std

    @property
    def min(self) -> float:
        return self.__min

    @property
    def max(self) -> float:
        return self.__max

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"[{round(self.min, 3)}, {round(self.max, 3)}], avg = {round(self.avg, 3)} +- {round(self.std, 3)}"


class Panoptica_Statistic:

    def __init__(
        self,
        subj_names: list[str],
        value_dict: dict[str, dict[str, list[float]]],
    ) -> None:
        self.__subj_names = subj_names
        self.__value_dict = value_dict

        self.__groupnames = list(value_dict.keys())
        self.__metricnames = list(value_dict[self.__groupnames[0]].keys())

        # assert length of everything
        for g in self.groupnames:
            assert len(self.metricnames) == len(
                list(value_dict[g].keys())
            ), f"Group {g}, has inconsistent number of metrics, got {len(list(value_dict[g].keys()))} but expected {len(self.metricnames)}"
            for m in self.metricnames:
                assert len(self.get(g, m)) == len(
                    self.subjectnames
                ), f"Group {g}, m {m} has not right subjects, got {len(self.get(g, m))}, expected {len(self.subjectnames)}"

    @property
    def subjectnames(self):
        return self.__subj_names

    @property
    def groupnames(self):
        return self.__groupnames

    @property
    def metricnames(self):
        return self.__metricnames

    @classmethod
    def from_file(cls, file: str):
        # check integrity of header and so on
        with open(str(file), "r", encoding="utf8", newline="") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")

            rows = [row for row in rd]

        header = rows[0]
        assert (
            header[0] == "subject_name"
        ), "First column is not subject_names, something wrong with the file?"

        keys_in_order = list([tuple(c.split("-")) for c in header[1:]])
        metric_names = []
        for k in keys_in_order:
            if k[1] not in metric_names:
                metric_names.append(k[1])
        group_names = list(set([k[0] for k in keys_in_order]))

        print(f"Found {len(rows)-1} entries")
        print(f"Found metrics: {metric_names}")
        print(f"Found groups: {group_names}")

        # initialize collection
        subj_names = []
        # list of floats in order fo subject_names
        # from group to metric to list of values
        value_dict: dict[str, dict[str, list[float]]] = {}

        # now load entries
        for r in rows[1:]:
            sn = r[0]  # subject_name
            subj_names.append(sn)

            for idx, value in enumerate(r[1:]):
                group_name, metric_name = keys_in_order[idx]
                if group_name not in value_dict:
                    value_dict[group_name] = {m: [] for m in metric_names}

                if len(value) > 0:
                    value = float(value)
                    if value is not None and not np.isnan(value) and value != np.inf:
                        value_dict[group_name][metric_name].append(float(value))
                    else:
                        value_dict[group_name][metric_name].append(None)
                else:
                    value_dict[group_name][metric_name].append(None)

        return Panoptica_Statistic(subj_names=subj_names, value_dict=value_dict)

    def _assertgroup(self, group):
        assert (
            group in self.__groupnames
        ), f"group {group} not existent, only got groups {self.__groupnames}"

    def _assertmetric(self, metric):
        assert (
            metric in self.__metricnames
        ), f"metric {metric} not existent, only got metrics {self.__metricnames}"

    def _assertsubject(self, subjectname):
        assert (
            subjectname in self.__subj_names
        ), f"subject {subjectname} not in list of subjects, got {self.__subj_names}"

    def get(self, group, metric, remove_nones: bool = False) -> list[float]:
        """Returns the list of values for given group and metric

        Args:
            group (_type_): _description_
            metric (_type_): _description_

        Returns:
            list[float]: _description_
        """
        self._assertgroup(group)
        self._assertmetric(metric)

        assert (
            group in self.__value_dict and metric in self.__value_dict[group]
        ), f"Values not found for group {group} and metric {metric} evem though they should!"
        if not remove_nones:
            return self.__value_dict[group][metric]
        return [i for i in self.__value_dict[group][metric] if i is not None]

    def get_one_subject(self, subjectname: str):
        """Gets the values for ONE subject for each group and metric

        Args:
            subjectname (str): _description_

        Returns:
            _type_: _description_
        """
        self._assertsubject(subjectname)
        sidx = self.__subj_names.index(subjectname)
        return {
            g: {m: self.get(g, m)[sidx] for m in self.__metricnames}
            for g in self.__groupnames
        }

    def get_one_metric(self, metricname: str):
        """Gets the dictionary mapping the group to the metrics specified

        Args:
            metricname (str): _description_

        Returns:
            _type_: _description_
        """
        self._assertmetric(metricname)
        return {g: self.get(g, metricname) for g in self.__groupnames}

    def get_one_group(self, groupname: str):
        """Gets the dictionary mapping metric to values for ONE group

        Args:
            groupname (str): _description_

        Returns:
            _type_: _description_
        """
        self._assertgroup(groupname)
        return {m: self.get(groupname, m) for m in self.__metricnames}

    def get_across_groups(self, metric) -> list[float]:
        """Given metric, gives list of all values (even across groups!) Treat with care!

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        values = []
        for g in self.__groupnames:
            values += self.get(g, metric)
        return values

    def get_summary_across_groups(self) -> dict[str, ValueSummary]:
        """Calculates the average and std over all groups (so group-wise avg first, then average over those)

        Returns:
            dict[str, tuple[float, float]]: _description_
        """
        summary_dict = {}
        for m in self.__metricnames:
            value_list = [self.get_summary(g, m).avg for g in self.__groupnames]
            assert len(value_list) == len(self.__groupnames)
            summary_dict[m] = ValueSummary(value_list)
        return summary_dict

    def get_summary_dict(
        self, include_across_group: bool = True
    ) -> dict[str, dict[str, ValueSummary]]:
        summary_dict = {
            g: {m: self.get_summary(g, m) for m in self.__metricnames}
            for g in self.__groupnames
        }
        if include_across_group:
            summary_dict["across_groups"] = self.get_summary_across_groups()
        return summary_dict

    def get_summary(self, group, metric) -> ValueSummary:
        values = self.get(group, metric, remove_nones=True)
        return ValueSummary(values)

    def print_summary(
        self,
        ndigits: int = 3,
        only_across_groups: bool = True,
    ):
        summary = self.get_summary_dict(include_across_group=only_across_groups)
        print()
        groups = list(summary.keys())
        if only_across_groups:
            groups = ["across_groups"]
        for g in groups:
            print(f"Group {g}:")
            for m in self.__metricnames:
                avg, std = summary[g][m].avg, summary[g][m].std
                print(m, ":", round(avg, ndigits), "+-", round(std, ndigits))
            print()

    def get_summary_figure(
        self,
        metric: str,
        manual_metric_range: None | tuple[float, float] = None,
        name_method: str = "Structure",
        horizontal: bool = True,
        sort: bool = True,
        title: str = "",
    ):
        """Returns a figure object that shows the given metric for each group and its std

        Args:
            metric (str): _description_
            horizontal (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        data_plot = {
            g: np.asarray(self.get(g, metric, remove_nones=True))
            for g in self.__groupnames
        }
        if manual_metric_range is not None:
            assert manual_metric_range[0] < manual_metric_range[1], manual_metric_range
            change = (manual_metric_range[1] - manual_metric_range[0]) / 100
            manual_metric_range = (
                manual_metric_range[0] - change,
                manual_metric_range[1] + change,
            )
        return plot_box(
            data=data_plot,
            orientation_horizontal=horizontal,
            name_method=name_method,
            name_metric=metric,
            sort=sort,
            figure_title=title,
            manual_metric_range=manual_metric_range,
        )


def make_autc_plots(
    statistics_dict: dict[str | int | float, Panoptica_Statistic],
    metric: str,
    groups: list[str] | str | None = None,
    alternate_groupnames: list[str] | str | None = None,
):
    raise NotImplementedError("AUTC plots currently in works")


def make_curve_over_setups(
    statistics_dict: dict[str | int | float, Panoptica_Statistic],
    metric: str,
    groups: list[str] | str | None = None,
    alternate_groupnames: list[str] | str | None = None,
    fig: go.Figure | None = None,
    plot_as_barchart=True,
    plot_std: bool = True,
    figure_title: str = "",
    width: int = 850,
    height: int = 1200,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    manual_metric_range: None | tuple[float, float] = None,
):
    # TODO make this flexibel whether the second grouping are the groups or metrics?
    if groups is None:
        groups = list(statistics_dict.values())[0].groupnames
    #
    if isinstance(groups, str):
        groups = [groups]
    if isinstance(alternate_groupnames, str):
        alternate_groupnames = [alternate_groupnames]

    assert (
        plot_as_barchart or len(groups) == 1
    ), "When plotting without barcharts, you cannot plot more than one group at the same time"
    #
    for setupname, stat in statistics_dict.items():
        assert (
            metric in stat.metricnames
        ), f"metric {metric} not in statistic obj {setupname}"

    setupnames = list(statistics_dict.keys())
    convert_x_to_digit = True
    for s in setupnames:
        if not str(s).isdigit():
            convert_x_to_digit = False
            break

    # If X (setupnames) are digits only, plot as digits
    if convert_x_to_digit:
        X = [float(s) for s in setupnames]
    else:
        X = setupnames

    if fig is None:
        fig = go.Figure()

    # Y values are average metric values in that group and metric
    for idx, g in enumerate(groups):
        Y = [
            ValueSummary(stat.get(g, metric, remove_nones=True)).avg
            for stat in statistics_dict.values()
        ]

        name = g if alternate_groupnames is None else alternate_groupnames[idx]

        if plot_std:
            Ystd = [
                ValueSummary(stat.get(g, metric, remove_nones=True)).std
                for stat in statistics_dict.values()
            ]
        else:
            Ystd = None

        if plot_as_barchart:
            fig.add_trace(
                go.Bar(name=name, x=X, y=Y, error_y=dict(type="data", array=Ystd))
            )
        else:
            # lineplot
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=Y,
                    mode="lines+markers",
                    name="lines+markers",
                    error_y=dict(type="data", array=Ystd),
                )
            )

    fig.update_layout(
        autosize=False,
        barmode="group",
        width=width,
        height=height,
        showlegend=True,
        yaxis_title=metric if yaxis_title is None else yaxis_title,
        xaxis_title=(
            "Different setups and groups" if xaxis_title is None else xaxis_title
        ),
        font={"family": "Arial"},
        title=figure_title,
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="gray")
    if manual_metric_range is not None:
        fig.update_yaxes(range=[manual_metric_range[0], manual_metric_range[1]])
    return fig


def _flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def plot_box(
    data: dict[str, np.ndarray],
    sort=True,
    orientation_horizontal: bool = True,  # "h"
    name_method: str = "Structure",
    name_metric: str = "Dice-Score",
    figure_title: str = "",
    width=850,
    height=1200,
    manual_metric_range: None | tuple[float, float] = None,
):
    xaxis_title = name_metric if orientation_horizontal else name_method
    yaxis_title = name_metric if not orientation_horizontal else name_method
    orientation = "h" if orientation_horizontal else "v"

    data = {e.replace("_", " "): v for e, v in data.items()}
    df_data = pd.DataFrame(
        {
            name_method: _flatten_extend([([e] * len(y0)) for e, y0 in data.items()]),
            name_metric: np.concatenate([*data.values()], 0),
        }
    )
    if sort:
        df_by_spec_count = df_data.groupby(name_method).mean()
        df_by_spec_count = dict(df_by_spec_count[name_metric].items())
        df_data["mean"] = df_data[name_method].apply(
            lambda x: df_by_spec_count[x] * (1 if orientation_horizontal else -1)
        )
        df_data = df_data.sort_values(by="mean")
    if not orientation_horizontal:
        fig = px.strip(
            df_data,
            x=name_method,
            y=name_metric,
            stripmode="overlay",
            orientation=orientation,
        )
        fig.update_traces(marker={"size": 5, "color": "#555555"})
        for e in data.keys():
            fig.add_trace(
                go.Box(
                    y=df_data.query(f'{name_method} == "{e}"')[name_metric],
                    name=e,
                    orientation=orientation,
                )
            )
    else:
        fig = px.strip(
            df_data,
            y=name_method,
            x=name_metric,
            stripmode="overlay",
            orientation=orientation,
        )
        fig.update_traces(marker={"size": 5, "color": "#555555"})
        for e in data.keys():
            fig.add_trace(
                go.Box(
                    x=df_data.query(f'{name_method} == "{e}"')[name_metric],
                    name=e,
                    orientation=orientation,
                    boxpoints=False,
                )
            )
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        showlegend=False,
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        font={"family": "Arial"},
        title=figure_title,
    )
    if manual_metric_range is not None:
        if orientation == "h":
            fig.update_xaxes(range=[manual_metric_range[0], manual_metric_range[1]])
        else:
            fig.update_yaxes(range=[manual_metric_range[0], manual_metric_range[1]])
    fig.update_traces(orientation=orientation)
    return fig
