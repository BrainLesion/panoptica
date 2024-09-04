import csv
import numpy as np
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    print(e)
    print("OPTIONAL PACKAGE MISSING")


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
                    if not np.isnan(value) and value != np.inf:
                        value_dict[group_name][metric_name].append(float(value))

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

    def get(self, group, metric) -> list[float]:
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
        return self.__value_dict[group][metric]

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

    def get_across_groups(self, metric):
        """Given metric, gives list of all values (even across groups!) Treat with care!

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        values = []
        for g in self.__groupnames:
            values.append(self.get(g, metric))
        return values

    def get_summary_dict(self):
        return {
            g: {m: self.get_summary(g, m) for m in self.__metricnames}
            for g in self.__groupnames
        }

    def get_summary(self, group, metric):
        # TODO maybe more here? range, stuff like that
        return self.avg_std(group, metric)

    def avg_std(self, group, metric) -> tuple[float, float]:
        values = self.get(group, metric)
        avg = float(np.average(values))
        std = float(np.std(values))
        return (avg, std)

    def print_summary(self, ndigits: int = 3):
        summary = self.get_summary_dict()
        print()
        for g in self.__groupnames:
            print(f"Group {g}:")
            for m in self.__metricnames:
                avg, std = summary[g][m]
                print(m, ":", round(avg, ndigits), "+-", round(std, ndigits))
            print()

    def get_summary_figure(
        self,
        metric: str,
        horizontal: bool = True,
        # title overwrite?
    ):
        """Returns a figure object that shows the given metric for each group and its std

        Args:
            metric (str): _description_
            horizontal (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        orientation = "h" if horizontal else "v"
        data_plot = {g: np.asarray(self.get(g, metric)) for g in self.__groupnames}
        return plot_box(
            data=data_plot,
            orientation=orientation,
            score=metric,
        )

    # groupwise or in total
    # Mean over instances
    # mean over subjects
    # give below/above percentile of metric (the names)
    # make auc curve as plot


def make_curve_over_setups(
    statistics_dict: dict[str | int | float, Panoptica_Statistic],
    metric: str,
    groups: list[str] | str | None = None,
    alternate_groupnames: list[str] | str | None = None,
    fig: None = None,
    plot_dotsize: int | None = None,
    plot_lines: bool = True,
):
    if groups is None:
        groups = list(statistics_dict.values())[0].groupnames
    #
    if isinstance(groups, str):
        groups = [groups]
    if isinstance(alternate_groupnames, str):
        alternate_groupnames = [alternate_groupnames]
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
        X = range(len(setupnames))

    if fig is None:
        fig = plt.figure()

    if not convert_x_to_digit:
        plt.xticks(X, setupnames)

    plt.ylabel("Average " + metric)
    plt.grid("major")
    # Y values are average metric values in that group and metric
    for idx, g in enumerate(groups):
        Y = [stat.avg_std(g, metric)[0] for stat in statistics_dict.values()]

        if plot_lines:
            plt.plot(
                X,
                Y,
                label=g if alternate_groupnames is None else alternate_groupnames[idx],
            )

        if plot_dotsize is not None:
            plt.scatter(X, Y, s=plot_dotsize)

    plt.legend()
    return fig


def _flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def plot_box(
    data: dict[str, np.ndarray],
    sort=True,
    orientation="h",
    # graph_name: str = "Structure",
    score: str = "Dice-Score",
    width=850,
    height=1200,
    yaxis_title=None,
    xaxis_title=None,
):
    graph_name: str = "Structure"

    if xaxis_title is None:
        xaxis_title = score if orientation == "h" else graph_name
    if yaxis_title is None:
        yaxis_title = score if orientation != "h" else graph_name

    data = {e.replace("_", " "): v for e, v in data.items()}
    df_data = pd.DataFrame(
        {
            graph_name: _flatten_extend([([e] * len(y0)) for e, y0 in data.items()]),
            score: np.concatenate([*data.values()], 0),
        }
    )
    if sort:
        df_by_spec_count = df_data.groupby(graph_name).mean()
        df_by_spec_count = dict(df_by_spec_count[score].items())
        df_data["mean"] = df_data[graph_name].apply(
            lambda x: df_by_spec_count[x] * (1 if orientation == "h" else -1)
        )
        df_data = df_data.sort_values(by="mean")
    if orientation == "v":
        fig = px.strip(
            df_data, x=graph_name, y=score, stripmode="overlay", orientation=orientation
        )
        fig.update_traces(marker={"size": 5, "color": "#555555"})
        for e in data.keys():
            fig.add_trace(
                go.Box(
                    y=df_data.query(f'{graph_name} == "{e}"')[score],
                    name=e,
                    orientation=orientation,
                )
            )
    else:
        fig = px.strip(
            df_data, y=graph_name, x=score, stripmode="overlay", orientation=orientation
        )
        fig.update_traces(marker={"size": 5, "color": "#555555"})
        for e in data.keys():
            fig.add_trace(
                go.Box(
                    x=df_data.query(f'{graph_name} == "{e}"')[score],
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
    )
    fig.update_traces(orientation=orientation)
    return fig
