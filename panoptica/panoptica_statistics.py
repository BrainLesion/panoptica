from panoptica.utils import is_instance_row
from panoptica.utils import format_threshold_key
from panoptica.utils import is_threshold_key
from panoptica.utils import parse_threshold_key
import csv
import numpy as np
import warnings
from pathlib import Path

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    print(e)
    print("OPTIONAL PACKAGE MISSING")


class FloatDistribution:
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

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("Only integer indexing supported for FloatDistribution")
        return self.__value_list[key]

    def __setitem__(self, key, value):
        raise TypeError("FloatDistribution is immutable, cannot set item")

    @property
    def values(self) -> list[float]:
        return list(self.__value_list)

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

    def z_score(self, value: float) -> float:
        """Calculates the z-score of a value based on the summary statistics."""
        if self.std == 0:
            return 0.0
        return (value - self.avg) / self.std

    def __repr__(self):
        return str(self)

    def get_string_repr(self, ndigits: int = 3):
        return f"[{round(self.min, ndigits)}, {round(self.max, ndigits)}], avg = {round(self.avg, ndigits)} +- {round(self.std, ndigits)}"

    def __str__(self, ndigits: int = 3):
        return self.get_string_repr(ndigits)


class ValueSummary(FloatDistribution):
    """Deprecated alias for FloatDistribution."""

    def __init__(self, value_list: list[float]) -> None:
        warnings.warn(
            "ValueSummary is deprecated and will be removed in a future release. Use FloatDistribution instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(value_list)


class Panoptica_Statistic:
    def __init__(
        self,
        subj_names: list[str],
        value_dict: dict[str, dict[str, list[float]]],
    ) -> None:
        """_summary_

        Args:
            subj_names (list[str]): List of subject names in the same order as the list of values passed in value_dict
            value_dict (dict[str, dict[str, list[float]]]): Mapping Group to Metric to list of values
        """
        self.__subj_names = subj_names
        self.__value_dict = value_dict

        self.__groupnames = list(value_dict.keys())
        self.__metricnames = list(value_dict[self.__groupnames[0]].keys())
        self.__threshold_map: dict[str, list[float]] = {}

        for m in self.__metricnames:
            parsed = parse_threshold_key(m)
            if parsed:
                threshold_val, base_metric = parsed
                if base_metric not in self.__threshold_map:
                    self.__threshold_map[base_metric] = []
                self.__threshold_map[base_metric].append(threshold_val)

        for m in self.__threshold_map:
            self.__threshold_map[m].sort()

        # check length of everything
        for g in self.groupnames:
            if len(self.metricnames) != len(list(value_dict[g].keys())):
                raise ValueError(
                    f"Group {g}, has inconsistent number of metrics, got {len(list(value_dict[g].keys()))} but expected {len(self.metricnames)}"
                )
            for m in self.metricnames:
                if len(self.get(g, m)) != len(self.subjectnames):
                    raise ValueError(
                        f"Group {g}, m {m} has not right subjects, got {len(self.get(g, m))}, expected {len(self.subjectnames)}"
                    )

    @property
    def subjectnames(self):
        return self.__subj_names

    @property
    def groupnames(self):
        return self.__groupnames

    @property
    def metricnames(self):
        return self.__metricnames

    @property
    def master_subjects(self) -> list[str]:
        """Returns only the primary subject names (ignoring instance rows)."""
        return [sn for sn in self.__subj_names if not is_instance_row(sn)]

    @property
    def instance_subjects(self) -> list[str]:
        """Returns only the individual instance rows."""
        return [sn for sn in self.__subj_names if is_instance_row(sn)]
    
    @property
    def base_metric_names(self) -> list[str]:
        """Returns metric names that are not thresholded"""
        return [m for m in self.__metricnames if not is_threshold_key(m)]

    def master_values(self, values: list[float | None]) -> list[float]:
        """Pair each value with its subject name and filter out the instance rows."""
        return [
            val
            for sn, val in zip(self.__subj_names, values)
            if not is_instance_row(sn) and val is not None
        ]
        
    def get_thresholds_for_metric(self, metric: str) -> list[float]:
        """Returns available thresholds for a specific metric (e.g. 'pq')."""
        return self.__threshold_map.get(metric, [])

    @classmethod
    def from_file(cls, file: str | Path, verbose: bool = True):
        if isinstance(file, Path):
            file = str(file)
        if not file.endswith(".tsv"):
            file += ".tsv"
        # check integrity of header and so on
        with open(str(file), "r", encoding="utf8", newline="") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t", lineterminator="\n")

            rows = [row for row in rd]

        header = rows[0]
        if header[0] != "subject_name":
            raise ValueError(
                "First column is not subject_names, something wrong with the file?"
            )

        keys_in_order = list([tuple(c.split("-", maxsplit=1)) for c in header[1:]])
        keys_in_order = list(
            k if len(k) == 2 else ("ungrouped", k[0]) for k in keys_in_order
        )
        metric_names = []
        for k in keys_in_order:
            if k[1] not in metric_names:
                metric_names.append(k[1])
        group_names = list(set([k[0] for k in keys_in_order]))

        if verbose:
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
        if group not in self.__groupnames:
            raise KeyError(
                f"group {group} not existent, only got groups {self.__groupnames}"
            )

    def _assertmetric(self, metric):
        if metric not in self.__metricnames:
            raise KeyError(
                f"metric {metric} not existent, only got metrics {self.__metricnames}"
            )

    def _assertsubject(self, subjectname):
        if subjectname not in self.__subj_names:
            raise KeyError(
                f"subject {subjectname} not in list of subjects, got {self.__subj_names}"
            )

    def _remove_subject(self, subjectname):
        self._assertsubject(subjectname)
        sidx = self.__subj_names.index(subjectname)
        self.__subj_names.pop(sidx)
        for g in self.__groupnames:
            for m in self.__metricnames:
                self.__value_dict[g][m].pop(sidx)

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

        if not (group in self.__value_dict and metric in self.__value_dict[group]):
            raise KeyError(
                f"Values not found for group {group} and metric {metric} even though they should!"
            )
        if not remove_nones:
            return self.__value_dict[group][metric]
        return [i for i in self.__value_dict[group][metric] if i is not None]

    def get_dict(self, group, metric, remove_nones, sort_ascending: bool = True):
        values = self.get(group, metric, remove_nones=False)
        if remove_nones:
            vdict = {
                self.__subj_names[i]: values[i]
                for i in range(len(values))
                if values[i] is not None
            }
        else:
            vdict = {self.__subj_names[i]: values[i] for i in range(len(values))}
        vdict = dict(
            sorted(vdict.items(), key=lambda x: x[1], reverse=not sort_ascending)
        )
        return vdict

    def get_best_worst_k_entries(
        self,
        groups: list[str] | str | None = None,
        metrics: list[str] | str | None = None,
        k: int = 3,
    ):
        if groups is None:
            groups = self.__groupnames
        if metrics is None:
            metrics = self.__metricnames

        if isinstance(groups, str):
            groups = [groups]
        if isinstance(metrics, str):
            metrics = [metrics]

        best_dict = {}
        worst_dict = {}

        for g in groups:
            self._assertgroup(g)
            for m in metrics:
                self._assertmetric(m)

                d = self.get_dict(g, m, remove_nones=True, sort_ascending=False)
                best_dict[(g, m)] = {k: d[k] for k in list(d.keys())[:k]}
                worst_dict[(g, m)] = {k: d[k] for k in list(d.keys())[-k:]}
        return best_dict, worst_dict

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

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the statistic to a pandas dataframe

        Returns:
            pd.DataFrame: _description_
        """
        data = []
        for subj in self.__subj_names:
            subj_values = self.get_one_subject(subj)
            for g in self.__groupnames:
                entry = {"subject_name": subj}
                entry["group"] = g
                for m in self.__metricnames:
                    entry[m] = subj_values[g][m]
                data.append(entry)
        df = pd.DataFrame(data)
        return df

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

    def get_subject_wise_paired_values_to(
        self, other: "Panoptica_Statistic", group: str, metric: str
    ) -> tuple[list[str], list[float], list[float]]:
        """Calculates the subject-wise paired values in metric for given group to another Panoptica_Statistic object

        Args:
            other (Panoptica_Statistic): _description_
            group (str): _description_
            metric (str): _description_
        """
        self._assertgroup(group)
        self._assertmetric(metric)
        other._assertgroup(group)
        other._assertmetric(metric)

        if len(self.__subj_names) != len(other.__subj_names):
            raise ValueError(
                "Length of Subject names do not match between the two Panoptica_Statistic objects!"
            )
        if set(self.__subj_names) != set(other.__subj_names):
            raise ValueError(
                "Subject names do not match between the two Panoptica_Statistic objects!"
            )

        self_values = []
        other_values = []
        for subj in self.__subj_names:
            self._assertsubject(subj)
            other._assertsubject(subj)

            sidx_self = self.__subj_names.index(subj)
            sidx_other = other.__subj_names.index(subj)

            val_self = self.get(group, metric)[sidx_self]
            val_other = other.get(group, metric)[sidx_other]

            self_values.append(val_self)
            other_values.append(val_other)
        return list(self.__subj_names), self_values, other_values

    def get_subject_wise_difference_to(
        self, other: "Panoptica_Statistic", group: str, metric: str
    ) -> dict[str, float | None]:
        """Calculates the subject-wise difference in metric for given group to another Panoptica_Statistic object

        Args:
            other (Panoptica_Statistic): _description_
            group (str): _description_
            metric (str): _description_
        Returns:
            dict[str, float]: _description_
        """
        subj_names, self_v, other_v = self.get_subject_wise_paired_values_to(
            other, group, metric
        )
        diff_dict = {}
        for subj, val_self, val_other in zip(subj_names, self_v, other_v):
            if val_self is not None and val_other is not None:
                diff_dict[subj] = val_self - val_other
            else:
                diff_dict[subj] = None
        return diff_dict

    def get_summary_across_groups(self) -> dict[str, FloatDistribution]:
        """Calculates the average and std over all groups (so group-wise avg first, then average over those)

        Returns:
            dict[str, tuple[float, float]]: _description_
        """
        summary_dict = {}
        for m in self.__metricnames:
            value_list = [self.get_summary(g, m).avg for g in self.__groupnames]
            if len(value_list) != len(self.__groupnames):
                raise ValueError(
                    f"Unexpected mismatch in value_list length for metric {m}"
                )
            summary_dict[m] = FloatDistribution(value_list)
        return summary_dict

    def get_summary_dict(
        self, include_across_group: bool = True
    ) -> dict[str, dict[str, FloatDistribution]]:
        summary_dict = {
            g: {m: self.get_summary(g, m) for m in self.__metricnames}
            for g in self.__groupnames
        }
        if include_across_group:
            summary_dict["across_groups"] = self.get_summary_across_groups()
        return summary_dict

    def print_summary(
        self,
        ndigits: int = 3,
        only_across_groups: bool = True,
        include_thresholds: bool = False,
    ):
        summary = self.get_summary_dict(include_across_group=only_across_groups)
        print()
        groups = list(summary.keys())
        if only_across_groups:
            groups = ["across_groups"]
        for g in groups:
            print(f"Group {g}:")
            metrics_to_show = (
                self.__metricnames if include_thresholds else self.base_metric_names
            )
            for m in metrics_to_show:
                avg, std = summary[g][m].avg, summary[g][m].std
                print(m, ":", round(avg, ndigits), "+-", round(std, ndigits))
            print()

    def get_summary(self, group, metric, master_only: bool = True) -> FloatDistribution:
        """Gets a FloatDistribution for a given group and metric.
        If master_only is True, ignores individual instance rows to prevent double counting.
        """
        all_values = self.get(group, metric, remove_nones=False)

        if master_only:
            filtered_values = self.master_values(all_values)
        else:
            filtered_values = [val for val in all_values if val is not None]

        return FloatDistribution(filtered_values)

    def get_summary_figure(
        self,
        metric: str,
        groups: list[str] | str | None = None,
        manual_metric_range: None | tuple[float, float] = None,
        name_method: str = "Structure",
        horizontal: bool = True,
        sort: bool = True,
        title: str = "",
        master_only: bool = True,
    ):
        """Returns a figure object that shows the given metric for each group and its std"""
        if groups is None:
            groups = self.__groupnames
        if isinstance(groups, str):
            groups = [groups]

        data_plot = {}
        for g in groups:
            all_values = self.get(g, metric, remove_nones=False)
            if master_only:
                filtered_values = self.master_values(all_values)
            else:
                filtered_values = [val for val in all_values if val is not None]
            data_plot[g] = np.asarray(filtered_values)

        if manual_metric_range is not None:
            if manual_metric_range[0] >= manual_metric_range[1]:
                raise ValueError(
                    f"manual_metric_range must have lower bound less than upper bound, got {manual_metric_range}"
                )
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
    fig: go.Figure | None = None,
    plot_std: bool = True,
    figure_title: str = "",
    width: int = 850,
    height: int = 1200,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    manual_metric_range: None | tuple[float, float] = None,
) -> go.Figure:
    if groups is None:
        groups = list(statistics_dict.values())[0].groupnames
    if isinstance(groups, str):
        groups = [groups]
    if isinstance(alternate_groupnames, str):
        alternate_groupnames = [alternate_groupnames]
    if alternate_groupnames is not None and len(alternate_groupnames) != len(groups):
        raise ValueError(
            f"alternate_groupnames has length {len(alternate_groupnames)} but groups has length {len(groups)}; they must match."
        )

    if fig is None:
        fig = go.Figure()

    for setupname, stat in statistics_dict.items():
        thresholds = stat.get_thresholds_for_metric(metric)

        if not thresholds:
            print(f"Warning: No threshold data found for '{metric}' in '{setupname}'.")
            continue

        X = thresholds
        for idx, g in enumerate(groups):
            name = g if alternate_groupnames is None else alternate_groupnames[idx]

            if len(statistics_dict) == 1 and len(groups) == 1:
                legend_name = str(name)
            elif len(groups) == 1:
                legend_name = str(setupname)
            else:
                legend_name = f"{setupname} - {name}"

            Y = [
                FloatDistribution(
                    stat.get(g, format_threshold_key(t, metric), remove_nones=True)
                ).avg
                for t in thresholds
            ]

            if plot_std:
                Ystd = [
                    FloatDistribution(
                        stat.get(g, format_threshold_key(t, metric), remove_nones=True)
                    ).std
                    for t in thresholds
                ]
                error_y = dict(type="data", array=Ystd)
            else:
                error_y = None

            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=Y,
                    mode="lines+markers",
                    name=legend_name,
                    error_y=error_y,
                )
            )

    # Reuse the exact layout from make_curve_over_setups
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        showlegend=True,
        yaxis_title=metric if yaxis_title is None else yaxis_title,
        xaxis_title="Matching Threshold" if xaxis_title is None else xaxis_title,
        font={"family": "Arial"},
        title=figure_title,
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="gray")
    fig.update_xaxes(range=[-0.05, 1.05])

    if manual_metric_range is not None:
        fig.update_yaxes(range=[manual_metric_range[0], manual_metric_range[1]])

    return fig


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
    if alternate_groupnames is not None and len(alternate_groupnames) != len(groups):
        raise ValueError(
            f"alternate_groupnames has length {len(alternate_groupnames)} but groups has length {len(groups)}; they must match."
        )

    if not plot_as_barchart and len(groups) != 1:
        raise ValueError(
            "When plotting without barcharts, you cannot plot more than one group at the same time"
        )
    #
    for setupname, stat in statistics_dict.items():
        if metric not in stat.metricnames:
            raise ValueError(f"metric {metric} not in statistic obj {setupname}")

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
            FloatDistribution(stat.get(g, metric, remove_nones=True)).avg
            for stat in statistics_dict.values()
        ]

        name = g if alternate_groupnames is None else alternate_groupnames[idx]

        if plot_std:
            Ystd = [
                FloatDistribution(stat.get(g, metric, remove_nones=True)).std
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
