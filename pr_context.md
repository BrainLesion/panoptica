# PR to integrate part-aware metrics into Panoptica

## Primary design struggle.

Inherently, Panoptica does everything classwise. In order to make the "Part Setting Work", the particular "segmentation class group" [ref:] https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/utils/segmentation_class.py#L9 has have information of both the thing and the part class. Which is kinda against the design scheme.

This was challenging because we have to maintain a strucutre which works well for the matching (this needs the information about both the classes), and the final "Class" scores (needs single class information). **This was why I chose to pass the Labelgroup information everywhere.** 

As of now, the matching, TP, FP and FN values work as intended. But the global scores for this `PartLabelGroup` class (discussed below) is sending in multilabel values but https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/panoptica_result.py#L19 needs binary. 

## ***Primary Discussion Point***: One of the following 3 can happen: 

1. Do we adopt all metrics to multilabel just for this? 
        - This should not take too long to do, even with setting up tests. It may also set up a nice scheme to add in-house tests and comparison for "non matching based metrics". I am aware panoptica already does that inherently.
2. Do we build a seperate evaluator and result class just for Part cases?
3. or, is my entire structure faulty right now?

Note: Although not present in brats, *I am yet to test for multiple parts of different classes within a thing*. The code should work for it as is but I have not tested for it.

# Discussing the current state of the potential PR

Link to repo: https://github.com/aymuos15/panoptica. It will based off of this. Currently, I have kept all print statements uncommented so its easy to immediately gauge whats happening. Find the a visual representation of stuff in `context.ipynb`. Most of the viz comes from `_calc_matching_metric_of_overlapping_partlabels` in `panoptica/_functionals.py`

## TLDR

Users can now call `class LabelPartGroup(LabelGroup)` to make a class part-aware.

Abstracting things, the main change is as follows:

Currently, this is how the BraTS config is set up: [Ref: https://github.com/BrainLesion/panoptica/issues/195#issuecomment-2826557028]

```python
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        segmentation_class_groups=SegmentationClassGroups(
            {
                "ET": (3, False),
                "NET": (1, False),
                "ED": (2, False),
                "TC": LabelMergeGroup([1, 3], False),
                "WT": LabelMergeGroup([1, 2, 3], False),
            }
        ),
    )
```

We propose to **allow users the option** to switch out `class LabelMergeGroup(LabelGroup)`

https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/utils/label_group.py#L111 

with `class LabelPartGroup(LabelGroup)` (proposed). This would now look like:

```python
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(),
        segmentation_class_groups=SegmentationClassGroups(
            {
                "ET": (3, False),
                "NET": (1, False),
                "ED": (2, False),
                "TC": LabelPartGroup([1, 3], False),
                "WT": LabelPartGroup([1, 2, 3], False),
            }
        ),
    )
```

Visually for a case: 

![alt text](image-2.png)

`LabelMergaGroup` looks like this:

![alt text](image-1.png)

`LabelPartGroup` looks like this:

![alt text](image.png)

Notice -> binary vs multi label.

##  Some additional details.

For now, I have made the following assumptions (in terms of code) for `LabelPartGroup`:

1. The first index of the Group has to be a thing.
2. There will be only 1 thing for Group.
3. The remaining indexes will be for parts.

The code is scalable to any number of these groups, and any number of parts in each group. 

I have not checked for optimisation yet. Happy to discuss in this thread, change relevant parts for optimisation and then do the final trigger.

Like PartPQ, I have not considered the "part in part" situation. That is out of scope for this PR.

# Overarching Changes.

### 1. `class LabelPartGroup(LabelGroup)` in `panoptica/utils/label_group.py`

Through out the code base, I have passed `LabeGroup` https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/utils/label_group.py#L7 as args in https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/panoptica_evaluator.py#L28. This is was the simplest way I found to minimize change across the main functionality of the codebase. 

Will not go into detail. But Example:

```python
        result = panoptic_evaluate(
            input_pair=processing_pair_grouped,
            edge_case_handler=self.__edge_case_handler,
            instance_approximator=self.__instance_approximator,
            instance_matcher=self.__instance_matcher,
            instance_metrics=self.__eval_metrics,
            global_metrics=self.__global_metrics,
            decision_metric=self.__decision_metric,
            decision_threshold=decision_threshold,
            result_all=result_all,
            log_times=self.__log_times if log_times is None else log_times,
            verbose=True if verbose is None else verbose,
            verbose_calc=self.__verbose if verbose is None else verbose,
            label_group=label_group,  # <-- pass label_group
        )
```

### 2. Instance approxiamation. 

The Instance approxiamation step had to be based on an one hot encoded vector rather than the normal/deafult one currently:

(https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/instance_approximator.py#L109)

* This is primiarly because I was not able to generate part matching based on the normal one (even though choosing cc3d off the shelf gives multi-label components.)

Note: this can be achieved this in a hacky way when only 1 class of parts is present within a thing. It becomes unsolvable when there are two classes of parts in a thing. As mentioned, this is not a problem for Brats, so can revert if required.

### 3. The main part matching logic

The main part matching logic can be found in `_calc_matching_metric_of_overlapping_partlabels` and its other required funcs within `panoptica/_functionals.py`. I have made sure the output is exactly like `_calc_matching_metric_of_overlapping_labels` https://github.com/BrainLesion/panoptica/blob/6dd4ab2b6cd0794491ecb4c84265f3295317346f/panoptica/_functionals.py#L13 . The primary logic is that, the part class values (based on the matching of the overarching thing class in the same LabelPartGroup) are added to each other.

### 4. Change to the current Matchers

In order to allow for part matching in an existing matching algo, the change has to be impemented in the class itself. Ex:

```python
class NaiveThresholdMatching(InstanceMatchingAlgorithm):
    """
    Instance matching algorithm that performs one-to-one matching based on IoU values.

    Attributes:
        iou_threshold (float): The IoU threshold for matching instances.

    Methods:
        __init__(self, iou_threshold: float = 0.5) -> None:
            Initialize the NaiveOneToOneMatching instance.
        _match_instances(self, unmatched_instance_pair: UnmatchedInstancePair, **kwargs) -> Instance_Label_Map:
            Perform one-to-one instance matching based on IoU values.

    Raises:
        AssertionError: If the specified IoU threshold is not within the valid range.

    Example:
    >>> matcher = NaiveOneToOneMatching(iou_threshold=0.6)
    >>> unmatched_instance_pair = UnmatchedInstancePair(...)
    >>> result = matcher.match_instances(unmatched_instance_pair)
    """

    def __init__(
        self,
        matching_metric: Metric = Metric.IOU,
        matching_threshold: float = 0.5,
        allow_many_to_one: bool = False,
    ) -> None:
        """
        Initialize the NaiveOneToOneMatching instance.

        Args:
            iou_threshold (float, optional): The IoU threshold for matching instances. Defaults to 0.5.

        Raises:
            AssertionError: If the specified IoU threshold is not within the valid range.
        """
        self._allow_many_to_one = allow_many_to_one
        self._matching_metric = matching_metric
        self._matching_threshold = matching_threshold

    def _match_instances(
        self,
        unmatched_instance_pair: UnmatchedInstancePair,
        label_group=None,  # <-- add label_group argument
        **kwargs,
    ) -> InstanceLabelMap:
        """
        Perform one-to-one instance matching based on IoU values.

        Args:
            unmatched_instance_pair (UnmatchedInstancePair): The unmatched instance pair to be matched.
            label_group: Optional label group information.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance_Label_Map: The result of the instance matching.
        """
        ref_labels = unmatched_instance_pair.ref_labels

        # Initialize variables for True Positives (tp) and False Positives (fp)
        labelmap = InstanceLabelMap()

        pred_arr, ref_arr = (
            unmatched_instance_pair.prediction_arr,
            unmatched_instance_pair.reference_arr,
        )

        # Calculate matching metric pairs based on whether it's a part group
        is_part_group = label_group is not None and hasattr(label_group, "part_labels")

        # Calculate matching metric pairs based on whether it's a part group
        is_part_group = label_group is not None and hasattr(label_group, "part_labels")
        if is_part_group:
            mm_pairs = _calc_matching_metric_of_overlapping_partlabels(
                pred_arr,
                ref_arr,
                ref_labels,
                matching_metric=self._matching_metric,
            )
        else:
            mm_pairs = _calc_matching_metric_of_overlapping_labels(
                pred_arr, ref_arr, ref_labels, matching_metric=self._matching_metric
            )

        # Loop through matched instances to compute PQ components
        for matching_score, (ref_label, pred_label) in mm_pairs:
            if (
                labelmap.contains_or(pred_label, ref_label)
                and not self._allow_many_to_one
            ):
                continue  # -> doesnt make speed difference
            # TODO always go in here, but add the matching score to the pair (so evaluation over multiple thresholds becomes easy)
            if self._matching_metric.score_beats_threshold(
                matching_score, self._matching_threshold
            ):
                # Match found, increment true positive count and collect IoU and Dice values
                labelmap.add_labelmap_entry(pred_label, ref_label)
                # map label ref_idx to pred_idx

        return labelmap

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "matching_metric": node._matching_metric,
            "matching_threshold": node._matching_threshold,
            "allow_many_to_one": node._allow_many_to_one,
        }
```

Again, Definitely up for discussion about a better way to do this.