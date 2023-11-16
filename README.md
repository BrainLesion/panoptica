[![PyPI version panoptica](https://badge.fury.io/py/panoptica.svg)](https://pypi.python.org/pypi/panoptica/)

# Panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

## Features

The package provides 3 core modules:

1. Instance Approximator: instance approximation algorithms in panoptic segmentation evaluation. Available now: connected components algorithm.
1. Instance Matcher: instance matching algorithm in panoptic segmentation evaluation, to align and compare predicted instances with reference instances.
1. Instance Evaluator: Evaluation of panoptic segmentation performance by evaluating matched instance pairs and calculating various metrics like true positives, Dice score, IoU, and ASSD for each instance.

![workflow_figure](https://github.com/BrainLesion/panoptica/blob/main/examples/figures/workflow.png)

## Installation

The current release requires python 3.10. To install it, you can simply run:

```sh
pip install panoptica
```

## Use Cases

All use cases have tutorials showcasing the usage that can be found at [BrainLesion/tutorials/panoptica](https://github.com/BrainLesion/tutorials/tree/main/panoptica).

### Semantic Segmentation Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/semantic.png?raw=true" alt="semantic_figure" height="300"/>

Although for many biomedical segmentation problems, an instance-wise evaluation is highly relevant and desirable, they are still addressed as semantic segmentation problems due to lack of appropriate instance labels.

Modules [1-3] can be used to obtain panoptic metrics of matched instances based on a semantic segmentation input.

[Jupyter Notebook Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_semantic.ipynb)

### Unmatched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/unmatched_instance.png?raw=true" alt="unmatched_instance_figure" height="300"/>

It is a common issue that instance segementation outputs have good segmentations with mismatched labels.

For this case modules [2-3] can be utilized to match the instances and report panoptic metrics.

[Jupyter Notebook Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_unmatched_instance.ipynb)

### Matched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/matched_instance.png?raw=true" alt="matched_instance_figure" height="300"/>

Ideally the input data already provides matched instances.

In this case module 3 can be used to directly report panoptic metrics without requiring any internal preprocessing.

[Jupyter Notebook Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_matched_instance.ipynb)

## Citation

If you have used panoptica in your research, please cite us!

The citation can be exported from: _TODO_
