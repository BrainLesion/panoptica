[![PyPI version panoptica](https://badge.fury.io/py/panoptica.svg)](https://pypi.python.org/pypi/panoptica/)

# panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

## Features

The package provides three core modules:

1. Instance Approximator: instance approximation algorithms to extract instances from semantic segmentation maps/model outputs.
2. Instance Matcher: matches predicted instances with reference instances.
3. Instance Evaluator: computes segmentation and detection quality metrics for pairs of predicted - and reference segmentation maps.

![workflow_figure](https://github.com/BrainLesion/panoptica/blob/main/examples/figures/workflow.png?raw=true)

## Installation

With a Python 3.10+ environment, you can install panoptica from [pypi.org](https://pypi.org/project/panoptica/):

```sh
pip install panoptica
```

## Use cases and tutorials

For tutorials featuring various use cases, cf. [BrainLesion/tutorials/panoptica](https://github.com/BrainLesion/tutorials/tree/main/panoptica).

### Semantic Segmentation Input
Although an instance-wise evaluation is highly relevant and desirable for many biomedical segmentation problems, they are still addressed as semantic segmentation problems due to the lack of appropriate instance labels.

[Jupyter Notebook Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_semantic.ipynb)

This tutorial leverages all three modules.

### Unmatched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/unmatched_instance.png?raw=true" alt="unmatched_instance_figure" height="300"/>

It is a common issue that instance segmentation outputs feature good outlines but mismatched instance labels.
For this case, modules 2 and 3 can be utilized to match the instances and report metrics.

[Jupyter Notebook Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_unmatched_instance.ipynb)

### Matched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/matched_instance.png?raw=true" alt="matched_instance_figure" height="300"/>

If your predicted instances already match the reference instances, you can directly compute metrics with the third module, see [Jupyter Notebook](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_matched_instance.ipynb) Example](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_matched_instance.ipynb)

## Citation

If you use panoptica in your research, please cite it to support the development!

TBA

```
upcoming citation
```
