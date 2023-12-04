[![PyPI version panoptica](https://badge.fury.io/py/panoptica.svg)](https://pypi.python.org/pypi/panoptica/)
[![Documentation Status](https://readthedocs.org/projects/panoptica/badge/?version=latest)](http://panoptica.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml)


# panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

## Features

The package provides three core modules:

1. Instance Approximator: instance approximation algorithms to extract instances from semantic segmentation maps/model outputs.
2. Instance Matcher: matches predicted instances with reference instances.
3. Instance Evaluator: computes segmentation and detection quality metrics for pairs of predicted - and reference segmentation maps.

![workflow_figure](https://github.com/BrainLesion/panoptica/blob/main/examples/figures/workflow.png?raw=true)

## Installation

With a Python 3.10+ environment, you can install panoptica from [pypi.org](https://pypi.org/project/panoptica/)

```sh
pip install panoptica
```

## Use cases and tutorials

For tutorials featuring various use cases, see: [BrainLesion/tutorials/panoptica](https://github.com/BrainLesion/tutorials/tree/main/panoptica)

### Semantic Segmentation Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/semantic.png?raw=true" alt="semantic_figure" height="300"/>

[Jupyter notebook tutorial](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_semantic.ipynb)


Although an instance-wise evaluation is highly relevant and desirable for many biomedical segmentation problems, they are still addressed as semantic segmentation problems due to the lack of appropriate instance labels.

This tutorial leverages all three modules of panoptica: instance approximation, -matching and -evaluation.

### Unmatched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/unmatched_instance.png?raw=true" alt="unmatched_instance_figure" height="300"/>

[Jupyter notebook tutorial](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_unmatched_instance.ipynb)

It is a common issue that instance segmentation outputs feature good outlines but mismatched instance labels.
For this case, the matcher module can be utilized to match instances and the evaluator to report metrics.


### Matched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/matched_instance.png?raw=true" alt="matched_instance_figure" height="300"/>

[Jupyter notebook tutorial](https://github.com/BrainLesion/tutorials/tree/main/panoptica/example_spine_matched_instance.ipynb) 

If your predicted instances already match the reference instances, you can directly compute metrics using the evaluator module.

## Citation

If you use panoptica in your research, please cite it to support the development!

TBA

```
upcoming citation
```
