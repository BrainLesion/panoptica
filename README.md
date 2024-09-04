# panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

## Features

The package provides three core modules:

1. Instance Approximator: instance approximation algorithms to extract instances from semantic segmentation maps/model outputs.
2. Instance Matcher: matches predicted instances with reference instances.
3. Instance Evaluator: computes segmentation and detection quality metrics for pairs of predicted - and reference segmentation maps.

<b>blinded figure</b>

## Installation

With a Python 3.10+ environment, you can install panoptica from <b>blinded pypi link</b>

```sh
pip install panoptica
```

## Use cases and tutorials

For tutorials featuring various use cases, see: <b>blinded link</b>

### Metrics

Panoptica supports a large range of metrics. An overview of the supported metrics and their formulas can be found: <b>blinded link</b>

### Semantic Segmentation Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>


Although an instance-wise evaluation is highly relevant and desirable for many biomedical segmentation problems, they are still addressed as semantic segmentation problems due to the lack of appropriate instance labels.

This tutorial leverages all three modules of panoptica: instance approximation, -matching and -evaluation.

### Unmatched Instances Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>

It is a common issue that instance segmentation outputs feature good outlines but mismatched instance labels.
For this case, the matcher module can be utilized to match instances and the evaluator to report metrics.


### Matched Instances Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>

If your predicted instances already match the reference instances, you can directly compute metrics using the evaluator module.


### Using Configs (saving and loading)

You can construct Panoptica_Evaluator (among many others) objects and save their arguments, so you can save project-specific configurations and use them later.

<b>blinded link</b>

It uses ruamel.yaml in a readable way.


## Citation

If you use panoptica in your research, please cite it to support the development!

<b>Blinded Citation</b>