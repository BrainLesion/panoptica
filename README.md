[![PyPI version panoptica](https://badge.fury.io/py/panoptica.svg)](https://pypi.python.org/pypi/panoptica/)

# Panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

## Features

The package provides 3 core modules:

1. Instance Approximator
1. Instance Matcher
1. Panoptic Evaluator

<!-- ?TODO: add module figure? -->

## Installation

To install the current release, you can simply run:

```sh
pip install panoptica
```

## Use Cases

### Semantic Segmentation Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/spine_seg/semantic/fig.png?raw=true" alt="semantic_figure" height="300"/>

Although for many biomedical segmentation problems, an instance-wise evaluation is highly relevant and desirable, they are still addressed as semantic segmentation problems due to lack of appropriate instance labels.

Modules [1-3] can be used to obtain panoptic metrics of matched instances based on a semantic segmentation input.

### Unmatched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/spine_seg/unmatched_instance/fig.png?raw=true" alt="unmatched_instance_figure" height="300"/>

It is a common issue that instance segementation outputs have good segmentations with mismatched labels.

For this case modules [2-3] can be utilized to match the instances and report panoptic metrics.

### Matched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/spine_seg/matched_instance/fig.png?raw=true" alt="matched_instance_figure" height="300"/>

Ideally the input data already provides matched instances.

In this case module 3 can be used to directly report panoptic metrics without requiring any internal preprocessing.

## Tutorials

Juypter notebook Tutorials are avalable for all use cases in our [tutorials repo](https://github.com/BrainLesion/tutorials/tree/main/panoptica).

## Citation

If you have used panoptica in your research, please cite us! The citation can be exported from: TODO
