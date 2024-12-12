


# panoptica

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

 
[Use Cases & Tutorials](#use-cases--tutorials) | [Documentation](#documentation)

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

## Available Metrics

> [!NOTE]
> Panoptica supports a large range of metrics. <br>
> An overview of the supported metrics and their formulas can be found here: <b>blinded link</b>

## Use Cases & Tutorials


### Minimal example

A minimal example of using panoptica could look e.g. like this (here with Matched Instances as Input):
```python
from panoptica import InputType, Panoptica_Evaluator
from panoptica.metrics import Metric

from auxiliary.nifti.io import read_nifti # feel free to use any other way to read nifti files

ref_masks = read_nifti("reference.nii.gz")
pred_masks = read_nifti("prediction.nii.gz")

evaluator = Panoptica_Evaluator(
    expected_input=InputType.MATCHED_INSTANCE,
    decision_metric=Metric.IOU,
    decision_threshold=0.5,
)

result, intermediate_steps_data = evaluator.evaluate(pred_masks, ref_masks)["ungrouped"]
```


> [!TIP]
> We provide Jupyter Notebook tutorials showcasing various use cases. <br>
> You can explore them here: <b>blinded link</b><br>

### Semantic Segmentation Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>


Although an instance-wise evaluation is highly relevant and desirable for many biomedical segmentation problems, they are still addressed as semantic segmentation problems due to the lack of appropriate instance labels.

This tutorial leverages all three modules of panoptica: instance approximation, -matching and -evaluation.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_semantic.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Unmatched Instances Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>

It is a common issue that instance segmentation outputs feature good outlines but mismatched instance labels.
For this case, the matcher module can be utilized to match instances and the evaluator to report metrics.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_unmatched_instance.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_unmatched_instance.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



### Matched Instances Input

<b>blinded figure</b>

<b>blinded link to jupyter notebook tutorial</b>

If your predicted instances already match the reference instances, you can directly compute metrics using the evaluator module.

### Matching Algorithm Example
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matching_algorithm.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matching_algorithm.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Using Configs (saving and loading)

You can construct Panoptica_Evaluator (among many others) objects and save their arguments, so you can save project-specific configurations and use them later.

<b>blinded link</b>

It uses ruamel.yaml in a readable way.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_config.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_config.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Documentation

We provide a readthedocs documentation of our codebase [here](https://panoptica.readthedocs.io/en/latest/?badge=latest)

## Citation

> [!IMPORTANT]
> If you use panoptica in your research, please cite it to support the development!

<b>Blinded Citation</b>