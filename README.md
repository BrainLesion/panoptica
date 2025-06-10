

# panoptica

[![Python Versions](https://img.shields.io/pypi/pyversions/panoptica)](https://pypi.org/project/panoptica/)
[![Stable Version](https://img.shields.io/pypi/v/panoptica?label=stable)](https://pypi.python.org/pypi/panoptica/)
[![Documentation Status](https://readthedocs.org/projects/panoptica/badge/?version=latest)](http://panoptica.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/panoptica/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/panoptica/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/panoptica)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

 
[Use Cases & Tutorials](#use-cases--tutorials) | [Documentation](#documentation)

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

Or, alternatively, from [conda](https://anaconda.org/conda-forge/panoptica):

```sh
conda install -c conda-forge panoptica
```

## Available Metrics

> [!NOTE]
> Panoptica supports a large range of metrics. <br>
> An overview of the supported metrics and their formulas can be found here: [panoptica/metrics.md](https://github.com/BrainLesion/panoptica/tree/main/metrics.md)

## Use Cases & Tutorials


### Minimal example

A minimal example of using panoptica could look e.g. like this (here with Matched Instances as Input):
```python
from panoptica import InputType, Panoptica_Evaluator
from panoptica.metrics import Metric

from auxiliary.io import read_image # feel free to use any other way to read nifti files

ref_masks = read_image("reference.nii.gz")
pred_masks = read_image("prediction.nii.gz")

evaluator = Panoptica_Evaluator(
    expected_input=InputType.MATCHED_INSTANCE,
    decision_metric=Metric.IOU,
    decision_threshold=0.5,
)

result = evaluator.evaluate(pred_masks, ref_masks)["ungrouped"]
```


> [!TIP]
> We provide Jupyter Notebook tutorials showcasing various use cases. <br>
> You can explore them here: [BrainLesion/tutorials/panoptica](https://github.com/BrainLesion/tutorials/tree/main/panoptica) <br>

### Semantic Segmentation Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/semantic.png?raw=true" alt="semantic_figure" height="300"/>

Although an instance-wise evaluation is highly relevant and desirable for many biomedical segmentation problems, they are still addressed as semantic segmentation problems due to the lack of appropriate instance labels.

This tutorial leverages all three modules of panoptica: instance approximation, -matching and -evaluation.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_semantic.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Unmatched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/unmatched_instance.png?raw=true" alt="unmatched_instance_figure" height="300"/>

It is a common issue that instance segmentation outputs feature good outlines but mismatched instance labels.
For this case, the matcher module can be utilized to match instances and the evaluator to report metrics.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_unmatched_instance.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_unmatched_instance.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



### Matched Instances Input

<img src="https://github.com/BrainLesion/panoptica/blob/main/examples/figures/matched_instance.png?raw=true" alt="matched_instance_figure" height="300"/>

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matched_instance.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matched_instance.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

If your predicted instances already match the reference instances, you can directly compute metrics using the evaluator module.

### Matching Algorithm Example
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matching_algorithm.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_spine_matching_algorithm.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Using Configs (saving and loading)

You can construct Panoptica_Evaluator (among many others) objects and save their arguments, so you can save project-specific configurations and use them later.
It uses ruamel.yaml in a readable way.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/BrainLesion/tutorials/blob/main/panoptica/example_config.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/panoptica/example_config.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Using Panoptica Aggregator

Using our Panoptica Aggregator, you can evaluate your data while the Aggregator takes care of writing the results into a .tsv file (threading-safe). This allows you to focus only on the important part, that is loading and iterating over your data.
From the aggregator, you can immediately get a Panoptica Statistics object that yield you the statistics over your metrics as well as create nice figures for you.

Tutorial: TBD

## Documentation

We provide a readthedocs documentation of our codebase [here](https://panoptica.readthedocs.io/en/latest/?badge=latest)

## Citation

> [!IMPORTANT]
> If you use panoptica in your research, please cite it to support the development!

Kofler, F., Möller, H., Buchner, J. A., de la Rosa, E., Ezhov, I., Rosier, M., Mekki, I., Shit, S., Negwer, M., Al-Maskari, R., Ertürk, A., Vinayahalingam, S., Isensee, F., Pati, S., Rueckert, D., Kirschke, J. S., Ehrlich, S. K., Reinke, A., Menze, B., Wiestler, B., & Piraud, M. (2023). *Panoptica -- instance-wise evaluation of 3D semantic and instance segmentation maps.* [arXiv preprint arXiv:2312.02608](https://arxiv.org/abs/2312.02608).

```
@misc{kofler2023panoptica,
      title={Panoptica -- instance-wise evaluation of 3D semantic and instance segmentation maps}, 
      author={Florian Kofler and Hendrik Möller and Josef A. Buchner and Ezequiel de la Rosa and Ivan Ezhov and Marcel Rosier and Isra Mekki and Suprosanna Shit and Moritz Negwer and Rami Al-Maskari and Ali Ertürk and Shankeeth Vinayahalingam and Fabian Isensee and Sarthak Pati and Daniel Rueckert and Jan S. Kirschke and Stefan K. Ehrlich and Annika Reinke and Bjoern Menze and Benedikt Wiestler and Marie Piraud},
      year={2023},
      eprint={2312.02608},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing

We welcome all kinds of contributions from the community!

Some guides on extending panoptica can be found in [panoptica/extending_panoptica.md](https://github.com/BrainLesion/panoptica/tree/main/extending_panoptica.md)

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/panoptica/issues).
