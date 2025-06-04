# Extending Panoptica

Here are a couple of guides on how to implement new modules into panoptica

## Metrics


In order to implement a new metric to panoptica, you have to perform the following steps:

1. under panoptica/metrics, create a new python file for your metrics, add an implementation similar to the other metrics (e.g. panoptica/metrics/dice.py)
2. In panoptica/metrics.metrics.py: Import your new metric function, add a new entry to the Metric Enum
```
class Metric(_Enum_Compare):

    METRICID = _Metric("METRICID", "Fully descriptive name for this metric", False, _function_that_calculates_metric)
```

3. In panoptica/panoptica_result.py: Depending on whether the metric can be reported as segmentation quality (average over only TP), add a new region by naming it and then adding it with the self._add_metric() call. Additionally, if it can be globally reported, add the corresponding self.global_bin_metric entry.
4. In panoptica/utils.edge_case_handling.py: Add default edge case handling for your new metric.


## Input Data Type

To make panoptica support a new input data type, perform the following steps:

1. Under /utils/input_check_and_conversion/, make a new file similar to "check_torch_image.py"
2. In there, you have to import the specialized package using the find_spec function. You need a function that loads the file if it does support any file endings, and one function that proceeds as follows:
    A. Load the data if necessary (str | Path)
    B. Asserts the data is the correct data type
    C. Proceeds with Sanity checks, that either return True and the tuple of converted numpy arrays, or False and a error message string (see check_sitk_image.py)
3. To actually make it work, add your new function with the corresponding specifications in the sanity_checker.py (example below):
```
class INPUTDTYPE(_Enum_Compare):
    NEW = _InputDataTypeChecker(
        supported_file_endings=[
            ".fileending",
            ".fileending2",
        ],
        required_package_names=["the-package-name"],
        sanity_check_handler=your_new_function_from_the_second_bulletpoint,
    )
```
4. In panoptica/panoptica_evaluator.py, import the newly supported package
```
if TYPE_CHECKING:
    import torch
    import SimpleITK as sitk
    import nibabel as nib
```
here and then add it to the type hint in the evaluate function, see below:
```
def evaluate(
        self,
        prediction_arr: Union[np.ndarray, "torch.Tensor", "nib.nifti1.Nifti1Image", "sitk.Image"],
        reference_arr: Union[np.ndarray, "torch.Tensor", "nib.nifti1.Nifti1Image", "sitk.Image"],
        result_all: bool = True,
        save_group_times: bool | None = None,
        log_times: bool | None = None,
        verbose: bool | None = None,
    ) -> dict[str, PanopticaResult]:
```

## Instance Matching Algorithm

To add a new Instance matching algorithm, simply head to panoptica/instance_matcher.py, copy one of the existing algorithms and make your corresponding changes.