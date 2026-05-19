from panoptica import Panoptica_Evaluator, Panoptica_Aggregator

ref_masks = "sample/BraTS-MET-00011-000-ref.nii.gz"
pred_masks = "sample/BraTS-MET-00011-000_pred.nii.gz"

evaluator = Panoptica_Evaluator.load_from_config("config_mets.yaml")

agg = Panoptica_Aggregator(evaluator, "output-BraTS-MET-00011.jsonl", continue_file=False, output_individual_instance_metrics=True)

result = agg.evaluate(pred_masks, ref_masks, "sample")
print(result)




# from typing import Callable, Optional
# from panoptica.utils.processing_pair import _ProcessingPairInstanced

# # The pipeline accepts ANY callable mapping a ProcessingPair to a ProcessingPair
# FilterType = Callable[[_ProcessingPairInstanced], _ProcessingPairInstanced]

# # Users can pass simple functions:
# def my_simple_filter(pair: _ProcessingPairInstanced) -> _ProcessingPairInstanced:
#     return pair

# # Or they can use your heavy-duty, stateful classes:
# class VolumeThresholdFilter:
#     def __init__(self, min_voxels: Optional[int], max_voxels: Optional[int]):
#         self.min = min_voxels
#         self.max = max_voxels

#     def __call__(self, pair: _ProcessingPairInstanced) -> _ProcessingPairInstanced:
#         # Complex logic here
#         return pair