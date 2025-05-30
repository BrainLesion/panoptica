# Metrics

A list of all metrics currently supported by Panoptica.
- DSC: Dice Similarity Coefficient
- IoU: Intersection over Union
- clDSC: Centerline Dice
- ASSD: Average Symmetric Surface Distance
- RVD: Relative Volume Difference
- RVAE: Relative Absolute Volume Error
- HD: Hausdorff Distance
- HD95: Hausdorff Distance 95 Percentile
- CEDI: Center Distance


For instance-wise metrics:
- True Positives (tp)
- False Positives (fp)
- False Negatives (fn)

And most importantly, the panoptic metrics:
- Recognition Quality (rq)
- Segmentation Quality (sq)
- Panoptic Quality (pq)

These three original come from [arXiv: Panoptic Segmentation](https://arxiv.org/abs/1801.00868).


---
### Missing a metric? Write us an [github: issue](https://github.com/BrainLesion/panoptica/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=New_Metric:)
Be sure to use the "metric" label.

<br/>
<br/>

# Metric Formulas

Formulas to calculate the different metrics

## Dice

$$
\text{DSC}(X,Y) = \frac{2 |X \cap Y|}{|X| + |Y|} \in [0,1]
$$
Represents an overlap score.

## Intersection over Union

$$
\text{IoU}(X,Y) = \frac{| X \cap Y | }{ |X \cup Y|} \in [0,1]
$$
Represents an overlap score. Is related to DSC, so giving both metrics doesn't really make sense (you can compute one with the other).

## Centerline Dice

Originally from [arXiv: clDice](https://arxiv.org/abs/2003.07311)

Represents an topology-preserving overlap score. Can be used as loss. Uses skeletonize and calculates the Dice coefficient on the skeletonized version of prediction and reference.

## Average Symmetric Surface Distance

$$
\text{ASSD}(X,Y) = \frac{\text{asd}(A,B) + \text{asd}(B,A)}{ |X| + |Y|} \in [0,\infty]
$$
with $\text{asd}(A, B)$ being the average surface distance from the border of $A$ to $B$:
$$
\text{asd}(A,B) = \sum_{a \in A}\min_{b \in B}\text{d}(a, b) \in [0,\infty]
$$
d(a,b) is the distance between both points.

ASSD is a typically good metric to report, as it shows whether errors are local or if the prediction has widespread noise voxels not even close to the reference.


## Relative (Voxel-)Volume Difference

$$
\text{RVD}(X,Y) = \frac{|X| - |Y|}{|Y|} \in [-\infty,\infty]
$$

The relative volume difference is the predicted volume of an instance in relation to the reference volume. For a journal, this might not be the most important metric. However, when the RVD is consistently positive, the predictions are oversegmenting. If negative, it is underpredicting (overall).

## Recognition Quality

$$
\text{RQ}(\text{tp},\text{fp}, \text{fn}) = \frac{\text{tp}}{\text{tp} + \frac{1}{2} \cdot (\text{fp} + \text{fn})} \in [0,1]
$$

It is the F1-score basically. Represents how well your instances match the references well (well = determined by threshold).

## Segmentation Quality

The segmentation quality is the average of all true positive metrics. So for sq_dsc, this is the average of all dice scores among the true positives in this prediction/reference pair.

As this metric is linked to another metric, we support all combinations of it. You can calculate the segmentation quality with IoU, DSC, clDSC, ASSD, and even RVD. 

For a metric $f$, we derive the Segmentation Quality (SQ):

$$
\text{SQ}_f(\text{TP}) = \sum_{(i_{\text{ref}}, i_{\text{pred}}) \in TP} \frac{f(i_{\text{ref}}, i_{\text{pred}})}{|TP|}
$$

## Panoptic Quality

$$
\text{PQ}_f(X,Y) = \text{SQ}_f \cdot \text{RQ} \in [0,1]
$$

Combines the F1-score of instances with the Segmentation Quality.
