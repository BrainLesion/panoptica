"""Developer-facing performance switches for :class:`Panoptica_Evaluator`.

The defaults reproduce the historical (pre-toggle) behavior of the pipeline exactly,
so ``PanopticaSpeedToggles()`` is a no-op. Flipping a toggle isolates the cost of a
single optimization, which is useful when benchmarking or debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoptica.utils.config import SupportsConfig


@dataclass(frozen=True)
class PanopticaSpeedToggles(SupportsConfig):
    """Configuration knobs for the biggest speed levers in the evaluator.

    Attributes:
        crop_at_start: Crop the input pair to its non-zero bounding box before running
            the pipeline (the current default). Disable to measure how much cropping
            saves.
        precompute_instance_bboxes: Run ``scipy.ndimage.find_objects`` once up front so
            each per-instance evaluation is done inside its own bounding box. Disabling
            falls back to full-array masks per instance.
        parallel_instance_eval: Evaluate matched instances in a process pool. Uses
            :class:`panoptica.utils.parallel_processing.NonDaemonicPool` so it is safe
            inside the aggregator's own worker processes.
        parallel_instance_eval_workers: Number of workers when parallel evaluation is
            enabled. ``None`` picks ``os.cpu_count()``.
    """

    crop_at_start: bool = True
    precompute_instance_bboxes: bool = True
    parallel_instance_eval: bool = False
    parallel_instance_eval_workers: int | None = None

    @classmethod
    def _yaml_repr(cls, node) -> dict:
        return {
            "crop_at_start": node.crop_at_start,
            "precompute_instance_bboxes": node.precompute_instance_bboxes,
            "parallel_instance_eval": node.parallel_instance_eval,
            "parallel_instance_eval_workers": node.parallel_instance_eval_workers,
        }
