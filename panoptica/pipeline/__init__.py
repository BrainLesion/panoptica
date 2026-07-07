"""Pipeline layer."""

from panoptica.pipeline.approximate import approximate
from panoptica.pipeline.evaluate import evaluate
from panoptica.pipeline.match import match
from panoptica.pipeline.regionwise import evaluate_regionwise

__all__ = ["approximate", "match", "evaluate", "evaluate_regionwise"]
