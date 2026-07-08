"""API layer — public Evaluator + result container."""

from panoptica.api.config import load_config, save_config
from panoptica.api.evaluator import Evaluator
from panoptica.api.result import EvalResult

__all__ = ["Evaluator", "EvalResult", "load_config", "save_config"]
