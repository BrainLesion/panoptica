"""Exception hierarchy for panoptica.

Every stream raises these — never a bare ``Exception``, never a silent
``except: pass``.
"""


class PanopticaError(Exception):
    """Base class for all panoptica errors."""


class MetricComputeError(PanopticaError):
    """A metric could not be computed (e.g. degenerate mask, edge case not handled)."""


class BackendUnavailable(PanopticaError):
    """A requested compute backend/device is not available (e.g. cuda without CuPy)."""


class InputValidationError(PanopticaError):
    """Input arrays failed a sanity check (shape/dtype/label-value)."""
