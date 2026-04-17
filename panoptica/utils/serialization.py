import re

_THRESHOLD_PREFIX = "t"
_THRESHOLD_SEP = "_"
_AUTC_PREFIX = "autc_"

########### AUTC Key formatting and parsing

def format_threshold_key(threshold: float, metric: str) -> str:
    return f"{_THRESHOLD_PREFIX}{threshold:g}{_THRESHOLD_SEP}{metric}"

def format_autc_key(metric: str) -> str:
    return f"{_AUTC_PREFIX}{metric}"

# Allow e, E, +, and - in the capture group to support scientific notation
_THRESHOLD_PATTERN = re.compile(r"^t([0-9\.eE+-]+)_(.+)$")

def parse_threshold_key(key: str) -> tuple[float, str] | None:
    """Returns (threshold, base_metric) or None if not a threshold key."""
    m = _THRESHOLD_PATTERN.match(key)
    if m:
        return float(m.group(1)), m.group(2)
    return None

def parse_autc_key(key: str) -> str | None:
    """Returns base_metric or None if not an AUTC key."""
    if key.startswith(_AUTC_PREFIX):
        return key[len(_AUTC_PREFIX):]
    return None

########### Helper

def is_threshold_key(key: str) -> bool:
    return parse_threshold_key(key) is not None

def is_autc_key(key: str) -> bool:
    return parse_autc_key(key) is not None
