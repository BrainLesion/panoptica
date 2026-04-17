import re

_INST_SEP = "-"
_INST_SUFFIX_BASE = "_inst_"

# Matches the pattern: <subject_name>-<group_name>_inst_<inst_idx>
# A subject name or group name containing '-' will cause to split ambiguously '-' is reserved in both subject and group names
_INSTANCE_PATTERN = re.compile(
    rf"^(.+){re.escape(_INST_SEP)}(.+){re.escape(_INST_SUFFIX_BASE)}(\d+)$"
)
########### Instance Key formatting and parsing


def format_instance_subject_name(
    subject_name: str, group_name: str, inst_idx: int
) -> str:
    """Formats a subject name for an individual instance row."""
    return f"{subject_name}{_INST_SEP}{group_name}{_INST_SUFFIX_BASE}{inst_idx}"


def parse_instance_subject_name(key: str) -> tuple[str, str, int] | None:
    """
    Parses an instance subject name.
    Returns (original_subject_name, group_name, instance_index) or None if not an instance row.
    """
    m = _INSTANCE_PATTERN.match(key)
    if m:
        try:
            return m.group(1), m.group(2), int(m.group(3))
        except ValueError:
            return None
    return None


########### Helper


def is_instance_row(key: str) -> bool:
    """
    Returns True if the key strictly matches the generated instance row format.
    """
    return _INSTANCE_PATTERN.match(key) is not None
