import re

_INST_SEP = "-"
_INST_SUFFIX_BASE = "_inst_"

# Matches the pattern: <subject_name>-<group_name>_inst_<inst_idx>
# Greedy `.+` groups split on the last '-'. This is only unambiguous if group names
# contain no '-' — enforced by validate_group_name below.
_INSTANCE_PATTERN = re.compile(
    rf"^(.+){re.escape(_INST_SEP)}(.+){re.escape(_INST_SUFFIX_BASE)}(\d+)$"
)

# Matches any string ending in the reserved instance suffix shape. A subject name
# matching this would be misclassified by is_instance_row even when no instance was
# ever formatted from it.
_SUBJECT_COLLISION_PATTERN = re.compile(
    rf".+{re.escape(_INST_SEP)}.+{re.escape(_INST_SUFFIX_BASE)}\d+$"
)

########### Validation


def validate_subject_name(name: str) -> None:
    """Raises ValueError if `name` would be misclassified as an instance row.

    Subject names are user-provided and may legitimately contain '-'. The only
    collision shape we need to reject is names that would match the instance-row
    regex on their own (e.g. 'patient-001_inst_5').
    """
    if _SUBJECT_COLLISION_PATTERN.match(name):
        raise ValueError(
            f"Subject name {name!r} collides with the reserved instance-row "
            f"suffix '<...>{_INST_SEP}<...>{_INST_SUFFIX_BASE}<int>'. "
            f"Rename the subject to avoid this suffix."
        )


def validate_group_name(name: str) -> None:
    """Raises ValueError if `name` contains a reserved structural token.

    '-' is the delimiter between subject and group in instance rows (and between
    group and metric in TSV headers); '_inst_' is the instance-index marker.
    Allowing either in a group name makes instance-row parsing ambiguous.
    """
    if _INST_SEP in name:
        raise ValueError(
            f"Group name {name!r} contains reserved delimiter {_INST_SEP!r}. "
            f"Group names must not contain '-'."
        )
    if _INST_SUFFIX_BASE in name:
        raise ValueError(
            f"Group name {name!r} contains reserved token {_INST_SUFFIX_BASE!r}. "
            f"Group names must not contain '_inst_'."
        )


########### Instance Key formatting and parsing


def format_instance_subject_name(
    subject_name: str, group_name: str, inst_idx: int
) -> str:
    """Formats a subject name for an individual instance row."""
    validate_subject_name(subject_name)
    validate_group_name(group_name)
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
