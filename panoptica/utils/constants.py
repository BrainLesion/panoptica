from enum import Enum, auto
from panoptica.utils.config import (
    _register_class_to_yaml,
    _load_from_config,
    _load_from_config_name,
    _save_to_config,
)
from pathlib import Path
import numpy as np


class _Enum_Compare(Enum):
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Enum):
            namecheck = self.name == __value.name
            if not namecheck:
                return False
            if self.value is None:
                return __value.value is None

            try:
                if np.isnan(self.value):
                    return np.isnan(__value.value)
            except Exception:
                pass

            return self.value == __value.value
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __init_subclass__(cls, **kwargs):
        # Registers all subclasses of this
        super().__init_subclass__(**kwargs)
        cls._register_permanently()

    @classmethod
    def _register_permanently(cls):
        _register_class_to_yaml(cls)

    @classmethod
    def load_from_config(cls, path: str | Path):
        return _load_from_config(cls, path)

    @classmethod
    def load_from_config_name(cls, name: str):
        return _load_from_config_name(cls, name)

    def save_to_config(self, path: str | Path):
        _save_to_config(self, path)

    @classmethod
    def to_yaml(cls, representer, node):
        # cls._register_permanently()
        # assert hasattr(cls, "_yaml_repr"), f"Class {cls.__name__} has no _yaml_repr(cls, node) defined"
        return representer.represent_scalar("!" + cls.__name__, str(node.name))

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls[node.value]


class CCABackend(_Enum_Compare):
    """
    Enumeration representing different connected component analysis (CCA) backends.

    This enumeration defines options for CCA backends, which are used for labeling connected components in segmentation masks.

    Members:
        - cc3d: Represents the Connected Components in 3D (CC3D) backend for CCA.
          [CC3D Website](https://github.com/seung-lab/connected-components-3d)
        - scipy: Represents the SciPy backend for CCA.
          [SciPy Website](https://www.scipy.org/)
    """

    cc3d = auto()
    scipy = auto()


if __name__ == "__main__":
    print(CCABackend.cc3d == "cc3d")
    print("cc3d" == CCABackend.cc3d)
