from enum import Enum, auto


class _Enum_Compare(Enum):
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Enum):
            return self.name == __value.name and self.value == __value.value
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return str(self)


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
