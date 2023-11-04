from enum import Enum, auto


class CCABackend(Enum):
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
