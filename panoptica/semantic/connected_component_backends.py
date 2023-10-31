from enum import Enum, auto


class CCABackend(Enum):
    cc3d = auto()
    scipy = auto()
