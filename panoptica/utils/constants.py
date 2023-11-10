from enum import Enum, auto
from typing_extensions import Self


class Enum_Compare(Enum):
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

    def __hash__(self) -> int:
        return self.value


# class Datatype_Status(Enum_Compare):
#    Semantic_Map = auto()
#    Unmatched_Instance_Map = auto()
#    Matched_Instance_Map = auto()


if __name__ == "__main__":
    #print(Datatype_Status.Semantic_Map)
