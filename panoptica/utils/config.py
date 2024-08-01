from ruamel.yaml import YAML
from pathlib import Path

####################


def load_yaml(file: str | Path):
    if isinstance(file, str):
        file = Path(file)
        yaml = YAML(typ="safe")
        data = yaml.load(file)
        assert isinstance(data, dict) or isinstance(data, object)
    return data


def save_yaml(data_dict: dict | object, out_file: str | Path, registered_class=None):
    if isinstance(out_file, str):
        out_file = Path(out_file)

        yaml = YAML(typ="safe")
        if registered_class is not None:
            yaml.register_class(registered_class)
            if isinstance(data_dict, object):
                yaml.dump(data_dict, out_file)
            else:
                yaml.dump([registered_class(*data_dict)], out_file)
        else:
            yaml.dump(data_dict, out_file)


class Person:
    name: str
    age: int

    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age


####################

# TODO split into general config and object configuration (latter saves as object yaml and loads directly as object?)


class Configuration:
    _data_dict: dict
    _registered_class = None

    def __init__(self, data_dict: dict, registered_class=None) -> None:
        self._data_dict = data_dict
        if registered_class is not None:
            self.register_to_class(registered_class)

    def register_to_class(self, cls):
        self._registered_class = cls

    @classmethod
    def save_from_object(cls, obj: object, file: str | Path):
        save_yaml(obj, file, registered_class=type(obj))
        return Configuration.load(file, registered_class=type(obj))

    @classmethod
    def load(cls, file: str | Path, registered_class=None):
        data = load_yaml(file)
        return Configuration(data, registered_class=registered_class)

    def save(self, out_file: str | Path):
        save_yaml(self._data_dict, out_file)

    def cls_object_from_this(self):
        assert self._registered_class is not None
        self._registered_class(*self._data_dict)


if __name__ == "__main__":
    c = Configuration.load("/DATA/NAS/ongoing_projects/hendrik/panoptica/repo/panoptica/base_configs/test.yaml")

    c.save("/DATA/NAS/ongoing_projects/hendrik/panoptica/repo/panoptica/base_configs/test_out.yaml")

    arya = Person("arya", 18)

    c = Configuration.save_from_object(arya, "/DATA/NAS/ongoing_projects/hendrik/panoptica/repo/panoptica/base_configs/arya.yaml")
    print(c._data_dict.name)
