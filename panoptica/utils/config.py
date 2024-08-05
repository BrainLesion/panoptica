from ruamel.yaml import YAML
from pathlib import Path
from panoptica.utils.filepath import config_by_name, config_dir_by_name
from abc import ABC, abstractmethod

supported_helper_classes = []


def _register_helper_classes(yaml: YAML):
    [yaml.register_class(s) for s in supported_helper_classes]


def _load_yaml(file: str | Path, registered_class=None):
    if isinstance(file, str):
        file = Path(file)
    yaml = YAML(typ="safe")
    _register_helper_classes(yaml)
    if registered_class is not None:
        yaml.register_class(registered_class)
    yaml.default_flow_style = None
    data = yaml.load(file)
    assert isinstance(data, dict) or isinstance(data, object)
    return data


def _save_yaml(data_dict: dict | object, out_file: str | Path, registered_class=None):
    if isinstance(out_file, str):
        out_file = Path(out_file)

    yaml = YAML(typ="safe")
    yaml.default_flow_style = None
    yaml.representer.ignore_aliases = lambda *data: True
    _register_helper_classes(yaml)
    if registered_class is not None:
        yaml.register_class(registered_class)
        assert isinstance(data_dict, registered_class)
        # if isinstance(data_dict, object):
        yaml.dump(data_dict, out_file)
        # else:
        #    yaml.dump([registered_class(*data_dict)], out_file)
    else:
        yaml.dump(data_dict, out_file)
    print(f"Saved config into {out_file}")


####################
# TODO Merge into SupportsConfig
class Configuration:
    """General Configuration class that handles yaml"""

    _data_dict: dict
    _registered_class = None

    def __init__(self, data_dict: dict, registered_class=None) -> None:
        assert isinstance(data_dict, dict)
        self._data_dict = data_dict
        if registered_class is not None:
            self.register_to_class(registered_class)

    def register_to_class(self, cls):
        global supported_helper_classes
        if cls not in supported_helper_classes:
            supported_helper_classes.append(cls)
        self._registered_class = cls
        return self

    @classmethod
    def save_from_object(cls, obj: object, file: str | Path):
        _save_yaml(obj, file, registered_class=type(obj))
        # return Configuration.load(file, registered_class=type(obj))

    @classmethod
    def load(cls, file: str | Path, registered_class=None):
        data = _load_yaml(file, registered_class)
        assert isinstance(
            data, dict
        ), f"The config at {file} is registered to a class. Use load_as_object() instead"
        return Configuration(data, registered_class=registered_class)

    @classmethod
    def load_as_object(cls, file: str | Path, registered_class=None):
        data = _load_yaml(file, registered_class)
        assert not isinstance(
            data, dict
        ), f"The config at {file} is not registered to a class. Use load() instead"
        return data

    def save(self, out_file: str | Path):
        _save_yaml(self._data_dict, out_file)

    def cls_object_from_this(self):
        assert self._registered_class is not None
        return self._registered_class(**self._data_dict)

    @property
    def data_dict(self):
        return self._data_dict

    @property
    def cls(self):
        return self._registered_class

    def __str__(self) -> str:
        return f"Config({self.cls.__name__ if self.cls is not None else 'NoClass'} = {self.data_dict})"  # type: ignore


#########
# Universal Functions
#########
def _register_class_to_yaml(cls):
    global supported_helper_classes
    if cls not in supported_helper_classes:
        supported_helper_classes.append(cls)


def _load_from_config(cls, path: str | Path):
    # cls._register_permanently()
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"load_from_config: {path} does not exist"
    obj = Configuration.load_as_object(path, registered_class=cls)
    assert isinstance(obj, cls), f"Loaded config was not for class {cls.__name__}"
    return obj


def _load_from_config_name(cls, name: str):
    path = config_by_name(name)
    assert path.exists(), f"load_from_config: {path} does not exist"
    return _load_from_config(cls, path)


def _save_to_config(obj, path: str | Path):
    if isinstance(path, str):
        path = Path(path)
    Configuration.save_from_object(obj, path)


def _save_to_config_by_name(obj, name: str):
    dir, name = config_dir_by_name(name)
    _save_to_config(obj, dir.joinpath(name))


class SupportsConfig:
    """Metaclass that allows a class to save and load objects by yaml configs"""

    def __init__(self) -> None:
        raise NotImplementedError(f"Tried to instantiate abstract class {type(self)}")

    def __init_subclass__(cls, **kwargs):
        # Registers all subclasses of this
        super().__init_subclass__(**kwargs)
        cls._register_permanently()

    @classmethod
    def _register_permanently(cls):
        _register_class_to_yaml(cls)

    @classmethod
    def load_from_config(cls, path: str | Path):
        obj = _load_from_config(cls, path)
        assert isinstance(
            obj, cls
        ), f"loaded object was not of the correct class, expected {cls.__name__} but got {type(obj)}"
        return obj

    @classmethod
    def load_from_config_name(cls, name: str):
        obj = _load_from_config_name(cls, name)
        assert isinstance(obj, cls)
        return obj

    def save_to_config(self, path: str | Path):
        _save_to_config(self, path)

    def save_to_config_by_name(self, name: str):
        _save_to_config_by_name(self, name)

    @classmethod
    def to_yaml(cls, representer, node):
        # cls._register_permanently()
        assert hasattr(
            cls, "_yaml_repr"
        ), f"Class {cls.__name__} has no _yaml_repr(cls, node) defined"
        return representer.represent_mapping("!" + cls.__name__, cls._yaml_repr(node))

    @classmethod
    def from_yaml(cls, constructor, node):
        # cls._register_permanently()
        data = constructor.construct_mapping(node, deep=True)
        return cls(**data)

    @classmethod
    @abstractmethod
    def _yaml_repr(cls, node) -> dict:
        pass  # return {"groups": node.__group_dictionary}
