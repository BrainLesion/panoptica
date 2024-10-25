from ruamel.yaml import YAML
from pathlib import Path
from panoptica.utils.filepath import config_by_name, config_dir_by_name
from abc import ABC, abstractmethod

supported_helper_classes = []


def _register_helper_classes(yaml: YAML):
    """Registers globally supported helper classes to a YAML instance.

    Args:
        yaml (YAML): The YAML instance to register helper classes to.
    """
    [yaml.register_class(s) for s in supported_helper_classes]


def _load_yaml(file: str | Path, registered_class=None):
    """Loads a YAML file into a Python dictionary or object, with optional class registration.

    Args:
        file (str | Path): Path to the YAML file.
        registered_class (optional): Optional class to register with the YAML parser.

    Returns:
        dict | object: Parsed content from the YAML file.
    """
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
    """Saves a Python dictionary or object to a YAML file, with optional class registration.

    Args:
        data_dict (dict | object): Data to save.
        out_file (str | Path): Output file path.
        registered_class (optional): Class type to register with YAML if saving an object.
    """
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


#########
# Universal Functions
#########
def _register_class_to_yaml(cls):
    """Registers a class to the global supported helper classes for YAML serialization.

    Args:
        cls: The class to register.
    """
    global supported_helper_classes
    if cls not in supported_helper_classes:
        supported_helper_classes.append(cls)


def _load_from_config(cls, path: str | Path):
    """Loads an instance of a class from a YAML configuration file.

    Args:
        cls: The class type to instantiate.
        path (str | Path): Path to the YAML configuration file.

    Returns:
        An instance of the specified class, loaded from configuration.
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"load_from_config: {path} does not exist"
    obj = _load_yaml(path, registered_class=cls)
    assert isinstance(obj, cls), f"Loaded config was not for class {cls.__name__}"
    return obj


def _load_from_config_name(cls, name: str):
    """Loads an instance of a class from a configuration file identified by name.

    Args:
        cls: The class type to instantiate.
        name (str): The name used to find the configuration file.

    Returns:
        An instance of the specified class.
    """
    path = config_by_name(name)
    assert path.exists(), f"load_from_config: {path} does not exist"
    return _load_from_config(cls, path)


def _save_to_config(obj, path: str | Path):
    """Saves an instance of a class to a YAML configuration file.

    Args:
        obj: The object to save.
        path (str | Path): The file path to save the configuration.
    """
    if isinstance(path, str):
        path = Path(path)
    _save_yaml(obj, path, registered_class=type(obj))


def _save_to_config_by_name(obj, name: str):
    """Saves an instance of a class to a configuration file by name.

    Args:
        obj: The object to save.
        name (str): The name used to determine the configuration file path.
    """
    dir, name = config_dir_by_name(name)
    _save_to_config(obj, dir.joinpath(name))


class SupportsConfig:
    """Base class that provides methods for loading and saving instances as YAML configurations.

    This class should be inherited by classes that wish to have load and save functionality for YAML
    configurations, with class registration to enable custom serialization and deserialization.

    Methods:
        load_from_config(cls, path): Loads a class instance from a YAML file.
        load_from_config_name(cls, name): Loads a class instance from a configuration file identified by name.
        save_to_config(path): Saves the instance to a YAML file.
        save_to_config_by_name(name): Saves the instance to a configuration file identified by name.
        to_yaml(cls, representer, node): YAML serialization method (requires _yaml_repr).
        from_yaml(cls, constructor, node): YAML deserialization method.
    """

    def __init__(self) -> None:
        """Prevents instantiation of SupportsConfig as it is intended to be a metaclass."""
        raise NotImplementedError(f"Tried to instantiate abstract class {type(self)}")

    def __init_subclass__(cls, **kwargs):
        """Registers subclasses of SupportsConfig to enable YAML support."""
        super().__init_subclass__(**kwargs)
        cls._register_permanently()

    @classmethod
    def _register_permanently(cls):
        """Registers the class to globally supported helper classes."""
        _register_class_to_yaml(cls)

    @classmethod
    def load_from_config(cls, path: str | Path):
        """Loads an instance of the class from a YAML file.

        Args:
            path (str | Path): The file path to load the configuration.

        Returns:
            An instance of the class.
        """
        obj = _load_from_config(cls, path)
        assert isinstance(
            obj, cls
        ), f"loaded object was not of the correct class, expected {cls.__name__} but got {type(obj)}"
        return obj

    @classmethod
    def load_from_config_name(cls, name: str):
        """Loads an instance of the class from a configuration file identified by name.

        Args:
            name (str): The name used to find the configuration file.

        Returns:
            An instance of the class.
        """
        obj = _load_from_config_name(cls, name)
        assert isinstance(obj, cls)
        return obj

    def save_to_config(self, path: str | Path):
        """Saves the instance to a YAML configuration file.

        Args:
            path (str | Path): The file path to save the configuration.
        """
        _save_to_config(self, path)

    def save_to_config_by_name(self, name: str):
        """Saves the instance to a configuration file identified by name.

        Args:
            name (str): The name used to determine the configuration file path.
        """
        _save_to_config_by_name(self, name)

    @classmethod
    def to_yaml(cls, representer, node):
        """Serializes the class to YAML format.

        Args:
            representer: YAML representer instance.
            node: The object instance to serialize.

        Returns:
            YAML node: YAML-compatible node representation of the object.
        """
        assert hasattr(
            cls, "_yaml_repr"
        ), f"Class {cls.__name__} has no _yaml_repr(cls, node) defined"
        return representer.represent_mapping("!" + cls.__name__, cls._yaml_repr(node))

    @classmethod
    def from_yaml(cls, constructor, node):
        """Deserializes a YAML node to an instance of the class.

        Args:
            constructor: YAML constructor instance.
            node: YAML node to deserialize.

        Returns:
            An instance of the class with attributes populated from YAML data.
        """
        data = constructor.construct_mapping(node, deep=True)
        return cls(**data)

    @classmethod
    @abstractmethod
    def _yaml_repr(cls, node) -> dict:
        """Abstract method for representing the class in YAML.

        Args:
            node: The object instance to represent in YAML.

        Returns:
            dict: A dictionary representation of the class.
        """
        pass  # return {"groups": node.__group_dictionary}
