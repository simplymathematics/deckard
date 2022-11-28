from importlib import import_module
from copy import deepcopy
import logging
from typing import Union, Tuple
from pathlib import Path
import yaml
logger = logging.getLogger(__name__)


def load_from_tup(obj_tuple: tuple, *args) -> object:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param obj_tuple: (full_object_name, params)
    """
    library_name = ".".join(obj_tuple[0].split(".")[:-1])
    class_name = obj_tuple[0].split(".")[-1]
    global tmp_library
    tmp_library = None
    global temp_object
    temp_object = None
    global params
    global positional_arg
    positional_arg = []
    params = obj_tuple[1]
    if library_name != "":
        tmp_library = import_module(library_name)
        if len(args) > 0:

            positional_arg = args[:]
            exec(
                f"temp_object = tmp_library.{class_name}(positional_arg, **{params})",
                globals(),
            )
            del positional_arg
        elif len(args) == 0:
            exec(f"temp_object = tmp_library.{class_name}(**params)", globals())
        else:
            raise ValueError("Too many positional arguments")
    else:
        if len(args) > 0:
            positional_arg = args[:]
            exec(
                f"temp_object = {class_name}(positional_arg, **{params})",
                globals(),
            )
            del positional_arg
        elif len(args) == 0:
            exec(f"temp_object = {class_name}(**params)", globals())
        else:
            raise ValueError("Too many positional arguments")
    del params
    del tmp_library
    result = deepcopy(temp_object)
    del temp_object
    return result


def factory(module_class_string, super_cls: type = None, *args, **kwargs) -> object:
    """
    :param module_class_string: full name of the class to create an object of
    :param super_cls: expected super class for validity, None if bypass
    :param kwargs: parameters to pass
    :return:
    """
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name,
        module_name,
    )
    logger.debug("reading class {} from module {}".format(class_name, module_name))
    cls = getattr(module, class_name)
    if super_cls is not None:
        assert issubclass(cls, super_cls), "class {} should inherit from {}".format(
            class_name,
            super_cls.__name__,
        )
    logger.debug("initialising {} with params {}".format(class_name, kwargs))
    try:
        obj = cls(*args, **kwargs)
    except Exception as e:
        raise e
    return obj


def parse_config_for_libraries(path:Union[str, Path], regex: str = r"(.*)\.yml", output:Union[str, Path] = "requirements.txt") -> Tuple[list, Path]:
    """
    Parses a folder for yml files and returns a list of libraries
    :param path: path to folder
    :param regex: regex to match files
    :return: list of libraries
    """
    path = Path(path)
    assert path.exists(), "Path does not exist"
    files = path.glob(regex)
    libraries = []
    for file in files:
        config = yaml.unsafe_load(open(file, "r"))
        if "data" in config:
            if "transform" in config["data"]:
                for key in config["data"]["transform"]:
                    libraries.append(key.split(".")[0])
        if "model" in config:
            libraries.append(config["model"]['name'].split(".")[0])
            if "transform" in config["model"]:
                for key in config["model"]["transform"]:
                    libraries.append(key.split(".")[0])
        if "scorers" in config:
            scorers = config["scorers"]
            for scorer in scorers:
                libraries.append(scorers[scorer]['name'].split(".")[0])
        if "attack" in config:
            libraries.append(config["attack"]['name'].split(".")[0])
    filename = path / output
    libraries = list(set(libraries))
    with filename.open("w") as f:
        for library in libraries:
            f.write(library +"\n")
    assert filename.exists(), "File {} does not exist".format(filename)
    return (libraries, filename.resolve())