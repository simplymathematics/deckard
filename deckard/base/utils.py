from importlib import import_module
from copy import deepcopy
from typing import Union, Tuple
from pathlib import Path
import yaml
import logging
logger = logging.getLogger(__name__)



def factory(module_class_string,  *args, super_cls: type = None, **kwargs) -> object:
    """
    :param module_class_string: full name of the class to create an object of
    :param super_cls: expected super class for validity, None if bypass
    :param kwargs: parameters to pass
    :return:
    """
    try:
        module_name, class_name = module_class_string.rsplit(".", 1)
    except Exception as e:  # noqa E722
        logger.warning(f"Invalid module_class_string: {module_class_string}")
        raise e
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


def parse_config_for_libraries(
    path: Union[str, Path],
    regex: str = r"(.*)\.yml",
    output: Union[str, Path] = "requirements.txt",
) -> Tuple[list, Path]:
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
            libraries.append(config["model"]["init"]["name"].split(".")[0])
            if "transform" in config["model"]:
                for key in config["model"]["transform"]:
                    libraries.append(key.split(".")[0])
        if "scorers" in config:
            scorers = config["scorers"]
            for scorer in scorers:
                libraries.append(scorers[scorer]["name"].split(".")[0])
        if "attack" in config:
            libraries.append(config["attack"]["init"]["name"].split(".")[0])
    filename = path / output
    libraries = list(set(libraries))
    with filename.open("w") as f:
        for library in libraries:
            f.write(library + "\n")
    assert filename.exists(), "File {} does not exist".format(filename)
    return (libraries, filename.resolve())
