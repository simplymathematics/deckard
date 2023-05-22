import logging
from itertools import product

__all__ = ["flatten_dict", "unflatten_dict", "make_grid"]
logger = logging.getLogger(__name__)


def flatten_dict(dictionary: dict, separator: str = ".", prefix: str = ""):
    """
    Flattens a dictionary into a list of dictionarieswith keys separated by the separator.
    :param dictionary: The dictionary to flatten.
    :param separator: The separator to use when flattening the dictionary.
    :param prefix: The prefix to use when flattening the dictionary.
    :return: A flattened dictionary.
    """
    stack = [(dictionary, prefix)]
    flat_dict = {}
    while stack:
        cur_dict, cur_prefix = stack.pop()
        for key, val in cur_dict.items():
            new_key = cur_prefix + separator + key if cur_prefix else key
            if isinstance(val, dict):
                logger.debug(f"Flattening {val} into {new_key}")
                stack.append((val, new_key))
            else:
                logger.debug(f"Adding {val} to {new_key}")
                flat_dict[new_key] = val
    return flat_dict


def unflatten_dict(dictionary: dict, separator: str = ".") -> dict:
    """Unflattens a dictionary into a nested dictionary.
    :param dictionary: The dictionary to unflatten.
    :param separator: The separator to use when unflattening the dictionary.
    :param prefix: The prefix to use when unflattening the dictionary.
    :return: An unflattened dictionary.
    """
    result = {}
    for key, val in dictionary.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        logger.debug(f"Adding {val} to {parts[-1]}")
        d[parts[-1]] = val
    return result


def make_grid(dictionary: list) -> list:
    """
    Makes a grid of parameters from a dictionary of lists.
    :param dictionary: The dictionary of lists to make a grid from.
    :return: A list of dictionaries with all possible combinations of parameters.
    """
    big = []
    if not isinstance(dictionary, list):
        assert isinstance(
            dictionary,
            dict,
        ), f"dictionary must be a list or dict, not {type(dictionary)}"
        dict_list = [dictionary]
    else:
        dict_list = dictionary
    for dictionary in dict_list:
        for k, v in dictionary.items():
            if isinstance(v, dict):
                logger.debug(f"Making grid from {v}")
                dictionary[k] = make_grid(v)
            elif isinstance(v, list):
                logger.debug(f"{v} is a list.")
                dictionary[k] = v
            else:
                logger.debug(f"{v} is type {type(v)}")
                dictionary[k] = [v]
        keys = dictionary.keys()
        combinations = product(*dictionary.values())
        big.extend(combinations)
    return [dict(zip(keys, cc)) for cc in big]
