from hashlib import md5
from numpy import ndarray
from typing import Callable
import pathlib


def my_hash(obj):
    return str(md5(str(obj).encode("utf-8")).hexdigest())


class BaseHashable(object):
    def __eq__(self, other) -> bool:
        """
        Checks if the data is equal to another data object, using the params as specified in the __init__ method.
        """
        return hash(str(self.params)) == hash(str(other.params))

    def __repr__(self) -> str:
        """
        Returns the human-readable string representation of the dataset
        """
        return str(self.params)

    def __str__(self) -> str:
        """
        Returns the reproducible representation of the data object.
        """
        dict_ = {**self.params}
        return f"deckard.base.hashable({dict_})"

    # def __iter__(self):
    #     """
    #     Iterates through the data object.
    #     """
    #     for key, value in self.params.items():

    #         yield key, value

    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        return int(my_hash(str(self.__repr__())), 32)

    def get_params(self):
        """
        Returns the parameters of the data object.
        """
        results = dict(self.params)
        for key, value in results.items():
            if isinstance(
                value,
                (pathlib.Path, pathlib.WindowsPath, pathlib.PosixPath),
            ):
                result = pathlib.Path(value).name
            elif isinstance(value, ndarray):
                result = my_hash(value.tolist())
            elif isinstance(value, (int, float, str, list, tuple)):
                result = value
            elif isinstance(value, Callable):
                result = value.__name__
            elif isinstance(value, BaseHashable):
                result = my_hash(value)
            elif isinstance(value, dict):
                result = value
            elif isinstance(value, type(None)):
                result = None
            else:
                result = my_hash(value)
            results[key] = str(result)
        return results
