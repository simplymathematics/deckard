from hashlib import md5 as my_hash
from numpy import ndarray


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

    def __iter__(self):
        """
        Iterates through the data object.
        """
        for key, value in self.params.items():

            yield key, value

    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        return int(my_hash(str(self.__repr__()).encode("utf-8")).hexdigest(), 32)

    def get_params(self, **kwargs):
        """
        Returns the parameters of the data object.
        """
        results = {}
        results.update(**kwargs)
        for key, value in self.params.items():
            if hasattr(value, "get_params") and not isinstance(value, BaseHashable):
                result = value.get_params()
            if isinstance(value, (int, float, str, list, tuple)):
                result = value
            elif isinstance(value, BaseHashable):
                result = vars(value)
            elif isinstance(value, dict):
                result = value
            elif hasattr(value, "params"):
                result = vars(value.params)
            elif hasattr(value, "__dict__"):
                result = vars(value)
            elif isinstance(value, type(None)):
                result = None
            elif isinstance(value, ndarray):
                value = value.tolist()
            else:
                result = hash(value)
            results[key] = result
            return results
