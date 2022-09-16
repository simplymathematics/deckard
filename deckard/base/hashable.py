
from hashlib import md5 as my_hash

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
        return int(my_hash(str(self.__repr__()).encode('utf-8')).hexdigest(), 32)
    
    def get_params(self):
        """
        Returns the parameters of the data object.
        """
        results = {}
        for key, value in self.params.items():
            if isinstance(value, (int, float, str, list, dict, tuple)):
                result = value
            elif isinstance(value, BaseHashable):
                result = vars(value)
            elif hasattr(value, 'params'):
                result = vars(value.params)
            elif hasattr(value, '__iter__'):
                result = [vars(item) for item in value]
            elif hasattr(value, '__dict__'):
                result = vars(value)
            else:
                result = hash(value)
            results[key] = result
            return results

    def set_params(self, params:dict = None):
        """
        :param params: A dictionary of parameters to set.
        Sets the parameters of the data object.
        """
        self.__init__(**params)
        try:
            self.__call__()
        except Exception as e:
            raise e