from deckard.base.model import Model
from deckard.base.data import Data
from typing import Union
from deckard.base.experiment import Experiment


def generate_experiment_list(
    model_list: Union[Model, list],
    data_list: Union[Data, list],
    model_type="sklearn",
    **kwargs,
) -> list:
    """
    Generates experiment list from model list.
    :param model_list: list of models
    :param data_list: data object
    """
    assert isinstance(
        model_list, (Model, list)
    ), "model_list must be a Model or a list of Models"
    assert isinstance(
        data_list, (Data, list)
    ), "data must be a Data object or a list of Data objects"
    if isinstance(model_list, Model):
        model_list = [model_list]
    if isinstance(data_list, Data):
        data_list = [data_list]
    experiment_list = list()
    for data in data_list:
        for model in model_list:
            if not isinstance(model, Model):
                model = Model(model, model_type=model_type, **kwargs)
            experiment = Experiment(data=data, model=model)
            experiment_list.append(experiment)
    return experiment_list
