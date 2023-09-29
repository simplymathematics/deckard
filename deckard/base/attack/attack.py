from dataclasses import dataclass, field
import numpy as np
from typing import Union
from pathlib import Path
import logging
from copy import deepcopy
from time import process_time_ns
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from art.utils import to_categorical, compute_success
from ..data import Data
from ..model import Model
from ..utils import my_hash

__all__ = [
    "Attack",
    "EvasionAttack",
    "PoisoningAttack",
    "InferenceAttack",
    "ExtractionAttack",
    "ReconstructionAttack",
    "AttackInitializer",
]

logger = logging.getLogger(__name__)


@dataclass
class AttackInitializer:
    name: str = field(default_factory=str)
    model: Model = field(default_factory=Model)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, model: Model, name: str, **kwargs):
        if isinstance(model, Model):
            self.model = model
        elif isinstance(model, DictConfig):
            model = Model(**model)
            self.model = model
        elif isinstance(model, dict):
            model = Model(**model)
            self.model = model
        else:
            raise TypeError(
                f"model must be of type Model, DictConfig, or dict. Got {type(model)}",
            )
        self.name = name
        self.kwargs = kwargs

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, model=None, data=None, attack_size=-1):
        logger.info(f"Fitting attack {self.name} with id: {self.__hash__()}")
        name = self.name
        kwargs = deepcopy(self.kwargs)
        pop_list = ["extract", "poison", "evade", "reconstruct", "infer"]
        for thing in pop_list:
            kwargs.pop(thing, None)
        logger.info(f"Initializing attack {name} with parameters {kwargs}")
        self.data = data
        self.model = model
        if "x_train" in kwargs:
            assert (
                data is not None
            ), "Data must be provided to call function if x_train is kwargs."
            kwargs["x_train"] = data[0][:attack_size]
        if "y_train" in kwargs:
            assert (
                data is not None
            ), "Data must be provided to call function if y_train is kwargs."
            y_train = data[2][:attack_size]
            if len(np.squeeze(y_train).shape) < 2:
                kwargs["y_train"] = to_categorical(y_train)
            else:
                kwargs["y_train"] = y_train
        if "x_val" in kwargs:
            assert (
                data is not None
            ), "Data must be provided to call function if x_val is kwargs."
            kwargs["x_val"] = data[1][:attack_size]
        if "y_val" in kwargs:
            assert (
                data is not None
            ), "Data must be provided to call function if y_val is kwargs."
            y_test = data[3][:attack_size]
            if len(np.squeeze(y_test).shape) < 2:
                kwargs["y_val"] = to_categorical(y_test)
            else:
                kwargs["y_val"] = y_test
        try:
            logger.info("Attempting black-box attack.")
            config = {"_target_": name}
            config.update(**kwargs)
            attack = instantiate(config, model)
        except TypeError as e:
            if "verbose" in str(e):
                config.pop("verbose", None)
                attack = instantiate(config, model)
            else:
                raise e
        return attack


@dataclass
class EvasionAttack:
    name: str = field(default_factory=str)
    init: Union[AttackInitializer, None] = field(default_factory=AttackInitializer)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    attack_size: int = -1
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(
        self, name: str, data: Data, model: Model, init: dict, attack_size=-1, **kwargs
    ):
        self.name = name
        self.data = data
        self.model = model
        self.attack_size = attack_size
        self.init = AttackInitializer(model, name, **init)
        self.kwargs = kwargs
        logger.info("Instantiating Attack with id: {}".format(self.__hash__()))

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(
        self,
        data=None,
        model=None,
        attack_file=None,
        adv_probabilities_file=None,
        adv_predictions_file=None,
        adv_losses_file=None,
    ):
        time_dict = {}
        results = {}
        kwargs = deepcopy(self.init.kwargs)
        scale_max = kwargs.pop("scale_max", 1)
        targeted = kwargs.pop("targeted", False)
        ben_samples = data[1][: self.attack_size]
        if attack_file is not None and Path(attack_file).exists():
            samples = self.data.load(attack_file)
        else:

            atk = self.init(model=model, attack_size=self.attack_size)

            if targeted is True:
                kwargs.update({"y": data[2][: self.attack_size]})
            if "AdversarialPatch" in self.name:
                start = process_time_ns()
                patches, masks = atk.generate(ben_samples, **kwargs)
                samples = atk.apply_patch(ben_samples, scale=scale_max)
            else:
                start = process_time_ns()
                samples = atk.generate(ben_samples, **kwargs)
            end = process_time_ns()
            time_dict.update({"adv_fit_time": (end - start) / 1e9})
            time_dict.update(
                {"adv_fit_time_per_sample": (end - start) / (len(samples) * 1e9)},
            )
        results["adv_samples"] = samples
        try:
            results["adv_success"] = compute_success(
                classifier=model,
                x_clean=ben_samples,
                labels=data[3][: self.attack_size],
                x_adv=samples,
                targeted=False,
            )
        except TypeError as e:
            logger.error(f"Failed to compute success rate. Error: {e}")
        if attack_file is not None:
            self.data.save(samples, attack_file)
        if adv_predictions_file is not None and Path(adv_predictions_file).exists():
            adv_predictions = self.data.load(adv_predictions_file)
            results["adv_predictions"] = adv_predictions
        else:
            adv_predictions = model.predict(samples)
            results["adv_predictions"] = adv_predictions
        if adv_predictions_file is not None:
            self.data.save(adv_predictions, adv_predictions_file)
        if adv_probabilities_file is not None and Path(adv_probabilities_file).exists():
            adv_probabilities = self.data.load(adv_probabilities_file)
            results["adv_probabilities"] = adv_probabilities
        else:
            if hasattr(self.model, "model") and hasattr(
                self.model.model,
                "predict_proba",
            ):
                start = process_time_ns()
                adv_probabilities = model.model.predict_proba(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
            try:
                start = process_time_ns()
                adv_probabilities = model.predict_proba(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
            except AttributeError:
                start = process_time_ns()
                adv_probabilities = model.predict(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
            results["adv_probabilities"] = adv_probabilities
        if adv_probabilities_file is not None:
            self.data.save(adv_probabilities, adv_probabilities_file)

        if adv_losses_file is not None and Path(adv_losses_file).exists():
            adv_loss = self.data.load(adv_losses_file)
            results["adv_loss"] = adv_loss
        elif adv_losses_file is not None:
            assert hasattr(
                model,
                "compute_loss",
            ), "Model does not have compute_loss method."
            try:
                adv_loss = model.compute_loss(samples, data[3][: self.attack_size])
            except NotImplementedError:
                from sklearn.metrics import log_loss

                preds = model.predict(samples[: self.attack_size])
                adv_loss = log_loss(data[3][: self.attack_size], preds)
            self.data.save(adv_loss, adv_losses_file)
            results["adv_loss"] = adv_loss
        if len(time_dict) > 0:
            results["time_dict"] = time_dict
        return results


@dataclass
class PoisoningAttack:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: dict = field(default_factory=dict)
    attack_size: int = -1
    target_image: str = None

    def __init__(
        self,
        name: str,
        data: Data,
        model: Model,
        init: dict,
        attack_size=-1,
        target_image=None,
        **kwargs,
    ):
        if target_image is not None:
            assert Path(
                target_image,
            ).exists(), f"target_image must be a valid path. Got {target_image}"
        self.target_image = target_image
        self.name = name
        self.data = data
        self.model = model
        self.attack_size = attack_size
        self.init = AttackInitializer(model, name, **init)
        self.kwargs = kwargs
        logger.info("Instantiating Attack with id: {}".format(self.__hash__()))

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(
        self,
        data=None,
        model=None,
        attack_file=None,
        adv_probabilities_file=None,
        adv_predictions_file=None,
        adv_losses_file=None,
    ):
        time_dict = {}
        results = {}
        params = deepcopy(self.kwargs.pop("poison", {}))
        if attack_file is not None and Path(attack_file).exists():
            samples = self.data.load(attack_file)
        else:
            # target_image = self.data.load(self.target_image)
            atk = self.init(model=model, data=data, attack_size=self.attack_size)
            if "GradientMatching" in self.name:
                trigger = params
                x_train = data[0][: self.attack_size]
                y_train = data[2][: self.attack_size]
                if isinstance(trigger, str):
                    x_trigger, y_trigger = self.data(data_file=trigger)
                elif isinstance(trigger, DictConfig):
                    trigger_dict = OmegaConf.to_container(trigger, resolve=True)
                    trigger_data = Data(**trigger_dict)
                    x_trigger, y_trigger = trigger_data()
                elif isinstance(trigger, dict):
                    trigger_data = Data(**trigger)
                    x_trigger, y_trigger = trigger_data()
                elif isinstance(trigger, Data):
                    x_trigger, _, y_trigger, _ = trigger()
                else:
                    raise TypeError(
                        f"trigger must be a path to a data file or a DictConfig. Got {type(trigger)}",
                    )
                try:
                    start = process_time_ns()
                    samples, _ = atk.poison(
                        x_trigger=x_trigger,
                        y_trigger=y_trigger,
                        x_train=x_train,
                        y_train=y_train,
                    )
                except RuntimeError as e:
                    if "expected scalar type Long" in str(e):
                        # if hasattr(y_train, "type"):
                        import torch

                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu",
                        )
                        y_train = torch.tensor(y_train, device=device)
                        y_trigger = torch.tensor(y_trigger, device=device)
                        x_train = torch.tensor(x_train, device=device)
                        x_trigger = torch.tensor(x_trigger, device=device)
                        y_trigger = y_trigger.to(torch.long)
                        y_trigger = y_trigger.to(torch.long)
                        start = process_time_ns()
                        samples, _ = atk.poison(
                            x_trigger=x_trigger,
                            y_trigger=y_trigger,
                            x_train=x_train,
                            y_train=y_train,
                        )
                    else:
                        raise e
                end = process_time_ns()
            time_dict.update({"adv_fit_time": (end - start) / 1e9})
            time_dict.update(
                {"adv_fit_time_per_sample": (end - start) / (len(samples) * 1e9)},
            )
        results["adv_samples"] = samples
        results["time_dict"] = time_dict
        if attack_file is not None:
            self.data.save(samples, attack_file)
        if adv_predictions_file is not None and Path(adv_predictions_file).exists():
            adv_predictions = self.data.load(adv_predictions_file)
        else:
            adv_predictions = model.predict(samples)
        results["adv_predictions"] = adv_predictions
        if adv_predictions_file is not None:
            self.data.save(adv_predictions, adv_predictions_file)
        if adv_probabilities_file is not None and Path(adv_probabilities_file).exists():
            adv_probabilities = self.data.load(adv_probabilities_file)
        else:
            if hasattr(self.model, "model") and hasattr(
                self.model.model,
                "predict_proba",
            ):
                start = process_time_ns()
                adv_probabilities = model.model.predict_proba(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
            try:
                start = process_time_ns()
                adv_probabilities = model.predict_proba(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
            except AttributeError:
                start = process_time_ns()
                adv_probabilities = model.predict(samples)
                end = process_time_ns()
                time_dict.update({"adv_predict_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_predict_time_per_sample": (end - start)
                        / (len(samples) * 1e9),
                    },
                )
        results["adv_probabilities"] = adv_probabilities
        if adv_probabilities_file is not None:
            self.data.save(adv_probabilities, adv_probabilities_file)

        if adv_losses_file is not None and Path(adv_losses_file).exists():
            adv_loss = self.data.load(adv_losses_file)
        elif adv_losses_file is not None:
            assert hasattr(
                model,
                "compute_loss",
            ), "Model does not have compute_loss method."
            adv_loss = model.compute_loss(samples, data[3][: self.attack_size])
            self.data.save(adv_loss, adv_losses_file)
        else:
            adv_loss = None
        results["adv_loss"] = adv_loss
        results["time_dict"] = time_dict
        return results


@dataclass
class InferenceAttack:
    name: str = field(default_factory=str)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: dict = field(default_factory=dict)
    initial_image: str = None
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(
        self, name: str, data: Data, model: Model, init: dict, attack_size=-1, **kwargs
    ):
        self.name = name
        self.data = data
        self.model = model
        self.attack_size = attack_size
        self.init = AttackInitializer(model, name, **init)
        self.kwargs = kwargs
        logger.info("Instantiating Attack with id: {}".format(self.__hash__()))

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(
        self,
        data=None,
        model=None,
        attack_file=None,
        adv_probabilities_file=None,
        adv_predictions_file=None,
        adv_losses_file=None,
    ):
        data_shape = data[0][0].shape
        time_dict = {}
        results = {}
        if attack_file is not None and Path(attack_file).exists():
            preds = self.data.load(attack_file)
        elif adv_predictions_file is not None and Path(adv_predictions_file).exists():
            preds = self.data.load(adv_predictions_file)
        elif (
            adv_probabilities_file is not None and Path(adv_probabilities_file).exists()
        ):
            preds = self.data.load(adv_probabilities_file)
        elif adv_losses_file is not None and Path(adv_losses_file).exists():
            preds = self.data.load(adv_losses_file)
        else:
            if self.initial_image == "white":
                initial_image = np.zeros(data_shape)
            elif self.initial_image == "grey":
                initial_image = np.ones(data_shape) * 0.5
            elif self.initial_image == "black":
                initial_image = np.ones(data_shape)
            elif self.initial_image == "random":
                initial_image = np.random.uniform(0, 1, data_shape)
            elif self.initial_image == "average":
                initial_image = np.zeroes(data_shape) + np.mean(
                    data[1][: self.attack_size],
                    axis=0,
                )
            elif self.initial_image is None:
                pass
            else:
                raise ValueError(
                    f"initial_image must be one of ['white', 'black', 'grey', 'random', 'average']. Got {self.initial_image}",
                )
            if self.initial_image is not None:
                initial_image = initial_image.astype(np.float32)
                initial_image = np.array([initial_image]) * self.attack_size
            atk = self.init(model=model, attack_size=self.attack_size)
            x_train = data[0][: self.attack_size]
            y_train = data[2][: self.attack_size]
            x_test = data[1][: self.attack_size]
            y_test = data[3][: self.attack_size]
            if "MembershipInferenceBlackBox" in self.name:
                infer = self.kwargs.pop("infer", {})
                fit = self.kwargs.pop("fit", {})
                start = process_time_ns()
                atk.fit(x=x_train, y=y_train, test_x=x_test, test_y=y_test, **fit)
                end = process_time_ns()
                time_dict.update({"adv_fit_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_fit_time_per_sample": (end - start)
                        / (self.attack_size * 1e9),
                    },
                )
                x_train = data[0][: self.attack_size]
                y_train = data[2][: self.attack_size]
                x_test = data[1][: self.attack_size]
                y_test = data[3][: self.attack_size]
                start = process_time_ns()
                preds = atk.infer(x_test, y_test, **infer)
                end = process_time_ns()
            else:
                raise NotImplementedError(f"Attack {self.name} not implemented.")
            time_dict.update({"adv_fit_time": (end - start) / 1e9})
            time_dict.update(
                {"adv_fit_time_per_sample": (end - start) / (self.attack_size * 1e9)},
            )
        results["time_dict"] = time_dict
        results["adv_predictions"] = preds
        results["time_dict"] = time_dict
        if adv_predictions_file is not None:
            self.data.save(preds, adv_predictions_file)
        if adv_probabilities_file is not None:
            self.data.save(preds, adv_probabilities_file)
        if adv_losses_file is not None:
            self.data.save(preds, adv_losses_file)
        if attack_file is not None:
            self.data.save(preds, attack_file)
        return results


@dataclass
class ExtractionAttack:
    name: str = field(default_factory=str)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: dict = field(default_factory=dict)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(
        self, name: str, data: Data, model: Model, init: dict, attack_size=-1, **kwargs
    ):
        self.name = name
        self.data = data
        self.model = model
        self.attack_size = attack_size
        self.init = AttackInitializer(model, name, **init)
        if isinstance(kwargs, DictConfig):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        elif isinstance(kwargs, dict):
            pass
        else:
            raise TypeError(
                f"kwargs must be of type DictConfig or dict. Got {type(kwargs)}",
            )
        self.kwargs = kwargs
        logger.info("Instantiating Attack with id: {}".format(self.__hash__()))

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(
        self,
        data=None,
        model=None,
        attack_file=None,
        adv_probabilities_file=None,
        adv_predictions_file=None,
        adv_losses_file=None,
    ):
        results = {}
        time_dict = {}
        kwargs = deepcopy(self.kwargs.pop("extract", {}))
        assert len(data) == 4, f"Expected 4, got {len(data)}"
        assert hasattr(model, "fit"), "Model does not have fit method."
        assert hasattr(model, "predict"), "Model does not have predict method."
        if attack_file is None or not Path(attack_file).exists():
            if "KnockoffNets" in self.name:
                assert (
                    "thieved_classifier" in kwargs
                ), f"thieved_classifier must be provided for KnockoffNets attack. Provided keys are {kwargs.keys()}"
                mod = Model(**kwargs.pop("thieved_classifier"))
                _, thieved_classifier = mod.initialize(data)
                attack = deepcopy(self.init)
                attack = attack(model=model, attack_size=self.attack_size)
                start = process_time_ns()
                attacked_model = attack.extract(
                    x=data[0][: self.attack_size],
                    y=data[2][: self.attack_size],
                    thieved_classifier=thieved_classifier,
                    **kwargs,
                )
                end = process_time_ns()
                time_dict.update({"adv_fit_time": (end - start) / 1e9})
                time_dict.update(
                    {
                        "adv_fit_time_per_sample": (end - start)
                        / (self.attack_size * 1e9),
                    },
                )
            else:
                raise NotImplementedError(f"Attack {self.name} not implemented.")
        else:
            attacked_model = self.model.load(attack_file)
        results["adv_model"] = attacked_model
        # Get predictions from adversarial model
        if adv_predictions_file is not None and Path(adv_predictions_file).exists():
            preds = self.data.load(adv_predictions_file)
        else:
            start = process_time_ns()
            preds = attacked_model.predict(data[1][: self.attack_size])
            end = process_time_ns()
            time_dict.update({"adv_predict_time": (end - start) / 1e9})
            time_dict.update(
                {"adv_predict_time_per_sample": (end - start) / (len(preds) * 1e9)},
            )
        results["time_dict"] = time_dict
        results["adv_predictions"] = preds

        # Get probabilities from adversarial model
        if adv_probabilities_file is not None and Path(adv_probabilities_file).exists():
            probs = self.data.load(adv_probabilities_file)
        else:
            if hasattr(attacked_model, "predict_proba"):
                probs = attacked_model.predict_proba(data[1][: self.attack_size])
            else:
                probs = preds
        results["adv_probabilities"] = probs
        # Get loss from adversarial model

        if adv_losses_file is not None and Path(adv_losses_file).exists():
            loss = self.data.load(adv_losses_file)
        else:
            if hasattr(attacked_model, "compute_loss"):
                loss = attacked_model.compute_loss(
                    data[1][: self.attack_size],
                    data[3][: self.attack_size],
                )
            else:
                from sklearn.metrics import log_loss

                loss = log_loss(data[3][: self.attack_size], preds)
        results["adv_loss"] = loss

        # Save files
        if adv_predictions_file is not None:
            self.data.save(preds, adv_predictions_file)

        if adv_probabilities_file is not None:
            self.data.save(probs, adv_probabilities_file)

        if adv_losses_file is not None:
            self.data.save(loss, adv_losses_file)

        if attack_file is not None:
            self.model.save(attacked_model, attack_file)

        return results


@dataclass
class ReconstructionAttack(EvasionAttack):
    def __call__(self):
        raise NotImplementedError("ReconstructionAttack not implemented.")


@dataclass
class Attack:
    name: str
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    method: str = "evasion"
    init: AttackInitializer = field(default_factory=AttackInitializer)
    name: str = None
    kwargs: Union[dict, None] = field(default_factory=dict)
    attack_size: int = -1

    def __init__(
        self,
        data: Data,
        model: Model,
        method: str = "evasion",
        init: AttackInitializer = field(default_factory=AttackInitializer),
        attack_size: int = -1,
        name=None,
        **kwargs,
    ):
        if isinstance(data, Data):
            self.data = data
        elif isinstance(data, DictConfig):
            data = Data(**data)
            self.data = data
        elif isinstance(data, dict):
            data = Data(**data)
            self.data = data
        else:
            raise TypeError(
                f"data must be of type Data, DictConfig, or dict. Got {type(data)}",
            )
        if isinstance(model, Model):
            self.model = model
        elif isinstance(model, DictConfig):
            model = Model(**model)
            self.model = model
        elif isinstance(model, dict):
            model = Model(**model)
            self.model = model
        else:
            raise TypeError(
                f"model must be of type Model, DictConfig, or dict. Got {type(model)}",
            )
        if isinstance(init, AttackInitializer):
            self.init = init
        elif isinstance(init, DictConfig):
            init = OmegaConf.to_container(init, resolve=True)
            self.init = AttackInitializer(**init)
        elif isinstance(init, dict):
            init = AttackInitializer(**init)
            self.init = init
        else:
            raise ValueError(
                f"init must be of type AttackInitializer, DictConfig, or dict. Got {type(init)}",
            )
        self.method = method
        self.attack_size = attack_size
        if isinstance(kwargs, DictConfig):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        elif isinstance(kwargs, dict):
            pass
        else:
            raise TypeError(
                f"kwargs must be of type DictConfig or dict. Got {type(kwargs)}",
            )
        while "kwargs" in kwargs:
            kwargs.update(**kwargs.pop("kwargs"))
        self.kwargs = kwargs
        self.name = name if name is not None else my_hash(self)
        logger.info("Instantiating Attack with id: {}".format(self.name))

    def __call__(
        self,
        data=None,
        model=None,
        attack_file=None,
        adv_predictions_file=None,
        adv_probabilities_file=None,
        adv_losses_file=None,
    ):
        name = self.init.name
        kwargs = deepcopy(self.kwargs)
        kwargs.update({"init": self.init.kwargs})
        data = self.data()
        data, model = self.model.initialize(data)
        if "art" not in str(type(model)):
            model = self.model.art(model=model, data=data)
        if self.method == "evasion":
            attack = EvasionAttack(
                name=name,
                data=self.data,
                model=self.model,
                attack_size=self.attack_size,
                **kwargs,
            )
            result = attack(
                data,
                model,
                attack_file=attack_file,
                adv_predictions_file=adv_predictions_file,
                adv_probabilities_file=adv_probabilities_file,
                adv_losses_file=adv_losses_file,
            )
        elif self.method == "poisoning":
            attack = PoisoningAttack(
                name=name,
                data=self.data,
                model=self.model,
                attack_size=self.attack_size,
                **kwargs,
            )
            result = attack(
                data,
                model,
                attack_file=attack_file,
                adv_predictions_file=adv_predictions_file,
                adv_probabilities_file=adv_probabilities_file,
                adv_losses_file=adv_losses_file,
            )
        elif self.method == "inference":
            attack = InferenceAttack(
                name=name,
                data=self.data,
                model=self.model,
                attack_size=self.attack_size,
                **kwargs,
            )
            result = attack(
                data,
                model,
                attack_file=attack_file,
                adv_predictions_file=adv_predictions_file,
                adv_probabilities_file=adv_probabilities_file,
                adv_losses_file=adv_losses_file,
            )
        elif self.method == "extraction":
            attack = ExtractionAttack(
                name=name,
                data=self.data,
                model=self.model,
                attack_size=self.attack_size,
                **kwargs,
            )
            result = attack(
                data,
                model,
                attack_file=attack_file,
                adv_predictions_file=adv_predictions_file,
                adv_probabilities_file=adv_probabilities_file,
                adv_losses_file=adv_losses_file,
            )
        else:
            raise NotImplementedError(f"Attack method {self.method} not implemented.")

        return result

    def __hash__(self):
        return int(self.name, 16)
