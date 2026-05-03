import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Subset

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    Tensor = ()
    DataLoader = ()
    Subset = None
    HAS_TORCH = False


def is_torch_model(model) -> bool:
    return HAS_TORCH and isinstance(model, torch.nn.Module)


def is_tensor(value) -> bool:
    return HAS_TORCH and isinstance(value, Tensor)


def is_dataloader(value) -> bool:
    return HAS_TORCH and isinstance(value, DataLoader)


def tensor_to_numpy(value, dtype=None):
    if not is_tensor(value):
        return value
    arr = value.detach().cpu().numpy()
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def get_torch_model_device(model):
    if not is_torch_model(model):
        return torch.device("cpu")
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.device("cpu")
    return getattr(first_param, "device", torch.device("cpu"))


def build_torch_art_model(model, data):
    if not HAS_TORCH:
        raise ImportError("Torch support requires optional dependency deckard[torch]")

    from art.estimators.classification import PyTorchClassifier

    if is_dataloader(data.X_train):
        first_batch = next(iter(data.X_train))
        if isinstance(first_batch, (tuple, list)):
            input_shape = tuple(first_batch[0].shape[1:])
        else:
            input_shape = tuple(first_batch.shape[1:])
    else:
        input_shape = tuple(data.X_train.shape[1:])

    import numpy as np

    nb_classes = len(np.unique(np.asarray(data.y_train).flatten()))
    art_model = model
    target_device = get_torch_model_device(model)
    device_type = "gpu" if getattr(target_device, "type", "cpu") == "cuda" else "cpu"

    estimator = PyTorchClassifier(
        model=art_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(art_model.parameters(), lr=0.01),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type=device_type,
    )

    # Directly override ART device internals because device_type only supports cpu/gpu.
    if hasattr(estimator, "_device"):
        estimator._device = target_device
    if hasattr(estimator, "_model") and hasattr(estimator._model, "to"):
        estimator._model = estimator._model.to(target_device)
    preprocessing = getattr(estimator, "preprocessing", None)
    if hasattr(preprocessing, "_device"):
        preprocessing._device = target_device
    for op in getattr(estimator, "preprocessing_operations", []) or []:
        if hasattr(op, "_device"):
            op._device = target_device

    return estimator


def collect_subset_from_dataloader(loader, n):
    if not HAS_TORCH:
        raise ImportError("Torch support requires optional dependency deckard[torch]")
    if not is_dataloader(loader):
        raise TypeError(f"Expected DataLoader, got {type(loader)}")

    dataset_len = len(loader.dataset)
    n = min(int(n), dataset_len)
    subset = Subset(loader.dataset, indices=range(n))

    x_subset = []
    y_subset = []
    for x, y in subset:
        x_subset.append(x if is_tensor(x) else torch.as_tensor(x))
        y_subset.append(y if is_tensor(y) else torch.as_tensor(y))

    x_tensor = torch.stack(x_subset) if x_subset else torch.empty(0)
    y_tensor = torch.stack(y_subset) if y_subset else torch.empty(0)
    return x_tensor, y_tensor
