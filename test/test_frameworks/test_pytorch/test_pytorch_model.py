import unittest
import tempfile
import shutil
from pathlib import Path
import pytest
from torch.utils.data import Dataset
from unittest.mock import patch


torch = pytest.importorskip("torch")
Tensor = pytest.importorskip("torch").Tensor
PytorchDataConfig = pytest.importorskip(
    "deckard.pytorch.data",
).PytorchDataConfig
PytorchCustomDataConfig = pytest.importorskip(
    "deckard.pytorch.data",
).PytorchCustomDataConfig

# TODO Canonical ModelConfig tests for dataset/tensor/Dataloader classification and regression