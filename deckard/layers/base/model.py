from dataclasses import dataclass, field
from typing import Union


@dataclass
class ArtPipelineConfig:
    stages: dict = field(default_factory=dict)


@dataclass
class SklearnPipelineConfig:
    stages: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    library: str = "sklearn"
    name: str = "sklearn.linear_model.SGDClassifier"
    init: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    sklearn_pipeline: dict = field(default_factory=dict)
    art_pipeline: dict = field(default_factory=dict)
    filename: Union[str, None] = None
    path: Union[str, None] = None
    filetype: Union[str, None] = None
