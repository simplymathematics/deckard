from dataclasses import dataclass, field


@dataclass
class ScorerConfig:
    name: str = "sklearn.metrics.accuracy_score"
    params: dict = field(default_factory=dict)
    alias: str = "accuracy"


@dataclass
class ScorerDictConfig:
    scorers: dict = field(default_factory=dict)
