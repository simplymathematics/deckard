from dataclasses import dataclass, field
from typing import Union


@dataclass
class AttackConfig:
    name: str = "art.attacks.evasion.FastGradientMethod"
    init: dict = field(default_factory=dict)
    generate: dict = field(default_factory=dict)
    filename: Union[str, None] = None
    path: Union[str, None] = None
    filetype: Union[str, None] = None
