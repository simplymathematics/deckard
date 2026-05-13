"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .defend import DefensePipelineConfig, _DefenseMixin
from ..utils import safe_store

if TYPE_CHECKING:
    pass


class _RegularizerDefenseMixin(_DefenseMixin):
    """Reusable regularizer defense behavior."""

    def __call__(
        self,
        *,
        data,
        defense_type,
        defense_subtype,
        defense_class,
        art_class,
        init_params,
        base_estimator,
        existing_preprocessors,
        existing_postprocessors,
    ):
        raise NotImplementedError(
            "Regularizer defenses are not implemented yet.",
        )


@dataclass(eq=False)
class RegularizerDefenseConfig(_RegularizerDefenseMixin, DefensePipelineConfig):
    """
    Configuration for regularizer-based defenses.
    
    Regularizers improve model robustness by adding constraints during training
    (e.g., adversarial training variants, gradient-based regularization).
    Defends against adversarial attacks at training time.
    """

    pass


# Register regularizer defense config
safe_store(
    group="model",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)

safe_store(
    group="search/models",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)
