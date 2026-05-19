from __future__ import annotations

# Standard library
import logging

# Third-party
from torch import Tensor

# Local / project
from ..types import RuntimeValue

# Logger
logger = logging.getLogger(__name__)


class TorchDataPipelineMixin:
    """PyTorch adapter methods for FrameworkDataPipelineConfig."""

    def build_pipeline(self) -> RuntimeValue:
        """
        Build a preprocessing pipeline.
        """
        create_pipeline = getattr(self, "create_pipeline", None)

        if callable(create_pipeline):
            self.pipeline = create_pipeline()
            return self.pipeline

        raise AttributeError(
            f"{type(self).__name__} must define "
            "create_pipeline() for pipeline compliance.",
        )

    def fit_presample(
        self,
        X: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Pre-sample hook.
        """
        return X, y

    def fit_X(
        self,
        X: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply feature-only transforms.
        """
        transform = getattr(self, "X_transform", None)

        if callable(transform):
            X = transform(X)

        return X, y

    def fit_y(
        self,
        X: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply target-only transforms.
        """
        transform = getattr(self, "y_transform", None)

        if callable(transform):
            y = transform(y)

        return X, y

    def fit_Xy(
        self,
        X: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply joint pipeline transform.
        """
        pipeline = getattr(self, "pipeline", None)

        if callable(pipeline):
            X = pipeline(X)

        return X, y

    def run_pipeline(
        self,
        pipeline: RuntimeValue | None = None,
    ) -> RuntimeValue:
        """
        Execute prepared pipeline.
        """
        if pipeline is None:
            pipeline = getattr(self, "pipeline", None)

        apply_pipeline = getattr(self, "apply_pipeline", None)

        if callable(apply_pipeline):
            return apply_pipeline(pipeline)

        if callable(pipeline):
            return pipeline(self.X)

        return pipeline
