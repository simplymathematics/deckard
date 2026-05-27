from __future__ import annotations

# Standard library
import logging

# Third-party
from torch import Tensor

# Local / project
from ..types import EstimatorLike

# Logger
logger = logging.getLogger(__name__)


class TorchDataPipelineMixin:
    """PyTorch adapter methods for framework DataConfig runtimes.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def build_pipeline(self) -> EstimatorLike | Tensor:
        """Build a preprocessing pipeline.

        Returns:
            Runtime pipeline object.

        Raises:
            AttributeError: If runtime does not expose ``create_pipeline``.
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
        """Run pre-sample hook.

        Args:
            X: Feature tensor.
            y: Target tensor.

        Returns:
            Feature and target tensors.
        """
        return X, y

    def fit_X(
        self,
        X: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply feature-only transforms.

        Args:
            X: Feature tensor.
            y: Target tensor.

        Returns:
            Transformed feature tensor and original target tensor.
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
        """Apply target-only transforms.

        Args:
            X: Feature tensor.
            y: Target tensor.

        Returns:
            Original feature tensor and transformed target tensor.
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
        """Apply joint pipeline transform.

        Args:
            X: Feature tensor.
            y: Target tensor.

        Returns:
            Transformed feature tensor and target tensor.
        """
        pipeline = getattr(self, "pipeline", None)

        if callable(pipeline):
            X = pipeline(X)

        return X, y

    def run_pipeline(
        self,
        pipeline: EstimatorLike | Tensor | None = None,
    ) -> EstimatorLike | Tensor:
        """Execute prepared pipeline.

        Args:
            pipeline: Optional pipeline override.

        Returns:
            Pipeline output payload.
        """
        if pipeline is None:
            pipeline = getattr(self, "pipeline", None)

        apply_pipeline = getattr(self, "apply_pipeline", None)

        if callable(apply_pipeline):
            return apply_pipeline(pipeline)

        if callable(pipeline):
            return pipeline(self.X)

        return pipeline
