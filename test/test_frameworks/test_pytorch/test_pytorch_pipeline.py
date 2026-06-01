from __future__ import annotations

import pytest

from deckard.frameworks.pytorch.pipeline import TorchDataPipelineMixin


class _PipelineHarness(TorchDataPipelineMixin):
    def __init__(self) -> None:
        self.X = "payload"
        self.pipeline = None

    def create_pipeline(self):
        return lambda value: f"pipe:{value}"

    def X_transform(self, x):
        return f"X:{x}"

    def y_transform(self, y):
        return f"Y:{y}"

    def apply_pipeline(self, pipeline):
        return ("applied", pipeline)


def test_build_pipeline_assigns_created_pipeline() -> None:
    harness = _PipelineHarness()

    pipeline = harness.build_pipeline()

    assert callable(pipeline)
    assert harness.pipeline is pipeline


def test_build_pipeline_requires_create_pipeline() -> None:
    class _NoCreate(TorchDataPipelineMixin):
        pass

    with pytest.raises(AttributeError, match="must define create_pipeline"):
        _NoCreate().build_pipeline()


def test_fit_stage_transforms() -> None:
    harness = _PipelineHarness()

    x1, y1 = harness.fit_presample("x", "y")
    x2, y2 = harness.fit_X("x", "y")
    x3, y3 = harness.fit_y("x", "y")
    harness.pipeline = lambda value: f"joint:{value}"
    x4, y4 = harness.fit_Xy("x", "y")

    assert (x1, y1) == ("x", "y")
    assert (x2, y2) == ("X:x", "y")
    assert (x3, y3) == ("x", "Y:y")
    assert (x4, y4) == ("joint:x", "y")


def test_run_pipeline_prefers_apply_pipeline() -> None:
    harness = _PipelineHarness()
    harness.pipeline = lambda value: f"should_not_run:{value}"

    result = harness.run_pipeline()

    assert result[0] == "applied"


def test_run_pipeline_falls_back_to_callable_pipeline() -> None:
    class _NoApply(TorchDataPipelineMixin):
        def __init__(self) -> None:
            self.X = "x"
            self.pipeline = lambda value: f"pipe:{value}"

    result = _NoApply().run_pipeline()

    assert result == "pipe:x"


def test_run_pipeline_returns_literal_pipeline() -> None:
    class _Literal(TorchDataPipelineMixin):
        def __init__(self) -> None:
            self.X = "x"
            self.pipeline = {"kind": "literal"}

    result = _Literal().run_pipeline()

    assert result == {"kind": "literal"}
