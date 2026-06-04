from dataclasses import dataclass
from typing import Any, cast

import pytest

from deckard.plot.base import PlotDictConfig, PlotterMixin


@dataclass
class _Runtime:
    value: int = 0


class _DummyMixin(PlotterMixin):
    def __call__(self, *args, **kwargs):
        return {
            "args": args,
            "kwargs": kwargs,
            "runtime_value": self.runtime.value,
        }


def test_plotter_mixin_base_call_raises_not_implemented():
    mixin = PlotterMixin(runtime=_Runtime())
    with pytest.raises(NotImplementedError, match="must implement"):
        mixin()


def test_plotter_mixin_attribute_forwarding_to_runtime():
    runtime = _Runtime(value=10)
    mixin = _DummyMixin(runtime=runtime)

    assert mixin.value == 10
    mixin.value = 12
    assert runtime.value == 12


def test_plotter_mixin_assignment_before_runtime_binding_sets_local_attr():
    mixin = _DummyMixin(runtime=None)
    mixin.local_only = "x"

    assert mixin.local_only == "x"


def test_plot_dict_config_iter_len_and_merge_behavior():
    cfg = PlotDictConfig(
        plots={"one": {"plot_type": "roc_auc"}},
        backend="yellowbrick",
    )
    other = PlotDictConfig(
        plots={"two": {"plot_type": "pr_curve"}},
        backend="yellowbrick",
    )

    assert len(cfg) == 1
    assert list(dict(cfg).keys()) == ["one"]

    merged = cfg.merge(other)

    assert merged is cfg
    assert len(cfg) == 2
    assert set(cfg.plots.keys()) == {"one", "two"}


def test_plot_dict_config_merge_ignores_non_plotdict_values():
    cfg = PlotDictConfig(
        plots={"one": {"plot_type": "roc_auc"}},
        backend="yellowbrick",
    )

    merged = cfg.merge(cast(Any, {"two": {"plot_type": "pr_curve"}}))

    assert merged is cfg
    assert set(cfg.plots.keys()) == {"one"}


def test_plot_dict_config_normalizes_backend_alias_through_canon() -> None:
    cfg = PlotDictConfig(plots={}, backend="sns")
    assert cfg.backend == "seaborn"
