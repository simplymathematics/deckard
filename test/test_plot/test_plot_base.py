from dataclasses import dataclass
from typing import Any, cast

import pytest

from deckard.plot.base import PlotDictConfig, PlotterMixin, PlotTypePlugin


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


def _runtime_plot_config(value: int) -> PlotDictConfig:
    runtime = PlotDictConfig(plots={}, backend="yellowbrick")
    object.__setattr__(runtime, "value", value)
    return runtime


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


def test_plot_type_plugin_resolve_mixins_and_handler_for_matching_backend():
    runtime = _runtime_plot_config(7)
    plugin = PlotTypePlugin(
        mixin_type=_DummyMixin,
        backend="yellowbrick",
        plot_family="classifier",
    )

    mixins = plugin.resolve_plotter_mixins(
        runtime,
        backend="yellowbrick",
        plot_family="classifier",
        default_mixins=(),
    )
    handler = plugin.resolve_plotter_handler(
        runtime,
        backend="yellowbrick",
        plot_family="classifier",
        default_handler=None,
        default_mixins=(),
    )

    assert mixins == (_DummyMixin,)
    assert callable(handler)
    result = handler("x", y=2)
    result_map = cast(dict, result)
    assert result_map["args"] == ("x",)
    assert result_map["kwargs"] == {"y": 2}
    assert result_map["runtime_value"] == 7


def test_plot_type_plugin_rejects_non_matching_backend_or_excluded_family():
    runtime = PlotDictConfig(plots={}, backend="seaborn")
    plugin = PlotTypePlugin(
        mixin_type=_DummyMixin,
        backend="yellowbrick",
        excluded_families=("cluster",),
    )

    assert (
        plugin.resolve_plotter_mixins(
            runtime,
            backend="seaborn",
            plot_family="classifier",
            default_mixins=(),
        )
        == ()
    )
    assert (
        plugin.resolve_plotter_handler(
            runtime,
            backend="yellowbrick",
            plot_family="cluster",
            default_handler=None,
            default_mixins=(),
        )
        is None
    )


def test_plot_type_plugin_supports_string_mixin_resolution():
    plugin = PlotTypePlugin(
        mixin_type="deckard.plot.base.PlotterMixin",
        backend="yellowbrick",
    )
    resolved = plugin._resolve_mixin_type()

    assert resolved is PlotterMixin


def test_plot_type_plugin_is_case_insensitive_for_backend_and_family():
    plugin = PlotTypePlugin(
        mixin_type=_DummyMixin,
        backend="YellowBrick",
        plot_family="Classifier",
    )
    runtime = _runtime_plot_config(2)

    mixins = plugin.resolve_plotter_mixins(
        runtime,
        backend="yellowbrick",
        plot_family="classifier",
        default_mixins=(),
    )

    assert mixins == (_DummyMixin,)


def test_plot_type_plugin_call_delegates_to_runtime_mixin():
    plugin = PlotTypePlugin(mixin_type=_DummyMixin, backend="yellowbrick")
    runtime = _runtime_plot_config(99)

    result = plugin(runtime, "a", enabled=True)
    result_map = cast(dict, result)

    assert result_map["args"] == ("a",)
    assert result_map["kwargs"] == {"enabled": True}
    assert result_map["runtime_value"] == 99


def test_plot_dict_config_iter_len_and_merge_behavior():
    cfg = PlotDictConfig(
        plots={"one": {"plot_type": "roc_auc"}}, backend="yellowbrick"
    )
    other = PlotDictConfig(
        plots={"two": {"plot_type": "pr_curve"}}, backend="yellowbrick"
    )

    assert len(cfg) == 1
    assert list(dict(cfg).keys()) == ["one"]

    merged = cfg.merge(other)

    assert merged is cfg
    assert len(cfg) == 2
    assert set(cfg.plots.keys()) == {"one", "two"}


def test_plot_dict_config_merge_ignores_non_plotdict_values():
    cfg = PlotDictConfig(
        plots={"one": {"plot_type": "roc_auc"}}, backend="yellowbrick"
    )

    merged = cfg.merge(cast(Any, {"two": {"plot_type": "pr_curve"}}))

    assert merged is cfg
    assert set(cfg.plots.keys()) == {"one"}
