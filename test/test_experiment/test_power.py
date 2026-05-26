from __future__ import annotations

import math
import subprocess

from deckard.experiment.power import DVCPowerMixin


def test_detect_gpu_tdp_falls_back_to_zero_when_telemetry_unavailable(monkeypatch):
    def _missing_tool(*args, **kwargs):
        raise FileNotFoundError("tool not found")

    monkeypatch.setattr(subprocess, "check_output", _missing_tool)

    assert math.isnan(DVCPowerMixin._detect_gpu_tdp())


def test_estimate_gpu_power_uses_nan_tdp_gracefully(monkeypatch):
    mixin = DVCPowerMixin(cpu_tdp_watts=100.0, gpu_tdp_watts=math.nan)
    mixin.gpu_utilization = 75.0

    monkeypatch.setattr(mixin, "_read_nvidia_gpu_power", lambda: None)

    assert math.isnan(mixin._estimate_gpu_power())
