"""Power scoring utilities for canonical *_score pipeline hooks."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(eq=False, kw_only=True)
class DVCPowerMixin:
    """
    Adds power scoring for canonical experiment score stages.

    Automatically derives CPU/GPU power limits from the host OS when possible.

    Intended hook usage:

        after_data_score      -> namespace="data"
        after_model_score     -> namespace="model"
        after_attack_score    -> namespace="attack"
        after_detector_score  -> namespace="detector"
    """

    cpu_tdp_watts: float = field(default_factory=lambda: DVCPowerMixin._detect_cpu_tdp())
    gpu_tdp_watts: float = field(default_factory=lambda: DVCPowerMixin._detect_gpu_tdp())

    _power_energy_wh: float = 0.0
    _power_last_ts: float | None = None

    def _log_power_score(
        self,
        *,
        namespace: str,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Hook target for HookPlugin(method_name="_log_power_score").

        Returns namespaced power metrics.
        """
        cpu_watts = self._estimate_cpu_power()
        gpu_watts = self._estimate_gpu_power()

        total_watts = cpu_watts + gpu_watts
        energy_wh = self._update_energy(total_watts)

        return {
            f"power/{namespace}/cpu_watts": cpu_watts,
            f"power/{namespace}/gpu_watts": gpu_watts,
            f"power/{namespace}/total_watts": total_watts,
            f"power/{namespace}/energy_wh": energy_wh,
        }

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def _estimate_cpu_power(self) -> float:
        """
        Estimate CPU power from available utilization.

        Expected optional runtime attrs:
            self.cpu_percent
            self.system_cpu
        """
        cpu_percent = self._first_float(
            getattr(self, "cpu_percent", None),
            getattr(self, "system_cpu", None),
        )

        if cpu_percent is None:
            return 0.0

        cpu_percent = max(0.0, min(100.0, cpu_percent))
        return (cpu_percent / 100.0) * self.cpu_tdp_watts

    @staticmethod
    def _detect_cpu_tdp() -> float:
        """
        Detect CPU package max power (watts).

        Linux:
            intel-rapl constraint_0_power_limit_uw
            AMD hwmon power*_cap

        Raises:
            RuntimeError if automatic detection fails.
        """
        logger.info("Attempting automatic CPU power limit detection.")

        # Intel RAPL
        rapl = Path(
            "/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw"
        )
        if rapl.exists():
            try:
                microwatts = int(rapl.read_text().strip())
                watts = microwatts / 1_000_000
                logger.info(
                    "Detected CPU power limit via Intel RAPL: %.2f W",
                    watts,
                )
                return watts
            except Exception:
                logger.exception(
                    "Failed reading Intel RAPL CPU power limit."
                )

        # AMD hwmon
        hwmon_root = Path("/sys/class/hwmon")
        if hwmon_root.exists():
            for cap in hwmon_root.glob("hwmon*/power*_cap"):
                try:
                    microwatts = int(cap.read_text().strip())
                    watts = microwatts / 1_000_000
                    logger.info(
                        "Detected CPU power limit via hwmon: %.2f W",
                        watts,
                    )
                    return watts
                except Exception:
                    logger.exception(
                        "Failed reading AMD hwmon CPU power limit."
                    )

        raise RuntimeError(
            "Automatic CPU power limit detection failed. "
            "Please specify 'cpu_tdp_watts' manually "
            "(total CPU package power consumption in watts)."
        )

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

    def _estimate_gpu_power(self) -> float:
        """
        Prefer actual NVIDIA telemetry.

        Fallback to utilization × configured GPU TDP.

        Expected optional runtime attrs:
            self.gpu_utilization
            self.system_gpu_utilization
        """
        measured = self._read_nvidia_gpu_power()
        if measured is not None:
            return measured

        gpu_percent = self._first_float(
            getattr(self, "gpu_utilization", None),
            getattr(self, "system_gpu_utilization", None),
        )

        if gpu_percent is None:
            return 0.0

        gpu_percent = max(0.0, min(100.0, gpu_percent))
        return (gpu_percent / 100.0) * self.gpu_tdp_watts

    @staticmethod
    def _detect_gpu_tdp() -> float:
        """
        Detect GPU power limit.

        NVIDIA:
            nvidia-smi --query-gpu=power.limit

        AMD:
            rocm-smi --showmaxpower

        Raises:
            RuntimeError if automatic detection fails.
        """
        logger.info("Attempting automatic GPU power limit detection.")

        # NVIDIA
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=power.limit",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            first = result.strip().splitlines()[0]
            watts = float(first)
            logger.info(
                "Detected GPU power limit via nvidia-smi: %.2f W",
                watts,
            )
            return watts

        except FileNotFoundError:
            logger.warning(
                "nvidia-smi not available; cannot automatically detect "
                "NVIDIA GPU power limit."
            )

        except Exception:
            logger.exception(
                "Failed reading GPU power limit from nvidia-smi."
            )

        # AMD ROCm
        try:
            result = subprocess.check_output(
                ["rocm-smi", "--showmaxpower"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in result.splitlines():
                if "W" in line:
                    tokens = line.split()
                    for token in tokens:
                        try:
                            watts = float(token)
                            logger.info(
                                "Detected GPU power limit via rocm-smi: %.2f W",
                                watts,
                            )
                            return watts
                        except ValueError:
                            continue

        except FileNotFoundError:
            logger.warning(
                "rocm-smi not available; cannot automatically detect "
                "AMD GPU power limit."
            )

        except Exception:
            logger.exception(
                "Failed reading GPU power limit from rocm-smi."
            )

        raise RuntimeError(
            "Automatic GPU power limit detection failed. "
            "Please specify 'gpu_tdp_watts' manually "
            "(total GPU board power consumption in watts)."
        )

    def _read_nvidia_gpu_power(self) -> float | None:
        """
        Read actual instantaneous GPU power from nvidia-smi.
        """
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return float(result.strip().splitlines()[0])

        except FileNotFoundError:
            logger.warning(
                "nvidia-smi not available; GPU power draw "
                "will be estimated from utilization."
            )
            return None

        except Exception:
            logger.exception(
                "Failed reading instantaneous GPU power from nvidia-smi."
            )
            return None

    # ------------------------------------------------------------------
    # Energy integration
    # ------------------------------------------------------------------

    def _update_energy(self, watts: float) -> float:
        """
        Integrate cumulative energy:

            Wh += watts * dt / 3600
        """
        now = time.time()

        if self._power_last_ts is None:
            self._power_last_ts = now
            return self._power_energy_wh

        dt_seconds = now - self._power_last_ts
        self._power_last_ts = now

        self._power_energy_wh += (watts * dt_seconds) / 3600.0
        return self._power_energy_wh

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_float(*values: Any) -> float | None:
        """
        Return first successfully cast float.
        """
        for value in values:
            try:
                if value is not None:
                    return float(value)
            except Exception:
                continue
        return None


__all__ = ["DVCPowerMixin"]