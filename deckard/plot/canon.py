"""Canonical runtime contract helpers for plot orchestration."""

from __future__ import annotations

from typing import Any, TypedDict

CANON_PLOT_BACKENDS: frozenset[str] = frozenset({"seaborn", "yellowbrick"})


class PlotRuntimeContract(TypedDict, total=False):
    """Canonical runtime payload for plot orchestration."""

    files: dict[str, str | None]
    times: dict[str, float]
    plot_state: dict[str, Any]
    backend: str


def normalize_plot_backend(backend: str | None) -> str:
    """Normalize backend token to canonical backend names.

    Accepted aliases:
    - ``None`` / empty -> ``seaborn``
    - ``yellow`` / ``yb`` -> ``yellowbrick``
    - ``sns`` -> ``seaborn``
    """

    if backend is None:
        return "seaborn"
    token = str(backend).strip().lower()
    if token == "":
        return "seaborn"
    if token in {"yellow", "yb"}:
        token = "yellowbrick"
    elif token == "sns":
        token = "seaborn"
    if token not in CANON_PLOT_BACKENDS:
        raise KeyError(f"Unsupported plot backend: {backend}")
    return token
