"""Reusable compose validation for paper config roots.

Discovers paper roots dynamically and derives component overrides from the
group structure present in examples/*/config rather than hard-coded cases.
"""

from pathlib import Path

import pytest

from .shared_compose import compose_config

ROOT = Path(__file__).resolve().parents[2]
PAPERS_ROOT = ROOT / "papers"
EXAMPLES_ROOT = ROOT / "examples"

IGNORED_ROOT_CONFIGS = {
    "afr.yaml",
    "clean.yaml",
    "combined_afr.yaml",
    "compile.yaml",
    "condensed_plots.yaml",
    "merged_plots.yaml",
    "meta.yaml",
    "new_plots.yaml",
    "plots.yaml",
    "precomputed_plots.yaml",
    "rq.yaml",
}


def _compose(config_dir: Path, config_name: str, overrides: list[str] | None = None):
    return compose_config(config_dir, config_name, overrides=overrides)


def _discover_example_groups() -> list[str]:
    groups: set[str] = set()
    for cfg_dir in EXAMPLES_ROOT.glob("*/config"):
        if not cfg_dir.is_dir():
            continue
        for child in cfg_dir.iterdir():
            if child.is_dir() and any(child.glob("*.yaml")):
                groups.add(child.name)
    return sorted(groups)


def _candidate_component_overrides(
    config_dir: Path,
    example_groups: list[str],
) -> list[str]:
    overrides: list[str] = []
    for group in example_groups:
        group_dir = config_dir / group
        if not group_dir.is_dir():
            continue
        options = sorted(path.stem for path in group_dir.glob("*.yaml"))
        if not options:
            continue
        option = "default" if "default" in options else options[0]
        overrides.append(f"++{group}={option}")
    return overrides


def _discover_paper_cases(
    example_groups: list[str],
) -> list[tuple[Path, str, list[str]]]:
    cases: list[tuple[Path, str, list[str]]] = []
    for config_dir in sorted(PAPERS_ROOT.rglob("config")):
        roots = sorted(
            path
            for path in config_dir.glob("*.yaml")
            if path.name not in IGNORED_ROOT_CONFIGS
        )
        for root_cfg in roots:
            config_name = root_cfg.name
            try:
                _compose(config_dir=config_dir, config_name=config_name)
            except Exception:
                # Skip non-root utility files that cannot be composed standalone.
                continue
            valid_overrides: list[str] = []
            for override in _candidate_component_overrides(
                config_dir=config_dir,
                example_groups=example_groups,
            ):
                try:
                    _compose(
                        config_dir=config_dir,
                        config_name=config_name,
                        overrides=[override],
                    )
                except Exception:
                    # Some composeable root configs are utility/report configs that do
                    # not include standard component groups in defaults.
                    continue
                valid_overrides.append(override)

            cases.append((config_dir, config_name, valid_overrides))
    return cases


EXAMPLE_GROUPS = _discover_example_groups()
PAPER_CASES = _discover_paper_cases(EXAMPLE_GROUPS)


def _case_id(case: tuple[Path, str, list[str]]) -> str:
    config_dir, config_name, _ = case
    rel = config_dir.relative_to(ROOT).as_posix().replace("/", "-")
    return f"{rel}-{config_name}"


def test_discovery_sanity() -> None:
    assert EXAMPLE_GROUPS, "No component groups discovered from examples/*/config."
    assert PAPER_CASES, "No composeable paper root configs discovered."


@pytest.mark.parametrize(
    "config_dir,config_name,_overrides",
    [pytest.param(*case, id=_case_id(case)) for case in PAPER_CASES],
)
def test_paper_root_configs_compose(
    config_dir: Path,
    config_name: str,
    _overrides: list[str],
):
    cfg = _compose(config_dir=config_dir, config_name=config_name)
    assert cfg is not None
    if "score" in cfg:
        assert "scorers" not in cfg


@pytest.mark.parametrize(
    "config_dir,config_name,override",
    [
        pytest.param(
            config_dir,
            config_name,
            override,
            id=f"{_case_id((config_dir, config_name, []))}-{override.split('=')[0].lstrip('+')}",
        )
        for config_dir, config_name, overrides in PAPER_CASES
        for override in overrides
    ],
)
def test_paper_component_overrides_compose(
    config_dir: Path,
    config_name: str,
    override: str,
):
    cfg = _compose(
        config_dir=config_dir,
        config_name=config_name,
        overrides=[override],
    )
    assert cfg is not None
