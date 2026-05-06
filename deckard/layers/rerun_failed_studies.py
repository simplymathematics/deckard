import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

import optuna
import yaml

logger = logging.getLogger(__name__)


def _load_meta_schema(meta_schema: str) -> dict[str, Any]:
    schema_path = Path(meta_schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Meta schema file not found: {schema_path}")
    with open(schema_path, "r") as f:
        payload = yaml.safe_load(f) or {}
    schema = payload.get("schema", payload)
    if not isinstance(schema, dict):
        raise ValueError(
            f"Schema in {schema_path} must be a dictionary, got {type(schema)}",
        )
    return schema


def _parse_study_metadata(study_name: str, schema: dict[str, Any]) -> dict[str, str]:
    sep = str(schema.get("sep", "_"))
    parts = study_name.split(sep)
    parsed: dict[str, str] = {}
    for key, locator in schema.items():
        if key == "sep":
            continue
        if isinstance(locator, int):
            parsed[key] = parts[locator] if locator < len(parts) else ""
        elif isinstance(locator, str) and ":" in locator:
            start, end = map(int, locator.split(":", 1))
            end = min(end, len(parts) - 1)
            parsed[key] = sep.join(parts[start : end + 1]) if start <= end else ""
        else:
            raise ValueError(
                f"Unsupported schema locator for key '{key}': {locator}",
            )
    return parsed


def _collect_failed_studies(
    storage: str,
    include_running: bool = False,
) -> list[str]:
    """Return study names where no trials completed successfully.

    A study is considered failed when it has at least one FAIL trial and no
    COMPLETE trials. RUNNING studies are skipped by default.
    """
    failed: list[str] = []
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        study_name = getattr(summary, "study_name", None) or getattr(summary, "name", None)
        if not study_name:
            continue
        study = optuna.study.load_study(storage=storage, study_name=study_name)
        states = [str(getattr(t, "state", "")) for t in study.get_trials(deepcopy=False)]
        if not states:
            continue
        has_fail = any("FAIL" in s for s in states)
        has_complete = any("COMPLETE" in s for s in states)
        has_running = any("RUNNING" in s for s in states)
        if has_fail and not has_complete:
            if has_running and not include_running:
                continue
            failed.append(study_name)
    return failed


def _build_rerun_command_for_study(
    study_name: str,
    *,
    schema: dict[str, Any],
) -> str | None:
    """Build a deckard optimize rerun command from parsed study metadata.

    The schema parser extracts fields from the study name. Recognized keys are
    translated to Hydra overrides when present: ``data``, ``model``, ``attack``,
    and ``defense``.
    """
    parsed = _parse_study_metadata(study_name, schema)
    overrides: list[str] = []
    for key in ["data", "model", "attack", "defense"]:
        value = parsed.get(key, "")
        if isinstance(value, str) and value.strip() != "":
            overrides.append(f"{key}={value.strip()}")
    if len(overrides) == 0:
        return None
    return "deckard optimize " + " ".join(shlex.quote(item) for item in overrides)


def rerun_failed_studies_main(
    optuna_db: str = "sqlite:///optuna.db",
    working_dir: str = ".",
    meta_schema: str = "config/meta.yaml",
    include_running: bool = False,
    execute: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Plan or execute reruns for failed Optuna studies.

    Parameters
    ----------
    optuna_db : str
        Optuna storage URI, e.g. ``sqlite:///optuna.db``.
    working_dir : str
        Directory where deckard optimize commands should run.
    meta_schema : str
        Path to schema YAML used to parse study names (e.g. config/meta.yaml).
    include_running : bool
        Include studies that currently have RUNNING trials.
    execute : bool
        If true, execute generated commands. Otherwise dry-run only.
    limit : int | None
        Optional maximum number of failed studies to process.
    """
    run_dir = Path(working_dir).resolve()
    schema = _load_meta_schema(meta_schema)
    failed = _collect_failed_studies(
        storage=optuna_db,
        include_running=include_running,
    )
    if limit is not None and limit > 0:
        failed = failed[:limit]

    planned: list[dict[str, str]] = []
    skipped: list[str] = []

    for study_name in failed:
        cmd = _build_rerun_command_for_study(study_name, schema=schema)
        if cmd is None:
            skipped.append(study_name)
            continue
        planned.append({"study_name": study_name, "command": cmd})

    results: list[dict[str, Any]] = []
    if execute:
        for item in planned:
            logger.info("Executing rerun for study %s", item["study_name"])
            proc = subprocess.run(
                ["bash", "-lc", item["command"]],
                cwd=run_dir.as_posix(),
                capture_output=True,
                text=True,
            )
            results.append(
                {
                    "study_name": item["study_name"],
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )
    else:
        for item in planned:
            logger.info("[dry-run] %s", item["command"])

    return {
        "failed_studies": failed,
        "planned": planned,
        "skipped": skipped,
        "executed": execute,
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        conflict_handler="resolve",
        description="Plan or rerun failed Optuna studies for deckard multiruns.",
    )
    parser.add_argument("--optuna_db", default="sqlite:///optuna.db")
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--meta_schema", default="config/meta.yaml")
    parser.add_argument("--include_running", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


rerun_failed_studies_parser = _build_parser()


if __name__ == "__main__":
    args = rerun_failed_studies_parser.parse_args()
    rerun_failed_studies_main(**vars(args))
