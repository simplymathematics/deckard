import logging
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import optuna
import yaml
from tqdm.auto import tqdm

from ..utils import create_parser_from_function

logger = logging.getLogger(__name__)


def _load_dvc_config(dvc_file: str) -> dict:
    dvc_path = Path(dvc_file)
    assert dvc_path.exists(), f"File {dvc_path} does not exist."

    with dvc_path.open("r") as f:
        return yaml.safe_load(f) or {}


def calculate_number_of_studies_per_stage(stage: str, dvc_file: str) -> int:
    """Return the number of studies expected for a DVC stage definition."""
    dvc_conf = _load_dvc_config(dvc_file)
    dvc_path = Path(dvc_file)

    stage_conf = dvc_conf.get("stages", {}).get(stage)
    if stage_conf is None:
        raise KeyError(f"Stage '{stage}' was not found in {dvc_path}.")

    return _count_studies_for_stage_conf(stage_conf)


def _count_studies_for_stage_conf(stage_conf: dict) -> int:
    if "matrix" in stage_conf:
        count = 1
        for values in stage_conf["matrix"].values():
            count *= len(values)
        return count

    if "foreach" in stage_conf:
        return len(stage_conf["foreach"])

    return 1


def _infer_stages_from_dvc(dvc_file: str) -> list:
    """Infer tracked stages from dvc.yaml by selecting stages that use --multirun."""
    dvc_conf = _load_dvc_config(dvc_file)
    stages_conf = dvc_conf.get("stages", {})
    inferred = []

    for stage_name, stage_conf in stages_conf.items():
        cmd = stage_conf.get("cmd", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(x) for x in cmd)
        cmd = str(cmd)
        if "--multirun" in cmd:
            inferred.append(stage_name)

    if len(inferred) == 0:
        raise ValueError(
            f"No multirun stages found in {dvc_file}. Pass stages explicitly via --stages."
        )

    return inferred


def _extract_stage_config_name(stage_conf: dict) -> Optional[str]:
    """Extract --config-name value from stage cmd if present."""
    cmd = stage_conf.get("cmd", "")
    if isinstance(cmd, list):
        cmd = " ".join(str(x) for x in cmd)
    cmd = str(cmd).strip()
    if not cmd:
        return None

    tokens = shlex.split(cmd)
    for i, token in enumerate(tokens):
        if token == "--config-name" and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith("--config-name="):
            return token.split("=", 1)[1]

    return None


def _resolve_hydra_config_for_stage(stage_conf: dict, hydra_cfg_file: str) -> str:
    """Resolve per-stage Hydra config file using --config-name (extension optional)."""
    default_cfg = Path(hydra_cfg_file)
    default_cfg_dir = default_cfg.parent

    config_name = _extract_stage_config_name(stage_conf)
    if config_name is None:
        cfg_path = default_cfg
    else:
        candidate = Path(config_name)
        if candidate.is_absolute():
            cfg_path = candidate
        else:
            cfg_path = default_cfg_dir / candidate

        if cfg_path.suffix == "":
            yaml_cfg = cfg_path.with_suffix(".yaml")
            yml_cfg = cfg_path.with_suffix(".yml")
            if yaml_cfg.exists():
                cfg_path = yaml_cfg
            elif yml_cfg.exists():
                cfg_path = yml_cfg
            else:
                raise FileNotFoundError(
                    f"Could not resolve config '{config_name}' to .yaml/.yml near {default_cfg_dir}."
                )

    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config file not found: {cfg_path}")

    return str(cfg_path)


def _collect_storage_finished_counts(optuna_db: str, end_states: set) -> tuple:
    """Return (finished_counts_per_study, earliest_start_time) for one Optuna storage."""
    finished_counts = []
    earliest = None

    summaries = optuna.study.get_all_study_summaries(storage=optuna_db)
    for summary in summaries:
        study_name = getattr(summary, "study_name", None) or getattr(
            summary, "name", None
        )
        if not study_name:
            continue

        study = optuna.study.load_study(storage=optuna_db, study_name=study_name)
        study_df = study.trials_dataframe()

        if "state" in study_df.columns:
            finished_counts.append(int(study_df["state"].isin(end_states).sum()))

        for trial in study.get_trials(deepcopy=False):
            dt = getattr(trial, "datetime_start", None)
            if dt is None:
                continue
            if earliest is None or dt < earliest:
                earliest = dt

    return finished_counts, earliest


def _count_completed_studies(
    observed_finished_counts: list, required_trials: list
) -> int:
    """Greedy max matching between observed finished-trial counts and required-trial thresholds."""
    obs = sorted(observed_finished_counts, reverse=True)
    req = sorted(required_trials, reverse=True)
    i, j, matched = 0, 0, 0

    while i < len(obs) and j < len(req):
        if obs[i] >= req[j]:
            matched += 1
            i += 1
            j += 1
        else:
            j += 1

    return matched


def _parse_csv_values(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_hydra_sweeper_config(hydra_cfg_file: str) -> tuple:
    cfg_path = Path(hydra_cfg_file)
    assert cfg_path.exists(), f"Missing Hydra config: {cfg_path}"

    with cfg_path.open("r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    hydra_section = raw_cfg.get("hydra", raw_cfg)
    sweeper = hydra_section.get("sweeper", {})
    storage = sweeper.get("storage")
    n_trials = int(sweeper.get("n_trials", 100))

    if storage is None:
        raise ValueError(f"No hydra.sweeper.storage found in {cfg_path}.")

    return storage, n_trials


def progress_bar_main(
    hydra_cfg_file: str = "config/default.yaml",
    dvc_file: str = "dvc.yaml",
    stages: str = "",
    poll_interval: float = 5.0,
    complete_states: str = "COMPLETE,FAILED,PRUNED",
) -> dict:
    """Track Optuna progress for expected studies/trials and render tqdm progress bars."""
    dvc_conf = _load_dvc_config(dvc_file)
    stages_conf = dvc_conf.get("stages", {})

    if stages is None or str(stages).strip() == "":
        optuna_stages = _infer_stages_from_dvc(dvc_file)
    else:
        optuna_stages = _parse_csv_values(stages)
    end_states = set(_parse_csv_values(complete_states))

    stage_specs = []
    storage_requirements = {}
    for stage in optuna_stages:
        stage_conf = stages_conf.get(stage)
        if stage_conf is None:
            raise KeyError(f"Stage '{stage}' was not found in {dvc_file}.")

        studies_count = _count_studies_for_stage_conf(stage_conf)
        stage_cfg = _resolve_hydra_config_for_stage(stage_conf, hydra_cfg_file)
        storage, n_trials = _get_hydra_sweeper_config(stage_cfg)

        stage_specs.append(
            {
                "stage": stage,
                "studies": studies_count,
                "config": stage_cfg,
                "storage": storage,
                "n_trials": n_trials,
            }
        )

        if storage not in storage_requirements:
            storage_requirements[storage] = {
                "expected_studies": 0,
                "expected_trials": 0,
                "required_trials": [],
            }

        storage_requirements[storage]["expected_studies"] += studies_count
        storage_requirements[storage]["expected_trials"] += studies_count * n_trials
        storage_requirements[storage]["required_trials"].extend(
            [n_trials] * studies_count
        )

    expected_studies = sum(
        item["expected_studies"] for item in storage_requirements.values()
    )
    expected_total_trials = sum(
        item["expected_trials"] for item in storage_requirements.values()
    )

    logger.info("Monitoring %d stages from %s", len(stage_specs), dvc_file)
    for spec in stage_specs:
        logger.info(
            "stage=%s config=%s storage=%s studies=%d n_trials=%d",
            spec["stage"],
            spec["config"],
            spec["storage"],
            spec["studies"],
            spec["n_trials"],
        )

    def _aggregate_progress() -> tuple:
        total_completed_studies = 0
        total_completed_trials = 0
        earliest = None

        for storage, req in storage_requirements.items():
            finished_counts, storage_earliest = _collect_storage_finished_counts(
                storage, end_states
            )
            storage_completed_studies = _count_completed_studies(
                observed_finished_counts=finished_counts,
                required_trials=req["required_trials"],
            )
            storage_completed_trials = min(sum(finished_counts), req["expected_trials"])

            total_completed_studies += min(
                storage_completed_studies, req["expected_studies"]
            )
            total_completed_trials += storage_completed_trials

            if storage_earliest is not None and (
                earliest is None or storage_earliest < earliest
            ):
                earliest = storage_earliest

        return total_completed_studies, total_completed_trials, earliest

    initial_complete_studies, initial_complete_trials, db_start_time = (
        _aggregate_progress()
    )

    studies_pbar = tqdm(
        total=expected_studies,
        initial=initial_complete_studies,
        desc="Optuna study progress",
        unit=" studies",
        position=0,
    )
    trials_pbar = tqdm(
        total=expected_total_trials,
        initial=initial_complete_trials,
        desc="Optuna trial progress",
        unit=" trials",
        position=1,
    )

    if db_start_time is not None:
        start_ts = db_start_time.timestamp()
        now_ts = time.time()
        for pbar in (studies_pbar, trials_pbar):
            pbar.start_t = start_ts
            pbar.last_print_t = now_ts

    prev_complete_studies = initial_complete_studies
    prev_complete_trials = initial_complete_trials

    while True:
        new_complete_studies, new_complete_trials, _ = _aggregate_progress()
        new_complete_studies = min(new_complete_studies, expected_studies)
        new_complete_trials = min(new_complete_trials, expected_total_trials)

        studies_pbar.update(new_complete_studies - prev_complete_studies)
        trials_pbar.update(new_complete_trials - prev_complete_trials)
        prev_complete_studies = new_complete_studies
        prev_complete_trials = new_complete_trials

        if (
            new_complete_trials >= expected_total_trials
            or new_complete_studies >= expected_studies
        ):
            break

        time.sleep(poll_interval)

    studies_pbar.close()
    trials_pbar.close()

    return {
        "storages": list(storage_requirements.keys()),
        "start_time": db_start_time.isoformat() if db_start_time is not None else None,
        "expected_studies": expected_studies,
        "expected_trials": expected_total_trials,
        "completed_studies": prev_complete_studies,
        "completed_trials": prev_complete_trials,
    }


progress_bar_parser = create_parser_from_function(progress_bar_main)


if __name__ == "__main__":
    args = progress_bar_parser.parse_args()
    progress_bar_main(**vars(args))
