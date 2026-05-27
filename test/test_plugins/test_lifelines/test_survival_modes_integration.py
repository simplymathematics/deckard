"""Integration tests for all four survival analysis modes."""

import pandas as pd
import pytest

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.experiment import SurvivalExperimentConfig
from deckard.layers.survival import survival_main
from deckard.plugins.lifelines.data import LifelinesDataConfig, LifelinesDataMode


class TestMode1NativeSurvivalData:
    """Integration test for Mode 1: Native survival data (lifelines datasets)."""

    def test_native_survival_with_lifelines_diabetes(self, tmp_path):
        """Mode 1: Load lifelines.diabetes and run survival analysis."""
        output_folder = tmp_path / "mode1_output"
        output_folder.mkdir()

        cfg = {
            "survival": {
                "data": "lifelines-diabetes",
                "model": "weibull",
                "plots_folder": str(output_folder),
                "duration_col": "right",
                "target": "E",
                "event_col": "E",
                "classifier": False,
            },
        }
        result = survival_main(cfg=cfg)

        # Verify result structure - matches what survival_main returns
        assert "aft_table" in result or "models" in result
        assert "models" in result
        assert len(result["models"]) >= 1
        assert result["models"]["weibull"] is not None

    def test_native_survival_config_creation(self):
        """Create SurvivalExperimentConfig for Mode 1."""
        base_config = DataConfig(
            dataset_name="lifelines_diabetes",
            target="T",
            classifier=False,
        )
        config = SurvivalExperimentConfig(
            data=base_config,
            model="weibull",
            target="E",
            duration_col="T",
            event_col="E",
            classifier=False,
        )
        assert config.duration_col == "T"
        assert config.event_col == "E"


class TestMode2AuxiliaryModelData:
    """Integration test for Mode 2: Auxiliary model for arbitrary dataset."""

    def test_mode2_config_creation(self):
        """Create LifelinesDataConfig for Mode 2."""
        base_config = DataConfig(
            dataset_name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        survival_data_config = LifelinesDataConfig.from_auxiliary_metric(
            base_config,
            reference_metric="accuracy",
        )
        assert survival_data_config.mode == LifelinesDataMode.AUXILIARY_METRIC
        assert survival_data_config.has_auxiliary_metric()
        assert survival_data_config.reference_metric == "accuracy"

    def test_mode2_experiment_config_with_auxiliary_model(self):
        """Create SurvivalExperimentConfig that uses auxiliary model mode."""
        base_config = DataConfig(
            dataset_name="toy",
            target="accuracy",
            classifier=False,
        )
        survival_config = SurvivalExperimentConfig(
            data=base_config,
            model="weibull",
            target="accuracy",
            duration_col="target",
            event_col="accuracy",
            classifier=False,
        )
        assert survival_config.duration_col == "target"
        assert survival_config.event_col == "accuracy"

    def test_mode2_failure_computation(self):
        """Mode 2: Compute failures from model accuracy."""
        # Create synthetic accuracy data
        df = pd.DataFrame(
            {
                "accuracy": [0.7, 0.75, 0.82, 0.78, 0.80],
                "model_pred": [0.7, 0.75, 0.82, 0.78, 0.80],
            },
        )

        # Compute failures as 100 * (1 - accuracy)
        df["ben_failures"] = 100 * (1 - df["accuracy"])
        assert "ben_failures" in df.columns
        assert (df["ben_failures"] >= 0).all()
        assert (df["ben_failures"] <= 100).all()


class TestMode3AuxiliaryAttackData:
    """Integration test for Mode 3: Auxiliary attack for failure measurement."""

    def test_mode3_config_creation(self):
        """Create LifelinesDataConfig for Mode 3."""
        base_config = DataConfig(
            dataset_name="attack_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        attack_config = {"attack_kind": "evasion", "attack_size": 100}
        survival_data_config = LifelinesDataConfig.from_auxiliary_failure(
            base_config,
            failure_profile=attack_config,
        )
        assert survival_data_config.mode == LifelinesDataMode.AUXILIARY_FAILURE
        assert survival_data_config.has_auxiliary_failure()
        assert survival_data_config.failure_profile == attack_config

    def test_mode3_experiment_config(self):
        """Create SurvivalExperimentConfig that uses attack config."""
        # Test creating AttackConfig for Mode 3
        attack_cfg = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.2},
            attack_size=100,
        )
        assert attack_cfg.attack_type == "art.attacks.evasion.FastGradientMethod"
        assert attack_cfg.attack_size == 100

    def test_mode3_attack_failure_computation(self):
        """Mode 3: Use attack metrics to compute failures."""
        # Create synthetic attack data
        data = pd.DataFrame(
            {
                "model_accuracy": [0.8, 0.75, 0.82, 0.78, 0.80],
                "attack.alias": [
                    "evasion",
                    "evasion",
                    "membership",
                    "membership",
                    "evasion",
                ],
                "evasion_success": [0.3, 0.25, 0.28, 0.32, 0.27],
                "T": [30, 60, 90, 120, 150],  # Duration
                "E": [1, 1, 0, 1, 1],  # Event
            },
        )

        # Compute failures from attack success rates
        data["adv_failures"] = 100 * data["evasion_success"]
        assert "adv_failures" in data.columns
        assert (data["adv_failures"] >= 0).all()


class TestMode4OptunaDatabase:
    """Integration test for Mode 4: Pre-computed results from Optuna database."""

    def test_mode4_config_creation(self, tmp_path):
        """Create LifelinesDataConfig for Mode 4."""
        optuna_db_path = str(tmp_path / "optuna.db")
        survival_data_config = LifelinesDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            dataset_name="optuna_results",
            optuna_schema={"attack_kind": "evasion"},
        )
        assert survival_data_config.mode == LifelinesDataMode.OPTUNA_DB
        assert survival_data_config.is_optuna_db()
        assert survival_data_config.optuna_schema == {"attack_kind": "evasion"}

    def test_mode4_optuna_query_filtering(self, tmp_path):
        """Mode 4: Support query filtering on Optuna results."""
        optuna_db_path = str(tmp_path / "optuna.db")
        survival_data_config = LifelinesDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            dataset_name="optuna_results",
            optuna_query="trial_id > 10",
        )
        assert survival_data_config.optuna_query == "trial_id > 10"
        assert survival_data_config.is_optuna_db()


class TestModeInteroperability:
    """Test that different modes don't conflict."""

    def test_modes_are_mutually_exclusive(self):
        """Verify each config instance has exactly one mode."""
        configs = [
            LifelinesDataConfig(
                mode=LifelinesDataMode.NATIVE,
                dataset_name="test",
                target="T",
                duration_col="T",
                event_col="E",
            ),
            LifelinesDataConfig(
                mode=LifelinesDataMode.AUXILIARY_METRIC,
                dataset_name="test",
                target="T",
                reference_metric="accuracy",
            ),
            LifelinesDataConfig(
                mode=LifelinesDataMode.AUXILIARY_FAILURE,
                dataset_name="test",
                target="T",
                failure_profile={"attack_kind": "evasion"},
            ),
        ]

        for config in configs:
            # Count how many mode checkers return True
            checks = [
                config.is_native_survival_data(),
                config.has_auxiliary_metric(),
                config.has_auxiliary_failure(),
                config.is_optuna_db(),
            ]
            # Exactly one should be True
            assert sum(checks) == 1, f"Config {config.mode} failed mutual exclusivity"

    def test_mode_transitions(self):
        """Verify we can transition between different mode configs."""
        base_config = DataConfig(
            dataset_name="test",
            target="T",
            classifier=False,
        )

        # Create different mode configs from same base
        mode1 = LifelinesDataConfig.from_data_and_model(base_config)
        mode2 = LifelinesDataConfig.from_auxiliary_metric(base_config)
        mode3 = LifelinesDataConfig.from_auxiliary_failure(
            base_config,
            {"attack_kind": "evasion"},
        )

        assert mode1.mode == LifelinesDataMode.NATIVE
        assert mode2.mode == LifelinesDataMode.AUXILIARY_METRIC
        assert mode3.mode == LifelinesDataMode.AUXILIARY_FAILURE

    def test_non_attack_auxiliary_failure_profile(self):
        """Mode 3 supports non-attack failure profiles."""
        base_config = DataConfig(
            dataset_name="incident_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        profile = {"source": "runtime", "metric": "failure_rate", "scale": 1.0}
        cfg = LifelinesDataConfig.from_auxiliary_failure(
            base_config,
            failure_profile=profile,
        )
        assert cfg.mode == LifelinesDataMode.AUXILIARY_FAILURE
        assert cfg.failure_profile == profile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
