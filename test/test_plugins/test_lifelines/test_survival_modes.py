"""Unit tests for LifelinesDataConfig and four survival analysis modes."""

import pytest

from deckard.data import DataConfig
from deckard.plugins.lifelines.data import LifelinesDataConfig, LifelinesDataMode


class TestLifelinesDataModeEnum:
    """Test LifelinesDataMode enum."""

    def test_all_modes_defined(self):
        """Verify all four modes are defined."""
        assert LifelinesDataMode.NATIVE
        assert LifelinesDataMode.AUXILIARY_METRIC
        assert LifelinesDataMode.AUXILIARY_FAILURE
        assert LifelinesDataMode.OPTUNA_DB

    def test_mode_string_values(self):
        """Verify mode string values."""
        assert LifelinesDataMode.NATIVE.value == "native"
        assert LifelinesDataMode.AUXILIARY_METRIC.value == "auxiliary_metric"
        assert LifelinesDataMode.AUXILIARY_FAILURE.value == "auxiliary_failure"
        assert LifelinesDataMode.OPTUNA_DB.value == "optuna_db"
        assert LifelinesDataMode.AUXILIARY_MODEL.value == "auxiliary_metric"
        assert LifelinesDataMode.AUXILIARY_ATTACK.value == "auxiliary_failure"


class TestLifelinesDataConfigNativeMode:
    """Test LifelinesDataConfig in NATIVE mode."""

    def test_native_mode_creation(self):
        """Create a NATIVE mode config."""
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.NATIVE,
            name="lifelines_diabetes",
            target="T",
            duration_col="T",
            event_col="E",
        )
        assert config.mode == LifelinesDataMode.NATIVE
        assert config.duration_col == "T"
        assert config.event_col == "E"

    def test_native_mode_from_classmethod(self):
        """Create NATIVE mode config using classmethod."""
        base_config = DataConfig(
            name="lifelines_diabetes",
            target="T",
            classifier=False,
        )
        config = LifelinesDataConfig.from_data_and_model(
            base_config,
            duration_col="T",
            event_col="E",
        )
        assert config.mode == LifelinesDataMode.NATIVE
        assert config.is_native_survival_data()

    def test_native_mode_requires_duration_col(self):
        """NATIVE mode requires duration_col."""
        with pytest.raises(ValueError, match="duration_col required"):
            LifelinesDataConfig(
                mode=LifelinesDataMode.NATIVE,
                name="test",
                target="T",
                duration_col="",
                event_col="E",
            )

    def test_native_mode_requires_event_col(self):
        """NATIVE mode requires event_col."""
        with pytest.raises(ValueError, match="event_col required"):
            LifelinesDataConfig(
                mode=LifelinesDataMode.NATIVE,
                name="test",
                target="T",
                duration_col="T",
                event_col="",
            )

    def test_native_mode_checker(self):
        """Test is_native_survival_data checker."""
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.NATIVE,
            name="test",
            target="T",
            duration_col="T",
            event_col="E",
        )
        assert config.is_native_survival_data()
        assert not config.has_auxiliary_metric()
        assert not config.has_auxiliary_failure()
        assert not config.is_optuna_db()


class TestLifelinesDataConfigAuxiliaryMetricMode:
    """Test LifelinesDataConfig in AUXILIARY_METRIC mode."""

    def test_auxiliary_metric_mode_creation(self):
        """Create an AUXILIARY_METRIC mode config."""
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.AUXILIARY_METRIC,
            name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
            reference_metric="accuracy",
        )
        assert config.mode == LifelinesDataMode.AUXILIARY_METRIC
        assert config.reference_metric == "accuracy"

    def test_auxiliary_metric_mode_from_classmethod(self):
        """Create AUXILIARY_METRIC mode config using classmethod."""
        base_config = DataConfig(
            name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        config = LifelinesDataConfig.from_auxiliary_metric(
            base_config,
            reference_metric="accuracy",
        )
        assert config.mode == LifelinesDataMode.AUXILIARY_METRIC
        assert config.has_auxiliary_metric()

    def test_auxiliary_metric_requires_reference_metric(self):
        """AUXILIARY_METRIC mode requires reference_metric."""
        with pytest.raises(ValueError, match="reference_metric required"):
            LifelinesDataConfig(
                mode=LifelinesDataMode.AUXILIARY_METRIC,
                name="test",
                target="T",
                reference_metric="",
            )

    def test_auxiliary_metric_mode_checker(self):
        """Test has_auxiliary_metric checker."""
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.AUXILIARY_METRIC,
            name="test",
            target="T",
            reference_metric="accuracy",
        )
        assert config.has_auxiliary_metric()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_failure()
        assert not config.is_optuna_db()


class TestLifelinesDataConfigAuxiliaryFailureMode:
    """Test LifelinesDataConfig in AUXILIARY_FAILURE mode."""

    def test_auxiliary_failure_mode_creation(self):
        """Create an AUXILIARY_FAILURE mode config."""
        failure_profile = {"attack_kind": "evasion", "attack_size": 100}
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.AUXILIARY_FAILURE,
            name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
            failure_profile=failure_profile,
        )
        assert config.mode == LifelinesDataMode.AUXILIARY_FAILURE
        assert config.failure_profile == failure_profile

    def test_auxiliary_failure_mode_from_classmethod(self):
        """Create AUXILIARY_FAILURE mode config using classmethod."""
        base_config = DataConfig(
            name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        failure_profile = {"attack_kind": "membership"}
        config = LifelinesDataConfig.from_auxiliary_failure(
            base_config,
            failure_profile=failure_profile,
        )
        assert config.mode == LifelinesDataMode.AUXILIARY_FAILURE
        assert config.has_auxiliary_failure()

    def test_auxiliary_failure_requires_failure_profile(self):
        """AUXILIARY_FAILURE mode requires failure_profile."""
        with pytest.raises(ValueError, match="failure_profile required"):
            LifelinesDataConfig(
                mode=LifelinesDataMode.AUXILIARY_FAILURE,
                name="test",
                target="T",
                failure_profile=None,
            )

    def test_auxiliary_failure_mode_checker(self):
        """Test has_auxiliary_failure checker."""
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.AUXILIARY_FAILURE,
            name="test",
            target="T",
            failure_profile={"attack_kind": "evasion"},
        )
        assert config.has_auxiliary_failure()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_metric()
        assert not config.is_optuna_db()


class TestLifelinesDataConfigOptunaMode:
    """Test LifelinesDataConfig in OPTUNA_DB mode."""

    def test_optuna_db_mode_creation(self, tmp_path):
        """Create an OPTUNA_DB mode config."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.OPTUNA_DB,
            name="optuna_results",
            target="optuna_result",
            classifier=False,
            optuna_db=optuna_db_path,
        )
        assert config.mode == LifelinesDataMode.OPTUNA_DB
        assert config.optuna_db == optuna_db_path

    def test_optuna_db_mode_from_classmethod(self, tmp_path):
        """Create OPTUNA_DB mode config using classmethod."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = LifelinesDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            name="optuna_results",
        )
        assert config.mode == LifelinesDataMode.OPTUNA_DB
        assert config.is_optuna_db()

    def test_optuna_db_mode_from_classmethod_with_canonical_name(self, tmp_path):
        """Create OPTUNA_DB mode config using canonical name constructor arg."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = LifelinesDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            name="optuna_results",
        )
        assert config.mode == LifelinesDataMode.OPTUNA_DB
        assert config.name == "optuna_results"

    def test_optuna_db_requires_optuna_db_path(self):
        """OPTUNA_DB mode requires optuna_db path."""
        with pytest.raises(ValueError, match="optuna_db path required"):
            LifelinesDataConfig(
                mode=LifelinesDataMode.OPTUNA_DB,
                name="test",
                target="T",
                optuna_db=None,
            )

    def test_optuna_db_with_schema_and_query(self, tmp_path):
        """OPTUNA_DB mode can have schema and query filters."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = LifelinesDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            optuna_schema={"attack_kind": "evasion"},
            optuna_query="trial_id > 10",
        )
        assert config.optuna_schema == {"attack_kind": "evasion"}
        assert config.optuna_query == "trial_id > 10"

    def test_optuna_db_mode_checker(self, tmp_path):
        """Test is_optuna_db checker."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = LifelinesDataConfig(
            mode=LifelinesDataMode.OPTUNA_DB,
            name="test",
            target="T",
            optuna_db=optuna_db_path,
        )
        assert config.is_optuna_db()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_metric()
        assert not config.has_auxiliary_failure()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
