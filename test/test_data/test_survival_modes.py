"""Unit tests for SurvivalDataConfig and four survival analysis modes."""
import pytest

from deckard.data import DataConfig
from deckard.data.survival import SurvivalDataConfig, SurvivalDataMode


class TestSurvivalDataModeEnum:
    """Test SurvivalDataMode enum."""

    def test_all_modes_defined(self):
        """Verify all four modes are defined."""
        assert SurvivalDataMode.NATIVE
        assert SurvivalDataMode.AUXILIARY_MODEL
        assert SurvivalDataMode.AUXILIARY_ATTACK
        assert SurvivalDataMode.OPTUNA_DB

    def test_mode_string_values(self):
        """Verify mode string values."""
        assert SurvivalDataMode.NATIVE.value == "native"
        assert SurvivalDataMode.AUXILIARY_MODEL.value == "auxiliary_model"
        assert SurvivalDataMode.AUXILIARY_ATTACK.value == "auxiliary_attack"
        assert SurvivalDataMode.OPTUNA_DB.value == "optuna_db"


class TestSurvivalDataConfigNativeMode:
    """Test SurvivalDataConfig in NATIVE mode."""

    def test_native_mode_creation(self):
        """Create a NATIVE mode config."""
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.NATIVE,
            dataset_name="lifelines_diabetes",
            target="T",
            duration_col="T",
            event_col="E",
        )
        assert config.mode == SurvivalDataMode.NATIVE
        assert config.duration_col == "T"
        assert config.event_col == "E"

    def test_native_mode_from_classmethod(self):
        """Create NATIVE mode config using classmethod."""
        base_config = DataConfig(
            dataset_name="lifelines_diabetes",
            target="T",
            classifier=False,
        )
        config = SurvivalDataConfig.from_data_and_model(
            base_config,
            duration_col="T",
            event_col="E",
        )
        assert config.mode == SurvivalDataMode.NATIVE
        assert config.is_native_survival_data()

    def test_native_mode_requires_duration_col(self):
        """NATIVE mode requires duration_col."""
        with pytest.raises(ValueError, match="duration_col required"):
            SurvivalDataConfig(
                mode=SurvivalDataMode.NATIVE,
                dataset_name="test",
                target="T",
                duration_col="",
                event_col="E",
            )

    def test_native_mode_requires_event_col(self):
        """NATIVE mode requires event_col."""
        with pytest.raises(ValueError, match="event_col required"):
            SurvivalDataConfig(
                mode=SurvivalDataMode.NATIVE,
                dataset_name="test",
                target="T",
                duration_col="T",
                event_col="",
            )

    def test_native_mode_checker(self):
        """Test is_native_survival_data checker."""
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.NATIVE,
            dataset_name="test",
            target="T",
            duration_col="T",
            event_col="E",
        )
        assert config.is_native_survival_data()
        assert not config.has_auxiliary_model()
        assert not config.has_auxiliary_attack()
        assert not config.is_optuna_db()


class TestSurvivalDataConfigAuxiliaryModelMode:
    """Test SurvivalDataConfig in AUXILIARY_MODEL mode."""

    def test_auxiliary_model_mode_creation(self):
        """Create an AUXILIARY_MODEL mode config."""
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.AUXILIARY_MODEL,
            dataset_name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
            benign_metric="accuracy",
        )
        assert config.mode == SurvivalDataMode.AUXILIARY_MODEL
        assert config.benign_metric == "accuracy"

    def test_auxiliary_model_mode_from_classmethod(self):
        """Create AUXILIARY_MODEL mode config using classmethod."""
        base_config = DataConfig(
            dataset_name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        config = SurvivalDataConfig.from_auxiliary_model(
            base_config,
            benign_metric="accuracy",
        )
        assert config.mode == SurvivalDataMode.AUXILIARY_MODEL
        assert config.has_auxiliary_model()

    def test_auxiliary_model_requires_benign_metric(self):
        """AUXILIARY_MODEL mode requires benign_metric."""
        with pytest.raises(ValueError, match="benign_metric required"):
            SurvivalDataConfig(
                mode=SurvivalDataMode.AUXILIARY_MODEL,
                dataset_name="test",
                target="T",
                benign_metric="",
            )

    def test_auxiliary_model_mode_checker(self):
        """Test has_auxiliary_model checker."""
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.AUXILIARY_MODEL,
            dataset_name="test",
            target="T",
            benign_metric="accuracy",
        )
        assert config.has_auxiliary_model()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_attack()
        assert not config.is_optuna_db()


class TestSurvivalDataConfigAuxiliaryAttackMode:
    """Test SurvivalDataConfig in AUXILIARY_ATTACK mode."""

    def test_auxiliary_attack_mode_creation(self):
        """Create an AUXILIARY_ATTACK mode config."""
        attack_config = {"attack_kind": "evasion", "attack_size": 100}
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.AUXILIARY_ATTACK,
            dataset_name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
            attack_config=attack_config,
        )
        assert config.mode == SurvivalDataMode.AUXILIARY_ATTACK
        assert config.attack_config == attack_config

    def test_auxiliary_attack_mode_from_classmethod(self):
        """Create AUXILIARY_ATTACK mode config using classmethod."""
        base_config = DataConfig(
            dataset_name="toy_dataset",
            target="adv_failure_rate",
            classifier=False,
        )
        attack_config = {"attack_kind": "membership"}
        config = SurvivalDataConfig.from_auxiliary_attack(
            base_config,
            attack_config=attack_config,
        )
        assert config.mode == SurvivalDataMode.AUXILIARY_ATTACK
        assert config.has_auxiliary_attack()

    def test_auxiliary_attack_requires_attack_config(self):
        """AUXILIARY_ATTACK mode requires attack_config."""
        with pytest.raises(ValueError, match="attack_config required"):
            SurvivalDataConfig(
                mode=SurvivalDataMode.AUXILIARY_ATTACK,
                dataset_name="test",
                target="T",
                attack_config=None,
            )

    def test_auxiliary_attack_mode_checker(self):
        """Test has_auxiliary_attack checker."""
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.AUXILIARY_ATTACK,
            dataset_name="test",
            target="T",
            attack_config={"attack_kind": "evasion"},
        )
        assert config.has_auxiliary_attack()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_model()
        assert not config.is_optuna_db()


class TestSurvivalDataConfigOptunaMode:
    """Test SurvivalDataConfig in OPTUNA_DB mode."""

    def test_optuna_db_mode_creation(self, tmp_path):
        """Create an OPTUNA_DB mode config."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.OPTUNA_DB,
            dataset_name="optuna_results",
            target="optuna_result",
            classifier=False,
            optuna_db=optuna_db_path,
        )
        assert config.mode == SurvivalDataMode.OPTUNA_DB
        assert config.optuna_db == optuna_db_path

    def test_optuna_db_mode_from_classmethod(self, tmp_path):
        """Create OPTUNA_DB mode config using classmethod."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = SurvivalDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            dataset_name="optuna_results",
        )
        assert config.mode == SurvivalDataMode.OPTUNA_DB
        assert config.is_optuna_db()

    def test_optuna_db_requires_optuna_db_path(self):
        """OPTUNA_DB mode requires optuna_db path."""
        with pytest.raises(ValueError, match="optuna_db path required"):
            SurvivalDataConfig(
                mode=SurvivalDataMode.OPTUNA_DB,
                dataset_name="test",
                target="T",
                optuna_db=None,
            )

    def test_optuna_db_with_schema_and_query(self, tmp_path):
        """OPTUNA_DB mode can have schema and query filters."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = SurvivalDataConfig.from_optuna_db(
            optuna_db=optuna_db_path,
            optuna_schema={"attack_kind": "evasion"},
            optuna_query="trial_id > 10",
        )
        assert config.optuna_schema == {"attack_kind": "evasion"}
        assert config.optuna_query == "trial_id > 10"

    def test_optuna_db_mode_checker(self, tmp_path):
        """Test is_optuna_db checker."""
        optuna_db_path = str(tmp_path / "optuna.db")
        config = SurvivalDataConfig(
            mode=SurvivalDataMode.OPTUNA_DB,
            dataset_name="test",
            target="T",
            optuna_db=optuna_db_path,
        )
        assert config.is_optuna_db()
        assert not config.is_native_survival_data()
        assert not config.has_auxiliary_model()
        assert not config.has_auxiliary_attack()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
