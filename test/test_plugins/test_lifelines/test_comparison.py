import pandas as pd

from deckard.plugins.lifelines.comparison import write_aft_comparison_table


def test_write_aft_comparison_table_uses_canonical_filename(tmp_path):
    table = pd.DataFrame([{"model": "weibull", "AIC": 1.0}])

    csv_path = write_aft_comparison_table(table, str(tmp_path))

    assert csv_path.name == "aft_comparison.csv"
    assert csv_path.exists()


def test_write_aft_comparison_table_preserves_existing_rows(tmp_path):
    """Writing a new model should not drop previously written models."""
    first = pd.DataFrame([{"model": "weibull", "AIC": 1.0}])
    write_aft_comparison_table(first, str(tmp_path))

    second = pd.DataFrame([{"model": "cox", "AIC": 2.0}])
    write_aft_comparison_table(second, str(tmp_path))

    result = pd.read_csv(tmp_path / "aft_comparison.csv")
    assert set(result["model"]) == {"weibull", "cox"}


def test_write_aft_comparison_table_updates_existing_model_row(tmp_path):
    """A second write for the same model key should replace its row."""
    first = pd.DataFrame([{"model": "weibull", "AIC": 999.0}])
    write_aft_comparison_table(first, str(tmp_path))

    second = pd.DataFrame([{"model": "weibull", "AIC": 1.0}])
    write_aft_comparison_table(second, str(tmp_path))

    result = pd.read_csv(tmp_path / "aft_comparison.csv")
    assert len(result) == 1
    assert result.loc[result["model"] == "weibull", "AIC"].iloc[0] == 1.0
