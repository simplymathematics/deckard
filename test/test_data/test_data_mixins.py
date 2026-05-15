from deckard.data.base import DataConfig


def test_data_config_exposes_data_mixins():
    cfg = DataConfig(dataset_name="synthetic", classifier=True)

    assert callable(getattr(cfg, "load_raw_data"))
    assert callable(getattr(cfg, "split_data"))
    assert callable(getattr(cfg, "compute_score"))
