from deckard.model.base import ModelConfig


class _FallbackEstimator:
    _estimator_type = "classifier"

    def __init__(self):
        self.alpha = 0.25
        self.beta = 7

    def get_params(self):
        raise RuntimeError("params unavailable")

    def predict_proba(self, x):
        return x


class _ParamNamesEstimator:
    _estimator_type = "regressor"

    def __init__(self):
        self.gamma = 3
        self.delta = 9
        self._private = "ignore"

    def _get_param_names(self):
        return ["gamma", "delta"]

    def get_params(self):
        raise RuntimeError("params unavailable")


def test_sync_model_signature_fallback_uses_runtime_attrs_when_get_params_fails():
    cfg = object.__new__(ModelConfig)
    cfg.model_params = {"seed": 42}
    cfg.probability = False

    estimator = _FallbackEstimator()
    cfg._sync_model_signature_from_estimator(estimator)

    assert cfg.name.endswith("_FallbackEstimator")
    assert cfg.model_params["alpha"] == 0.25
    assert cfg.model_params["beta"] == 7
    assert cfg.model_params["seed"] == 42
    assert cfg.probability is True


def test_sync_model_signature_prefers_param_name_contract_without_inspect():
    cfg = object.__new__(ModelConfig)
    cfg.model_params = {"existing": 1}
    cfg.probability = False

    estimator = _ParamNamesEstimator()
    cfg._sync_model_signature_from_estimator(estimator)

    assert cfg.name.endswith("_ParamNamesEstimator")
    assert cfg.model_params["gamma"] == 3
    assert cfg.model_params["delta"] == 9
    assert cfg.model_params["existing"] == 1
    assert "_private" not in cfg.model_params
