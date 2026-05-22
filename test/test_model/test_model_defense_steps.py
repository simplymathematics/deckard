from types import SimpleNamespace

from deckard.model.base import ModelConfig
from deckard.model.defend import DefensePipelineConfig, DefenseStep


class _DummyDefense:
    def __init__(self, name):
        self.defense_name = name
        self.calls = 0

    def apply_to(self, estimator, data):
        self.calls += 1
        return {"wrapped": estimator, "data": data, "name": self.defense_name}


def test_fairlearn_and_anjana_defense_steps_default_to_fit_and_predict():
    fair = _DummyDefense("fairlearn.reductions.ExponentiatedGradient")
    anjana = _DummyDefense("anjana.defense.PrivacyStep")

    fair_step = DefenseStep.from_defense(fair)
    anjana_step = DefenseStep.from_defense(anjana)

    assert fair_step.apply_fit is True
    assert fair_step.apply_predict is True
    assert anjana_step.apply_fit is True
    assert anjana_step.apply_predict is True


def test_art_defense_step_accepts_explicit_flags_from_step_payload():
    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense": _DummyDefense("art.defences.postprocessor.HighConfidence"),
                "apply_fit": False,
                "apply_predict": True,
            }
        ],
    )

    step = pipeline.defenses[0]
    assert isinstance(step, _DummyDefense)
    assert step.apply_fit is False
    assert step.apply_predict is True


def test_art_defense_flags_can_be_provided_in_defense_params():
    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": True, "apply_predict": False},
            }
        ],
    )

    step = pipeline.defenses[0]
    assert step.apply_fit is True
    assert step.apply_predict is False


def test_pipeline_stage_gates_defense_execution_by_step_flags():
    defense = _DummyDefense("art.defences.postprocessor.HighConfidence")
    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense": defense,
                "apply_fit": False,
                "apply_predict": True,
            }
        ],
    )

    fit_stage_output = pipeline.apply(
        estimator={"base": 1},
        data=SimpleNamespace(name="payload"),
        stage="pre_fit",
    )
    predict_stage_output = pipeline.apply(
        estimator={"base": 2},
        data=SimpleNamespace(name="payload"),
        stage="post_fit_pre_predict",
    )

    assert fit_stage_output == {"base": 1}
    assert predict_stage_output["wrapped"] == {"base": 2}
    assert defense.calls == 1


def test_defense_step_factory_proxies_flags():
    defense = _DummyDefense("art.defences.postprocessor.HighConfidence")
    step = DefenseStep.from_defense(defense, apply_fit=False, apply_predict=True)

    assert step.apply_fit is False
    assert step.apply_predict is True
    assert step.defense is defense


def test_pretrained_model_with_fit_defense_snapshots_and_retrains(monkeypatch):
    model = ModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 10},
        alias="demo",
    )
    model._model = SimpleNamespace(existing=True)
    model.score_dict = {"baseline_accuracy": 0.9}
    model.training_predictions = [1, 0]
    model.predictions = [0, 1]
    model.training_time = 1.0
    model.prediction_time = 2.0
    model.defense = object()

    defense_pipeline = SimpleNamespace(
        requires_fit_application=lambda: True,
        resolve_stage=lambda **_kwargs: "post_fit_pre_predict",
        apply=lambda estimator, data, stage: estimator,
        defense_application_time=None,
        score_dict={},
    )

    retrain_calls = {}

    def fake_train_with_runtime_trainer(data, model_file, times, force_retrain=False):
        retrain_calls["force_retrain"] = force_retrain
        retrain_calls["model_file"] = model_file
        retrain_calls["times"] = dict(times)
        model.training_time = 3.0
        model.training_n = len(data.X_train)
        return {**times, "training_time": 3.0, "training_n": len(data.X_train)}

    monkeypatch.setattr(model, "_is_model_fitted", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model, "_require_defense_pipeline", lambda: defense_pipeline)
    monkeypatch.setattr(model, "_train_with_runtime_trainer", fake_train_with_runtime_trainer)
    monkeypatch.setattr(model, "_initialize_model", lambda: setattr(model, "_model", SimpleNamespace(retrained=True)))
    monkeypatch.setattr(model, "_apply_defense", lambda _data: model._model)

    data = SimpleNamespace(X_train=[0, 1], y_train=[0, 1])
    times = model._load_or_train_model(data, model_file=None, times={})

    assert retrain_calls["force_retrain"] is True
    assert model._pre_defense_runtime_state["predictions"] == [0, 1]
    assert model.score_dict["pre-demo-defense"]["retrain_required"] is True
    assert times["training_time"] == 3.0
