from deckard.plugins.anjana.model import AnjanaModelConfig
from deckard.plugins.anjana.score import DefaultAnjanaModelScorerDictConfig


def test_anjana_model_auto_scorer_uses_anjana_default(monkeypatch):
    class _StubScorer:
        def __call__(self, **kwargs):
            _ = kwargs
            return {"k_anonymity": 1.0}

    monkeypatch.setattr(
        "deckard.plugins.anjana.model.load_class",
        lambda _: _StubScorer(),
    )

    model = AnjanaModelConfig(
        model_type="sklearn.linear_model.LogisticRegression",
        classifier=True,
        model_params={"max_iter": 10},
        scorer="auto",
    )

    assert callable(model.scorer)


def test_default_anjana_model_scorer_includes_privacy_metrics():
    scorer = DefaultAnjanaModelScorerDictConfig()

    assert set(scorer.scorers) == {"k_anonymity", "l_diversity", "t_closeness"}
