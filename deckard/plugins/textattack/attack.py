"""Helpers for running TextAttack recipes against Deckard transformer models."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...attack.base import AttackConfig
from ...artifacts import ScoreDict
from ...model import ModelConfig
from ..text_runtime import (
    apply_attack_runtime_outputs,
    TransformerTextAdapter,
    resolve_runtime_model,
    resolve_text_batch,
    resolve_text_max_length,
)


def _build_model_wrapper(adapter: TransformerTextAdapter):
    try:
        from textattack.models.wrappers import ModelWrapper
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TextAttack helpers require the optional 'textattack' dependency.",
        ) from exc

    class DeckardTextAttackModelWrapper(ModelWrapper):
        def __init__(self):
            super().__init__()
            # TextAttack goal-function validators inspect wrapper.model.
            self.model = adapter.model

        def __call__(self, text_input_list, **kwargs):
            _ = kwargs
            return adapter.predict_logits(list(text_input_list))

    return DeckardTextAttackModelWrapper()


def _resolve_recipe_name(attack_config: AttackConfig) -> str:
    attack_name = str(attack_config.resolve_name(default="") or "").strip()
    if attack_name == "":
        raise ValueError(
            "AttackConfig must include a recipe class path for TextAttack."
        )
    return attack_name.split(".")[-1]


def _instantiate_recipe(recipe_name: str, model_wrapper: Any):
    try:
        from textattack import attack_recipes
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TextAttack helpers require the optional 'textattack' dependency.",
        ) from exc
    recipe_cls = getattr(attack_recipes, recipe_name, None)
    if recipe_cls is None:
        raise ValueError(f"Unsupported TextAttack recipe: {recipe_name}")
    return recipe_cls.build(model_wrapper)


def _result_text(result: Any, attr_name: str, fallback: str) -> str:
    result_obj = getattr(result, attr_name, None)
    attacked_text = getattr(result_obj, "attacked_text", None)
    text = getattr(attacked_text, "text", None)
    return str(text) if text is not None else fallback


def _runtime_plugin_options(runtime: AttackConfig) -> tuple[str, bool]:
    params = runtime.attack_params if isinstance(runtime.attack_params, dict) else {}
    split = str(params.get("split", "test") or "test")
    fail_on_error = bool(params.get("fail_on_error", False))
    return split, fail_on_error


def run_textattack_attack_config(
    runtime: AttackConfig,
    *,
    data: Any,
    model: ModelConfig | Any,
    split: str = "test",
    fail_on_error: bool = False,
) -> ScoreDict:
    """Run one configured TextAttack recipe and update runtime attack outputs."""
    try:
        adapter, recipe_name, recipe, texts, labels = _prepare_runtime_objects(
            runtime=runtime,
            data=data,
            model=model,
            split=split,
        )
        records = _collect_attack_records(
            recipe=recipe,
            recipe_name=recipe_name,
            adapter=adapter,
            texts=texts,
            labels=labels,
        )
        successful, count, attack_name, error = _build_success_metadata(
            records=records,
            attack_name=recipe_name,
        )
    except Exception as exc:  # pragma: no cover
        if fail_on_error:
            raise
        records, successful, count, attack_name, error = _build_failure_metadata(
            runtime=runtime,
            exc=exc,
        )

    apply_attack_runtime_outputs(
        runtime=runtime,
        records=records,
        library="textattack",
        attack_name=attack_name,
        successful_examples=successful,
        count=count,
        error=error,
    )
    return runtime.score_dict


class TextAttackConfig(AttackConfig):
    """Dedicated runtime attack config for TextAttack-backed attack names."""

    def __call__(
        self,
        data: Any,
        model: ModelConfig | Any,
        art_model: Any,
        attack: Any,
        attack_family: str,
        attack_sub_family: str,
    ) -> ScoreDict:
        _ = (art_model, attack, attack_sub_family)
        if (attack_family or "").lower() != "evasion":
            raise ValueError(
                f"TextAttackConfig received unsupported attack family: {attack_family}",
            )
        split, fail_on_error = _runtime_plugin_options(self)
        return run_textattack_attack_config(
            self,
            data=data,
            model=model,
            split=split,
            fail_on_error=fail_on_error,
        )


def _prepare_runtime_objects(
    *,
    runtime: AttackConfig,
    data: Any,
    model: ModelConfig | Any,
    split: str,
) -> tuple[TransformerTextAdapter, str, Any, list[str], np.ndarray]:
    texts, labels, tokenizer = resolve_text_batch(
        data,
        runtime,
        split=split,
    )
    runtime_model = resolve_runtime_model(model)
    adapter = TransformerTextAdapter(
        model=runtime_model,
        tokenizer=tokenizer,
        max_length=resolve_text_max_length(data),
    )
    wrapper = _build_model_wrapper(adapter)
    recipe_name = _resolve_recipe_name(runtime)
    recipe = _instantiate_recipe(recipe_name, wrapper)
    return adapter, recipe_name, recipe, texts, labels


def _collect_attack_records(
    *,
    recipe: Any,
    recipe_name: str,
    adapter: TransformerTextAdapter,
    texts: list[str],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for text, label in zip(texts, labels):
        result_item = recipe.attack(text, int(label))
        adversarial_text = _result_text(
            result_item,
            "perturbed_result",
            text,
        )
        original_logits = adapter.predict_logits([text])[0]
        adversarial_logits = adapter.predict_logits([adversarial_text])[0]
        original_pred = int(np.argmax(original_logits))
        adversarial_pred = int(np.argmax(adversarial_logits))
        records.append(
            {
                "library": "textattack",
                "recipe": recipe_name,
                "original_text": text,
                "adversarial_text": adversarial_text,
                "ground_truth": int(label),
                "original_prediction": original_pred,
                "adversarial_prediction": adversarial_pred,
                "success": bool(
                    adversarial_text != text and adversarial_pred != original_pred
                ),
                "result_type": type(result_item).__name__,
            },
        )
    return records


def _build_success_metadata(
    *,
    records: list[dict[str, Any]],
    attack_name: str,
) -> tuple[int, int, str, None]:
    successful = sum(1 for record in records if record["success"])
    count = len(records)
    return successful, count, attack_name, None


def _build_failure_metadata(
    *,
    runtime: AttackConfig,
    exc: Exception,
) -> tuple[list[dict[str, Any]], int, int, str, str]:
    return (
        [],
        0,
        0,
        str(runtime.resolve_name(default="textattack")),
        str(exc),
    )

__all__ = [
    "run_textattack_attack_config",
    "TextAttackConfig",
]
