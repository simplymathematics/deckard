"""Helpers for running OpenAttack attacks against Deckard transformer models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ...artifacts import ScoreDict
from ...attack.base import AttackConfig
from ...model import ModelConfig
from ..text_runtime import (
    TransformerTextAdapter,
    build_transformer_text_adapter,
    finalize_attack_runtime_outputs,
    resolve_text_batch,
)


class _DeckardOpenAttackClassifierBase:
    """Adapter-backed classifier API consumed by OpenAttack AttackEval."""

    def __init__(self, adapter: TransformerTextAdapter):
        self._adapter = adapter

    def get_prob(self, input_):
        """Return class probabilities for a batch of raw text inputs.

        Args:
            input_: Iterable text payload accepted by OpenAttack classifier API.

        Returns:
            Probability matrix for the input batch.
        """
        return self._adapter.predict_proba(list(input_))

    def get_pred(self, input_):
        """Return predicted class labels for a batch of raw text inputs.

        Args:
            input_: Iterable text payload accepted by OpenAttack classifier API.

        Returns:
            Predicted class label vector.
        """
        return self.get_prob(input_).argmax(axis=1)


def _build_openattack_classifier(adapter: TransformerTextAdapter):
    try:
        OpenAttack = import_module("OpenAttack")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenAttack helpers require the optional 'OpenAttack' dependency.",
        ) from exc

    classifier_type = type(
        "DeckardOpenAttackClassifier",
        (_DeckardOpenAttackClassifierBase, OpenAttack.Classifier),
        {},
    )
    return classifier_type(adapter)


def _resolve_attacker_name(attack_config: AttackConfig) -> str:
    attack_name = str(attack_config.resolve_name(default="") or "").strip()
    if attack_name == "":
        raise ValueError(
            "AttackConfig must include an attacker class path for OpenAttack.",
        )
    return attack_name.split(".")[-1]


def _runtime_plugin_options(runtime: AttackConfig) -> tuple[str, bool]:
    params = runtime.attack_params if isinstance(runtime.attack_params, dict) else {}
    split = str(params.get("split", "test") or "test")
    fail_on_error = bool(params.get("fail_on_error", False))
    return split, fail_on_error


def run_openattack_attack_config(
    attack_config: AttackConfig,
    *,
    data: Any,
    model: ModelConfig | Any,
    split: str | None = None,
) -> dict[str, Any]:
    """Run one OpenAttack attacker against raw text samples from a Deckard dataset."""
    texts, labels, tokenizer = resolve_text_batch(data, attack_config, split=split)
    adapter = build_transformer_text_adapter(
        data=data,
        model=model,
        tokenizer=tokenizer,
    )
    victim = _build_openattack_classifier(adapter)

    try:
        OpenAttack = import_module("OpenAttack")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenAttack helpers require the optional 'OpenAttack' dependency.",
        ) from exc

    attacker_name = _resolve_attacker_name(attack_config)
    attacker_cls = getattr(OpenAttack.attackers, attacker_name, None)
    if attacker_cls is None:
        raise ValueError(f"Unsupported OpenAttack attacker: {attacker_name}")
    attack_eval = OpenAttack.AttackEval(
        attacker_cls(**dict(getattr(attack_config, "attack_params", {}) or {})),
        victim,
    )
    dataset = [{"x": text, "y": int(label)} for text, label in zip(texts, labels)]

    records: list[dict[str, Any]] = []
    for item in attack_eval.ieval(dataset):
        original_text = str(item["data"]["x"])
        adversarial_text = (
            item["result"] if item["result"] is not None else original_text
        )
        probabilities = victim.get_prob([original_text, adversarial_text])
        records.append(
            {
                "library": "openattack",
                "attacker": attacker_name,
                "original_text": original_text,
                "adversarial_text": adversarial_text,
                "ground_truth": int(item["data"]["y"]),
                "original_prediction": int(np.argmax(probabilities[0])),
                "adversarial_prediction": int(np.argmax(probabilities[1])),
                "success": bool(item["success"]),
                "metrics": dict(item.get("metrics", {})),
            },
        )

    success_count = sum(1 for record in records if record["success"])
    return {
        "library": "openattack",
        "attack_name": attacker_name,
        "num_examples": len(records),
        "successful_examples": success_count,
        "results": records,
    }


class OpenAttackConfig(AttackConfig):
    """Dedicated runtime attack config for OpenAttack-backed attack names.

    Attributes:
        Inherits runtime attack configuration fields from ``AttackConfig``.
    """

    def __call__(
        self,
        data: Any,
        model: ModelConfig | Any,
        art_model: Any,
        attack: Any,
        attack_family: str,
        attack_sub_family: str,
    ) -> ScoreDict:
        """Execute OpenAttack runtime flow and persist attack runtime outputs.

        Args:
            data: Runtime data payload.
            model: Runtime model payload.
            art_model: Unused ART model payload (plugin manages runtime directly).
            attack: Unused attack object placeholder.
            attack_family: Canonical attack family token.
            attack_sub_family: Canonical attack subtype token.

        Returns:
            Runtime score dictionary updated with OpenAttack outputs.

        Raises:
            ValueError: If attack family is unsupported for OpenAttack runtime.
        """
        _ = (art_model, attack, attack_sub_family)
        if (attack_family or "").lower() != "evasion":
            raise ValueError(
                f"OpenAttackConfig received unsupported attack family: {attack_family}",
            )

        split, fail_on_error = _runtime_plugin_options(self)
        runtime = self
        try:
            result = run_openattack_attack_config(
                runtime,
                data=data,
                model=model,
                split=split,
            )
            records = list(result.get("results", []))
            successful = int(result.get("successful_examples", 0) or 0)
            count = int(result.get("num_examples", len(records)) or 0)
            error = None
            attack_name = str(
                result.get(
                    "attack_name",
                    runtime.resolve_name(default="openattack"),
                )
                or "openattack",
            )
        except Exception as exc:  # pragma: no cover
            if fail_on_error:
                raise
            records = []
            successful = 0
            count = 0
            attack_name = str(
                runtime.resolve_name(default="openattack") or "openattack",
            )
            error = str(exc)

        return finalize_attack_runtime_outputs(
            runtime,
            records=records,
            library="openattack",
            attack_name=attack_name,
            successful_examples=successful,
            count=count,
            error=error,
        )


__all__ = [
    "run_openattack_attack_config",
    "OpenAttackConfig",
]
