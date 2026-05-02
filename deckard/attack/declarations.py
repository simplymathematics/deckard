"""Static attack configuration declarations and ConfigStore registrations."""

from ..utils import safe_store


ATTACK_BOUNDARY = {
    "attack_type": "art.attacks.evasion.BoundaryAttack",
    "attack_params": {
        "batch_size": "${..attack_size}",
        "targeted": False,
        "delta": 0.01,
        "epsilon": 0.01,
        "max_iter": 10,
        "num_trial": 25,
        "sample_size": "${..attack_size}",
        "init_size": "${..attack_size}",
        "min_epsilon": 0.0,
        "verbose": False,
    },
    "attack_size": 10,
    "_target_": "deckard.attack.AttackConfig",
    "alias": "boundary",
}

ATTACK_FGM = {
    "attack_type": "art.attacks.evasion.FastGradientMethod",
    "attack_params": {
        "eps_step": 0.01,
        "norm": "inf",
        "targeted": False,
        "eps": 1,
        "minimal": False,
    },
    "_target_": "deckard.attack.AttackConfig",
    "alias": "fgm",
}

ATTACK_HSJ = {
    "attack_type": "art.attacks.evasion.HopSkipJump",
    "attack_params": {
        "max_iter": 10,
        "init_eval": 1,
        "max_eval": 20,
        "init_size": 100,
        "norm": 2,
        "targeted": False,
        "verbose": False,
    },
    "attack_size": 10,
    "_target_": "deckard.attack.AttackConfig",
    "alias": "hsj",
}

ATTACK_MEMBERSHIP = {
    "attack_type": "art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
    "attack_size": 100,
    "attack_params": {
        "attack_model": "${model}",
    },
    "_target_": "deckard.attack.AttackConfig",
    "alias": "membership",
}

ATTACK_ATTRIBUTE_BB = {
    "attack_type": "art.attacks.inference.attribute_inference.AttributeInferenceBlackBox",
    "targeted_attribute": ["age"],
    "attack_params": {
        "attack_model_type": "nn",
        "scale_range": "(0, 89)",
        "is_continuous": True,
        "nn_model_epochs": 1,
    },
    "attack_size": 1000,
    "_target_": "deckard.attack.AttackConfig",
    "alias": "attribute_bb",
}

ATTACK_ZOO = {
    "attack_type": "art.attacks.evasion.ZooAttack",
    "attack_params": {
        "confidence": 0.0,
        "targeted": False,
        "max_iter": 10,
        "binary_search_steps": 1,
        "initial_const": 0.001,
        "abort_early": True,
        "use_resize": True,
        "use_importance": True,
        "batch_size": 1,
        "variable_h": 0.0001,
        "verbose": False,
        "nb_parallel": 16,
    },
    "attack_size": 128,
    "_target_": "deckard.attack.AttackConfig",
    "alias": "zoo",
}


safe_store(group="attack", name="attribute-bb", node=ATTACK_ATTRIBUTE_BB)
safe_store(group="attack", name="boundary", node=ATTACK_BOUNDARY)
safe_store(group="attack", name="fgm", node=ATTACK_FGM)
safe_store(group="attack", name="hsj", node=ATTACK_HSJ)
safe_store(group="attack", name="membership", node=ATTACK_MEMBERSHIP)
safe_store(group="attack", name="zoo", node=ATTACK_ZOO)

safe_store(group="search/attacks", name="attribute-bb", node=ATTACK_ATTRIBUTE_BB)
safe_store(group="search/attacks", name="boundary", node=ATTACK_BOUNDARY)
safe_store(group="search/attacks", name="fgm", node=ATTACK_FGM)
safe_store(group="search/attacks", name="hsj", node=ATTACK_HSJ)
safe_store(group="search/attacks", name="membership", node=ATTACK_MEMBERSHIP)
safe_store(group="search/attacks", name="zoo", node=ATTACK_ZOO)
