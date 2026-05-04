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

ATTACK_MODEL_INVERSION = {
    "attack_type": "art.attacks.inference.model_inversion.mi_face.MIFace",
    "attack_size": 10,
    "attack_params": {
        "max_iter": 200,
        "threshold": 1.0,
        "initialization": "average",
        "split": "test",
    },
    "_target_": "deckard.attack.AttackConfig",
    "alias": "model_inversion",
}

ATTACK_DATABASE_RECONSTRUCTION = {
    "attack_type": "art.attacks.inference.reconstruction.DatabaseReconstruction",
    "attack_size": 1,
    "attack_params": {
        "split": "train",
        "missing_index": -1,
    },
    "_target_": "deckard.attack.AttackConfig",
    "alias": "database_reconstruction",
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
safe_store(
    group="attack",
    name="database-reconstruction",
    node=ATTACK_DATABASE_RECONSTRUCTION,
)
safe_store(group="attack", name="model-inversion", node=ATTACK_MODEL_INVERSION)
safe_store(group="attack", name="boundary", node=ATTACK_BOUNDARY)
safe_store(group="attack", name="fgm", node=ATTACK_FGM)
safe_store(group="attack", name="hsj", node=ATTACK_HSJ)
safe_store(group="attack", name="membership", node=ATTACK_MEMBERSHIP)
safe_store(group="attack", name="zoo", node=ATTACK_ZOO)

safe_store(
    group="search/attacks",
    name="attribute-bb",
    node=ATTACK_ATTRIBUTE_BB,
)
safe_store(
    group="search/attacks",
    name="database-reconstruction",
    node=ATTACK_DATABASE_RECONSTRUCTION,
)
safe_store(
    group="search/attacks",
    name="model-inversion",
    node=ATTACK_MODEL_INVERSION,
)
safe_store(group="search/attacks", name="boundary", node=ATTACK_BOUNDARY)
safe_store(group="search/attacks", name="fgm", node=ATTACK_FGM)
safe_store(group="search/attacks", name="hsj", node=ATTACK_HSJ)
safe_store(group="search/attacks", name="membership", node=ATTACK_MEMBERSHIP)
safe_store(group="search/attacks", name="zoo", node=ATTACK_ZOO)
