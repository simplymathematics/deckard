from deckard.attack.base import AttackConfig


def test_with_targeted_attack_labels_for_targeted_evasion_uses_target_token():
    cfg = AttackConfig(
        attack_type="art.attacks.evasion.HopSkipJump",
        attack_params={"targeted": True},
    )

    scores = cfg._with_targeted_attack_labels(
        {"evasion_accuracy": 0.75, "evasion_success": 0.2},
        "evasion",
    )

    assert scores["targeted_evasion_accuracy"] == 0.75
    assert scores["targeted_evasion_success"] == 0.2


def test_with_targeted_attack_labels_for_poisoning_uses_class_target_token():
    cfg = AttackConfig(
        attack_type="art.attacks.poisoning.PoisoningAttackSVM",
        attack_params={"class_target": 7},
    )

    scores = cfg._with_targeted_attack_labels(
        {"poisoned_accuracy": 0.4, "benign_accuracy": 0.9},
        "poisoning",
    )

    assert scores["7_evasion_accuracy"] == 0.4
    assert scores["7_evasion_benign_accuracy"] == 0.9
