bash scripts/models.sh \
    stage=attack \
    attack=default \
    ++attack.init.name=art.attacks.evasion.HopSkipJump \
    ++attack.init.max_iter=1,3,5,10,15 \
    ++attack.init.init_eval=3 \
    ++attack.init.max_eval=10 \
    ++attack.init.norm=2 \
    atk_name=HSJ $@
