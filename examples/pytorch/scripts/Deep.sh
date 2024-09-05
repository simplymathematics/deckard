bash scripts/models.sh \
    stage=attack \
    attack=default \
    ++attack.init.name=art.attacks.evasion.DeepFool \
    ++attack.init.max_iter=10 \
    ++attack.init.batch_size=4096 \
    ++attack.init.nb_grads=1,3,5,8,10 \
    atk_name=Deep $@
