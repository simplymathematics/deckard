bash scripts/models.sh \
    stage=attack \
    attack=default \
    ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent \
    ++attack.init.eps=.001,.01,.1,.5,1 \
    ++attack.init.norm=2 \
    ++attack.init.eps_step=.001,.003,.01 \
    atk_name=PGD \
    ++attack.init.max_iter=1,5,10,50,100 $@