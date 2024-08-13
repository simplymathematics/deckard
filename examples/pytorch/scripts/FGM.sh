bash scripts/models.sh \
    stage=attack \
    attack=default \
    ++attack.init.name=art.attacks.evasion.FastGradientMethod \
    ++attack.init.eps=.001,.01,.1,.5,1 \
    ++attack.init.norm=2 \
    atk_name=FGM  $@
