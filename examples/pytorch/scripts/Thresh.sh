bash scripts/models.sh \
    stage=attack \
    attack=default \
    ++attack.init.name=art.attacks.evasion.ThresholdAttack \
    ~attack.init.batch_size \
    ++attack.init.th=1,4,16,64,256 \
    atk_name=Thresh $@