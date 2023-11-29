#!/bin/bash
MODEL_CONFIGS=$(ls conf/model/best_*.yaml)
CONFIG_NAMES=$(ls conf/model/best_*.yaml | cut -d'/' -f3 | cut -d'.' -f1)
TOTAL=$(( ${#CONFIG_NAMES[@]} ))
i=$(( 0 ))
mkdir -p logs/attacks/
for model_config in $CONFIG_NAMES; do
    kernel_name=$(echo $model_config | cut -d'_' -f2)
    i=$(( i + 1 ))
    if [ $model_config == "default" ]; then
        continue
    fi
    HYDRA_FULL_ERROR=1 python -m deckard.layers.optimise \
    ++model.init.kernel=kernel_name \
    ++stage=attack \
    ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent \
    ++attack.init.norm=1,2,inf \
    ++attack.init.eps_step=.001,.01,.1,.3,.5,1 \
    ++attack.init.batch_size=1,10,50,100 \
    ++attack.init.eps=.001,.01,.1,.3,.5,1 \
    ++attack.init.max_iter=1,10,100,1000 \
    ++hydra.sweeper.study_name=$model_config \
    ++attack.attack_size=100 \
    model=$model_config $@ --multirun >> logs/attacks/$model_config.log
    echo "Successfully completed model $model_config" >> attack_log.txt
done

# Other attacks listed below
# PGD
# bash models.sh ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent ++attack.init.norm=1,2,inf ++attack.init.eps_step=.001,.01,.1,.3,.5,1 ++attack.init.batch_size=1,10,50,100 ++attack.init.eps=.001,.01,.1,.3,.5,1 $@

# # Carlini L0
# bash models.sh ++attack.init.name=art.attacks.evasion.CarliniL0Method ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@

# # Carlini L2
# bash models.sh ++attack.init.name=art.attacks.evasion.CarliniL2Method ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@

# # Carlini LInf
# bash models.sh ++attack.init.name=art.attacks.evasion.CarliniLInfMethod ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@

# # DeepFool
# bash models.sh ++attack.init.nb_grads=1,3,5,10 ++attack.init.name=art.attacks.evasion.DeepFool ++attack.init.batch_size=100 $@

# #Threshold Attack
# bash models.sh ++attack.init.name=art.attacks.evasion.ThresholdAttack +attack.init.th=1,4,16,64,255 ++attack.init.batch_size=100 $@

# #Pixel Attack
# bash models.sh ++attack.init.name=art.attacks.evasion.PixelAttack +attack.init.th=1,4,16,64,255 ++attack.init.batch_size=100 $@

# #Adversarial Patch
# bash models.sh ++attack.init.name=art.attacks.evasion.AdversarialPatch +attack.init.scale_max=.1,.2,.3,.5,.8,.9,.99 ++attack.init.batch_size=100 $@

# #Hop Skip Jump
# bash models.sh ++attack.init.name=art.attacks.evasion.HopSkipJump ++attack.init.batch_size=100 $@
