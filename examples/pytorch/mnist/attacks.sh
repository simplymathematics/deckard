# #!/bin/bash

# # This script is used to generate the attacks for the example.

# Fast Gradient Method
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.FastGradientMethod ++attack.init.eps=.001,.01,.1,.5,1 ++attack.init.norm=inf,1,2 ++attack.init.eps_step=.001,.003,.01 ++attack.init.batch_size=1024 stage=attack  ++hydra.sweeper.study_name=fgm ++direction=maximize $@

# # Projected Gradient Descent
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent ++attack.init.eps=.001,.01,.1,.5,1 ++attack.init.norm=inf,1,2 ++attack.init.eps_step=.001,.003,.01 ++attack.init.batch_size=1024 ++attack.init.max_iter=10 stage=attack ++hydra.sweeper.study_name=pgd ++direction=maximize $@

# # DeepFool
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.DeepFool ++attack.init.max_iter=10 ++attack.init.batch_size=1024 ++attack.init.nb_grads=1,3,5,10 stage=attack ++hydra.sweeper.study_name=deep ++direction=maximize $@

# # HopSkipJump
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.HopSkipJump ++attack.init.max_iter=1,3,5,10  ++attack.init.init_eval=10 ++attack.init.norm=inf,2 stage=attack ++hydra.sweeper.study_name=hsj ++direction=maximize $@

# # #####################################################
# # PixelAttack
# bash models.sh attack=default  ++attack.init.name=art.attacks.evasion.PixelAttack ++attack.init.max_iter=10 ++attack.init.th=1,4,16,64,256 stage=attack ++hydra.sweeper.study_name=pixel ++direction=maximize $@

# ThresholdAttack
bash models.sh attack=default ++attack.init.name=art.attacks.evasion.ThresholdAttack ++attack.init.max_iter=10 ++attack.init.th=1,4,16,64,256 stage=attack ++hydra.sweeper.study_name=thresh ++direction=maximize $@

# # AdversarialPatch
# bash models.sh attack=default --attack.init.batch_size ++attack.init.name=art.attacks.evasion.AdversarialPatch ++attack.init.max_iter=10 ++attack.init.learning_rate=.5,5.0,50.0 stage=patch ++hydra.sweeper.study_name=attack ++direction=maximize ++attack.init.patch_shape=[1,28,28] $@
#####################################################

# # Carlini L0 Method
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.CarliniL0Method ++attack.init.batch_size=1024 ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 stage=cw0 ++hydra.sweeper.study_name=cw0 ++direction=maximize $@

# # Carlini L2 Method
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.CarliniL2Method ++attack.init.batch_size=1024 ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 stage=cw2 ++hydra.sweeper.study_name=cw2 ++direction=maximize $@

# # Carlini LInf Method
# bash models.sh attack=default ++attack.init.name=art.attacks.evasion.CarliniLInfMethod ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 stage=attack ++hydra.sweeper.study_name=cwinf ++direction=maximize $@

rm -rf output/models/*
