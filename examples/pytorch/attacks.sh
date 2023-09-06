# #!/bin/bash

# # This script is used to generate the attacks for the example.

# Fast Gradient Method
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.FastGradientMethod ++attack.init.eps=.001,.01,.03,.1,.2,.3,.5,.8,.1 ++attack.init.norm=inf,1,2 ++attack.init.eps_step=.001,.003,.01 ++attack.init.batch_size=1024 ++stage=attack  ++hydra.sweeper.study_name=fgm ++hydra.sweeper.direction=minimize $@

# Projected Gradient Descent
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent ++attack.init.eps=.01,.03,.3,.1 ++attack.init.norm=inf,1,2 ++attack.init.eps_step=.001,.003,.01 ++attack.init.batch_size=1024 ++attack.init.max_iter=10 ++stage=attack ++hydra.sweeper.study_name=pgd ++hydra.sweeper.direction=minimize $@

# DeepFool
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.DeepFool ++attack.init.max_iter=10 ++attack.init.batch_size=1024 ++attack.init.nb_grads=10,100,1000 ++stage=attack ++hydra.sweeper.study_name=deep ++hydra.sweeper.direction=minimize $@

# HopSkipJump
bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.HopSkipJump ++attack.init.max_iter=1,3,5,10  ++attack.init.init_eval=10 ++attack.init.norm=inf,2 ++stage=attack ++hydra.sweeper.study_name=hsj ++hydra.sweeper.direction=minimize $@

#####################################################
# PixelAttack
bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.PixelAttack ++attack.init.max_iter=10 ++attack.init.th=1,4,16,64,256 ++stage=attack ++hydra.sweeper.study_name=pixel ++hydra.sweeper.direction=minimize $@

# # ThresholdAttack
bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.ThresholdAttack ++attack.init.max_iter=10 ++attack.init.th=1,4,16,64,256 ++stage=attack ++hydra.sweeper.study_name=thresh ++hydra.sweeper.direction=minimize $@

# # AdversarialPatch
bash models.sh attack=hsj --attack.init.batch_size ++attack.init.name=art.attacks.evasion.AdversarialPatch ++attack.init.max_iter=10 ++attack.init.learning_rate=.5,5.0,50.0 ++stage=patch ++hydra.sweeper.study_name=attack ++hydra.sweeper.direction=minimize ++attack.init.patch_shape=[1,28,28] $@
#####################################################

# # Carlini L0 Method
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.CarliniL0Method ++attack.init.batch_size=1024 ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 ++stage=cw0 ++hydra.sweeper.study_name=attack ++hydra.sweeper.direction=minimize $@

# # # Carlini L2 Method
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.CarliniL2Method ++attack.init.batch_size=1024 ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 ++stage=cw2 ++hydra.sweeper.study_name=attack ++hydra.sweeper.direction=minimize $@

# # Carlini LInf Method
# bash models.sh attack=hsj ++attack.init.name=art.attacks.evasion.CarliniLInfMethod ++attack.init.max_iter=10 ++attack.init.confidence=.1,.9,.99 ++stage=attack ++hydra.sweeper.study_name=cwinf ++hydra.sweeper.direction=minimize $@

rm -rf output/models/*
