#!/bin/bash

# This script is used to generate the attacks for the example.

# Fast Gradient Method
bash models.sh $@ ++attack.init.name=art.attacks.evasion.FastGradientMethod +attack.init.eps=.001,.01,.03,.3,.5,.8,.1 +attack.init.norm=inf,1,2 +attack.init.eps_step=.001,.003,.01 +attack.init.batch_size=1024 ++stage=attack  ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# # Projected Gradient Descent
bash models.sh $@ ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent +attack.init.eps=.01,.03,.3,.1 +attack.init.norm=inf,1,2 +attack.init.eps_step=.001,.003,.01 +attack.init.batch_size=1024 +attack.init.max_iter=10 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# DeepFool
bash models.sh $@ ++attack.init.name=art.attacks.evasion.DeepFool +attack.init.max_iter=10 +attack.init.batch_size=1024 +attack.init.nb_grads=10,100,1000 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# HopSkipJump
bash models.sh $@ ++attack.init.name=art.attacks.evasion.HopSkipJump +attack.init.max_iter=10  +attack.init.init_eval=10 +attack.init.norm=inf,2 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

#####################################################
# # PixelAttack
bash models.sh $@ ++attack.init.name=art.attacks.evasion.PixelAttack +attack.init.max_iter=10 +attack.init.th=1,4,16,64,256 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# ThresholdAttack
bash models.sh $@ ++attack.init.name=art.attacks.evasion.ThresholdAttack +attack.init.max_iter=10 +attack.init.th=1,4,16,64,256 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# # AdversarialPatch
bash models.sh $@ ++attack.init.name=art.attacks.evasion.AdversarialPatch +attack.init.max_iter=10 +attack.init.learning_rate=.5,5.0,50.0 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize +attack.init.patch_shape=[28,28,1] --multirun
#####################################################

# Carlini L0 Method
bash models.sh $@ ++attack.init.name=art.attacks.evasion.CarliniL0Method +attack.init.batch_size=1024 +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# Carlini L2 Method
bash models.sh $@ ++attack.init.name=art.attacks.evasion.CarliniL2Method +attack.init.batch_size=1024 +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun

# Carlini LInf Method
bash models.sh $@ ++attack.init.name=art.attacks.evasion.CarliniLInfMethod +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 ++stage=attack ++hydra.sweeper.study_name=attack ++hydra.sweeper.storage=sqlite:///attack.db ++hydra.sweeper.direction=minimize --multirun
