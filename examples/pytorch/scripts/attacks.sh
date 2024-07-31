# #!/bin/bash

# # This script is used to generate the attacks for the example.

# Fast Gradient Method
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.FastGradientMethod \
#     ++attack.init.eps=.001,.01,.1,.5,1 \
#     ++attack.init.norm=2 \
#     atk_name=FGM  $@
# #####################################################
# # Projected Gradient Descent
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent \
#     ++attack.init.eps=.001,.01,.1,.5,1 \
#     ++attack.init.norm=2 \
#     ++attack.init.eps_step=.001,.003,.01 \
#     atk_name=PGD \
#     ++attack.init.max_iter=1,5,10,50,100 $@
# # #####################################################
# # DeepFool
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.DeepFool \
#     ++attack.init.max_iter=10 \
#     ++attack.init.batch_size=4096 \
#     ++attack.init.nb_grads=1,3,5,8,10 \
#     atk_name=Deep $@
# # #####################################################
# # HopSkipJump
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.HopSkipJump \
#     ++attack.init.max_iter=1,3,5,10,15 \
#     ++attack.init.init_eval=3 \
#     ++attack.init.max_eval=10 \
#     ++attack.init.norm=2 \
#     atk_name=HSJ $@
# #####################################################
# # # PixelAttack
# bash models.sh \
#     stage=attack \
#     attack=default  \
#     ++attack.init.name=art.attacks.evasion.PixelAttack \
#     ~attack.init.batch_size \
#     ++attack.init.th=1,4,16,64,256 \
#     atk_name=Pixel $@
# # #####################################################
# # ThresholdAttack
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.ThresholdAttack \
#     ~attack.init.batch_size \
#     ++attack.init.th=1,4,16,64,256 \
#     atk_name=Thresh $@
# #####################################################
# # ZooAttack
# bash models.sh \
#     stage=attack \
#     attack=default \
#     ++attack.init.name=art.attacks.evasion.ZooAttack \
#     ++attack.init.binary_search_steps=1,2,3,5,10 \
#     ++attack.init.abort_early=True \
#     atk_name=Zoo $@
# ####################################################
# # Carlini L0 Method
# bash models.sh \
#     attack=default \
#     stage=attack \
#     ++attack.init.name=art.attacks.evasion.CarliniL0Method \
#     ++attack.init.max_iter=10 \
#     ++attack.init.confidence=.1,.3,.5,.9,.99 \
#     atk_name=Patch $@
# # Carlini L2 Method
# bash models.sh \
#     attack=default \
#     stage=attack \
#     ++attack.init.name=art.attacks.evasion.CarliniL2Method \
#     ++attack.init.confidence=.1,.3,.5,.9,.99 \
#     atk_name=CW2 $@
# # # Carlini L2 Method
# bash models.sh \
#     attack=default \
#     stage=attack \
#     ++attack.init.name=art.attacks.evasion.CarliniL2Method \
#     ++attack.init.confidence=.1,.9,.99 \
#     atk_name=Patch $@
# rm -rf mnist/models/*
# rm -rf cifar/models/*
# rm -rf cifar100/models/*
