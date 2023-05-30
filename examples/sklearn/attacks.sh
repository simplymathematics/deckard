
# FGM
bash models.sh ++attack.init.name=art.attacks.evasion.FastGradientMethod ++attack.init.norm=1,2,inf ++attack.init.eps=.001,.01,.1,.3,.5,1 ++attack.init.eps_step=.001,.01,.1,.3,.5,1 ++attack.init.batch_size=100 $@ --multirun
# PGD
bash models.sh ++attack.init.name=art.attacks.evasion.ProjectedGradientDescent ++attack.init.norm=1,2,inf ++attack.init.eps_step=.001,.01,.1,.3,.5,1 $@ ++attack.init.batch_size=100  --multirun

# Carlini L0
bash models.sh ++attack.init.name=art.attacks.evasion.CarliniL0Method ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@ --multirun

# Carlini L2
bash models.sh ++attack.init.name=art.attacks.evasion.CarliniL2Method ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@ --multirun

# Carlini LInf
bash models.sh ++attack.init.name=art.attacks.evasion.CarliniLInfMethod ++attack.init.confidence=1,4,16,64,256 ++attack.init.confidence=1,4,16,64,256 ++attack.init.batch_size=100 $@ --multirun

# DeepFool
bash models.sh ++attack.init.nb_grads=1,3,5,10 ++attack.init.name=art.attacks.evasion.DeepFool ++attack.init.batch_size=100 $@ --multirun

#Threshold Attack
bash models.sh ++attack.init.name=art.attacks.evasion.ThresholdAttack +attack.init.th=1,4,16,64,255 ++attack.init.batch_size=100 $@ --multirun

#Pixel Attack
bash models.sh ++attack.init.name=art.attacks.evasion.PixelAttack +attack.init.th=1,4,16,64,255 ++attack.init.batch_size=100 $@ --multirun

#Adversarial Patch
bash models.sh ++attack.init.name=art.attacks.evasion.AdversarialPatch +attack.init.scale_max=.1,.2,.3,.5,.8,.9,.99 ++attack.init.batch_size=100 $@ --multirun

#Hop Skip Jump
bash models.sh ++attack.init.name=art.attacks.evasion.HopSkipJump ++attack.init.batch_size=100 ++attack.init.batch_size=100 $@ --multirun