#!/bin/bash

# This script is used to generate the attacks for the sklearn example.

# Fast Gradient Method
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.FastGradientMethod +attack.init.eps=.01,.03,.3,.1 +attack.init.norm=inf,1,2 +attack.init.eps_step=.001,.003,.01 +attack.init.batch_size=100 +stage=attack --multirun

# Projected Gradient Descent
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.ProjectedGradientDescent +attack.init.eps=.01,.03,.3,.1 +attack.init.norm=inf,1,2 +attack.init.eps_step=.001,.003,.01 +attack.init.batch_size=100 +attack.init.max_iter=10 +stage=attack --multirun

# Carlini L0 Method
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.CarliniL0Method +attack.init.batch_size=100 +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 +stage=attack --multirun

# Carlini L2 Method
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.CarliniL2Method +attack.init.batch_size=100 +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 +stage=attack --multirun

# Carlini LInf Method
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.CarliniLInfMethod +attack.init.max_iter=10 +attack.init.confidence=.1,.9,.99 +stage=attack --multirun

# DeepFool
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.DeepFool +attack.init.max_iter=10 +attack.init.batch_size=100 +attack.init.nb_grads=10,100,1000 +stage=attack --multirun

# HopSkipJump
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.HopSkipJump +attack.init.max_iter=10 +attack.init.max_eval=10 +attack.init.init_eval=10 +attack.init.norm=inf,2 +stage=attack --multirun

# PixelAttack
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.PixelAttack +attack.init.max_iter=10 +attack.init.th=.5,.9,.99 +stage=attack --multirun

# ThresholdAttack
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.ThresholdAttack +attack.init.max_iter=10 +attack.init.th=.5,.9,.99 +stage=attack --multirun

# AdversarialPatch
python -m deckard.layers.optimise +attack.init.name=art.attacks.evasion.AdversarialPatch +attack.init.max_iter=10 +attack.init.learning_rate=.5,5.0,50.0 +stage=attack --multirun
