#!/bin/bash

# This script is used to generate the models for the sklearn example.

# # Default model
echo "python -m deckard.layers.optimise " $@ "--multirun"
python -m deckard.layers.optimise  $@ --multirun

# # This line generates the model and adds the FeatureSqueezing preprocessing defence.
python -m deckard.layers.optimise ++model.art.preprocessor.name=art.defences.preprocessor.FeatureSqueezing +model.art.preprocessor.params.bit_depth=4,8,16,32,64 +model.art.preprocessor.params.clip_values=[0,1]  $@ --multirun

# # Gaussian Augmentation (Input)
python -m deckard.layers.optimise ++model.art.preprocessor.name=art.defences.preprocessor.GaussianAugmentation +model.art.preprocessor.params.sigma=.01,.1,.3,.5,1 +model.art.preprocessor.params.ratio=.1,.5,1 +model.art.preprocessor.params.augmentation=False  $@ --multirun

# # # Gaussian Noise (Output)
python -m deckard.layers.optimise ++model.art.postprocessor.name=art.defences.postprocessor.GaussianNoise ++model.art.postprocessor.params.scale=.01,.1,.3,.5,1  $@ --multirun

# # # High Confidence
python -m deckard.layers.optimise +model.art.postprocessor.name=art.defences.postprocessor.HighConfidence +model.art.postprocessor.params.cutoff=.1,.3,.5,.9,.99  $@ --multirun
