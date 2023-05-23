#!/bin/bash

# This script is used to generate the models for the sklearn example.


# This line generates the model and adds the FeatureSqueezing preprocessing defence.
python -m deckard.layers.optimise +model.art.preprocessor.name=art.defences.preprocessor.FeatureSqueezing +model.art.preprocessor.params.bit_depth=4,8,16,32,64 +stage=train +optimizers=accuracy ++model.art.preprocessor.clip_values=[0,1]--multirun

# Gaussian Augmentation (Input)
python -m deckard.layers.optimise +model.art.preprocessor.name=art.defences.preprocessor.GaussianAugmentation +model.art.preprocessor.params.sigma=.01,.1,.2,.3,.5,1 +model.art.preprocessor.params.ratio=.1,.5,1 +stage=train +optimizers=accuracy --multirun

# # Spatial Smoothing
# python -m deckard.layers.optimise +model.art.preprocessor.name=art.defences.preprocessor.SpatialSmoothing +model.art.preprocessor.params.window_size=2,3,4 +stage=train +optimizers=accuracy --multirun

# # Total Variance Minimisation
# python -m deckard.layers.optimise +model.art.preprocessor.name=art.defences.preprocessor.TotalVarMin +model.art.preprocessor.params.prob=.001,.01,.1 +model.art.preprocessor.params.norm=1,2,3 +model.art.preprocessor.params.lambda=.05,.5,.95 +model.art.preprocessor.params.max_iter=100 +stage=train +optimizers=accuracy --multirun

# # Gaussian Noise (Output)
# python -m deckard.layers.optimise +model.art.postprocessor.name=art.defences.postprocessor.GaussianNoise +model.art.postprocessor.params.scale=.1,.9,.999 +stage=train +optimizers=accuracy --multirun

# # High Confidence
# python -m deckard.layers.optimise +model.art.postprocessor.name=art.defences.postprocessor.HighConfidence +model.art.postprocessor.params.cutoff=.1,.5,.9,.99 +stage=train +optimizers=accuracy --multirun

# # Rounded (Output)
# python -m deckard.layers.optimise +model.art.postprocessor.name=art.defences.postprocessor.Rounded +model.art.postprocessor.params.decimals=1,2,4,8 +stage=train +optimizers=accuracy --multirun
