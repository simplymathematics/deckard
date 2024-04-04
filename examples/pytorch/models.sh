#!/bin/bash

# This script is used to generate the models for the sklearn example.

# # Default model
echo "python -m deckard.layers.optimise" $@ "--multirun" 
python -m deckard.layers.optimise  $@ --multirun

# This line generates the model and adds the FeatureSqueezing preprocessing defence.
python -m deckard.layers.optimise \
  ++model.art.preprocessor.name=art.defences.preprocessor.FeatureSqueezing \
  +model.art.preprocessor.bit_depth=4,8,16,32,64 \
  +model.art.preprocessor.clip_values=[0,255] \
  +model.art.preprocessor.apply_fit=True \
  +model.art.preprocessor.apply_predict=True \
  def_name=FSQ \
  $@ --multirun

# Gaussian Augmentation (Input)
python -m deckard.layers.optimise \
  ++model.art.preprocessor.name=art.defences.preprocessor.GaussianAugmentation \
  +model.art.preprocessor.sigma=.01,.1,.3,.5,1 \
  +model.art.preprocessor.ratio=.5 \
  +model.art.preprocessor.augmentation=False \
  +model.art.preprocessor.apply_fit=True \
  +model.art.preprocessor.apply_predict=True \
  def_name=Gauss-In \
  $@ --multirun

# Gaussian Noise (Output)
python -m deckard.layers.optimise \
  ++model.art.postprocessor.name=art.defences.postprocessor.GaussianNoise \
  ++model.art.postprocessor.scale=.01,.1,.3,.5,1 \
  ++model.art.postprocessor.apply_fit=True \
  ++model.art.postprocessor.apply_predict=True \
  def_name=Gauss-Out \
  $@ --multirun

# High Confidence
python -m deckard.layers.optimise \
    +model.art.postprocessor.name=art.defences.postprocessor.HighConfidence \
    +model.art.postprocessor.cutoff=.1,.3,.5,.9,.99 \
    ++model.art.postprocessor.apply_fit=True \
    ++model.art.postprocessor.apply_predict=True \
    def_name=Conf \
    $@ --multirun