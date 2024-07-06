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
  def_name=FSQ \
  $@ --multirun

rm -rf mnist/models
rm -rf cifar/models
rm -rf cifar100/models

# Gaussian Augmentation (Input)
python -m deckard.layers.optimise \
  ++model.art.preprocessor.name=art.defences.preprocessor.GaussianAugmentation \
  +model.art.preprocessor.sigma=.01,.1,.3,.5,1 \
  +model.art.preprocessor.ratio=.5 \
  +model.art.preprocessor.augmentation=False \
  def_name=Gauss-In \
  $@ --multirun

rm -rf mnist/models
rm -rf cifar/models
rm -rf cifar100/model

# Gaussian Noise (Output)
python -m deckard.layers.optimise \
  ++model.art.postprocessor.name=art.defences.postprocessor.GaussianNoise \
  ++model.art.postprocessor.scale=.01,.1,.3,.5,1 \
  ++model.art.postprocessor.apply_fit=True \
  def_name=Gauss-Out \
  $@ --multirun

rm -rf mnist/models
rm -rf cifar/models
rm -rf cifar100/models

# High Confidence
python -m deckard.layers.optimise \
    +model.art.postprocessor.name=art.defences.postprocessor.HighConfidence \
    +model.art.postprocessor.cutoff=.1,.3,.5,.9,.99 \
    def_name=Conf \
    $@ --multirun

rm -rf mnist/models
rm -rf cifar/models
rm -rf cifar100/models

