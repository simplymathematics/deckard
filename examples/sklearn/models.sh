#!/bin/bash

# This script is used to generate the models for the sklearn example.
python -m deckard.layers.optimise  model=linear  ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 $@

python -m deckard.layers.optimise ++data=small,medium,large,very_large model=poly ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 ++data.init.degree=1,2,3,4,5 ++model.init.Coef0=.00001,.0001,.01,.1,1,10,1000,1000000 $@

python -m deckard.layers.optimise ++data=small,medium,large,very_large model=rbf ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 $@