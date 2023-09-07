#!/bin/bash
sizes=( small medium large very_large )
for size in $sizes; do
# This script is used to generate the models for the sklearn example.
    python -m deckard.layers.optimise data=$size model=linear  ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 $@

    python -m deckard.layers.optimise data=$size model=poly ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 ++model.init.degree=1,2,3,4,5 ++model.init.Coef0=.00001,.0001,.01,.1,1,10,1000,1000000 $@

    python -m deckard.layers.optimise data=$size model=rbf ++data.sample.random_state=0,1,2,3,4,5 ++model.init.C=.00001,.0001,.01,.1,1,10,1000,1000000 $@
done
