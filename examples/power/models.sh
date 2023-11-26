#!/bin/bash

# This script is used to generate the models for the sklearn example.

# # Default model
echo "python -m deckard.layers.optimise " $@ "--multirun"
python -m deckard.layers.optimise  $@ --multirun