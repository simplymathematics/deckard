#!/bin/bash
paper_dir=~/ml_afr/
# set downstream to 2 or nothing
for d in */ ; do
    cd $d
    # run command and write to log file
    dvc repro --downstream clean -f  >| dvc_repro.log
    # dvc push
    cd -
done
# change to paper directory
cd $paper_dir
# run dvc repro and dvc push
dvc repro
dvc push
# change back to original directory
cd -
