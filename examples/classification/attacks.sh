
#!/bin/bash
# This script runs the attacks on the classification models.

# For each model in the 'best' directory, run the attacks on it.
BEST=$(find best_models -name '*params.yaml')
# MAX_ITERS=(1000 100 10 1)
MAX_ITERS=(1)
TOTAL=$((3 * ${#MAX_ITERS[@]}))
i=0
mkdir -p logs/attacks/best_models

# iterate over the models
for max_iter in ${MAX_ITERS[@]}
    do
      for model in ${BEST[@]}
        do
            i=$(($i+1))
            echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
            HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
            --multirun hydra.sweeper.study_name=attacks \
            hydra.sweeper.direction=minimize \
            +stage=attacks \
            attack.init.eps=.001,.001,.01,.1,1 \
            attack.init.eps_step=.0001,.001,.001,.01,.1 \
            attack.init.batch_size=1,10,100 \
            attack.init.max_iter=$max_iter \
            attack.generate.attack_size=100 \
            +files.attack_file="attack_file.pkl" \ # this is the file that will be saved and must be added to the base config
            ++files.attack_score_dict_file="attack_scores.json" \ # This is the scores file that will be saved and must will override the base config
            --config-name params.yaml \
            --config-dir $( dirname  ${model})
            >| logs/attacks/"$(basename $(dirname $model))"_$i.log
  done
done
