# #!/bin/bash
# # This script runs the attacks on the classification models.

# # For each model in the 'best' directory, run the attacks on it.
# BEST=$(find best_models -name '*.yaml')
# MAX_ITERS=(1 10 100 1000)
# TOTAL=$((4 * ${#MAX_ITERS[@]}))
# i=0
# # iterate over the models
# for model in ${BEST[@]}
#     do
#     for max_iter in ${MAX_ITERS[@]}
#         do
#         i=$(($i+1))
#         cp $model params.yaml
#         echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
#         HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
#         --multirun hydra.sweeper.study_name=attacks \
#         hydra.sweeper.n_trials=100 \
#         hydra.sweeper.direction=minimize \
#         +stage=attack \
#         +queue=attack_queue \
#         attack.init.eps=.001,.001,.01,.1,1 \
#         attack.init.eps_step=.0001,.001,.001,.01,.1 \
#         attack.init.batch_size=1,10,100 \
#         attack.init.max_iter=$max_iter \
#         attack.generate.attack_size=100 \
#         data.sample.train_size=10000 \
#         data.generate.n_features=100
#     done
# done


#!/bin/bash
# This script runs the attacks on the classification models.

# For each model in the 'best' directory, run the attacks on it.
BEST=$(find best_models -name '*.yaml')
MAX_ITERS=(1000 100 10 1)
TOTAL=$((3 * ${#MAX_ITERS[@]}))
i=0
mkdir -p logs/attacks/best_models

# iterate over the models
for max_iter in ${MAX_ITERS[@]}
    do
        
        
        i=$(($i+1))
        echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
        HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
        --multirun hydra.sweeper.study_name=attacks \
        +queue=attack_queue \
        model.init.name=sklearn.svm.SVC \
        ++model.init.kernel=poly \
        model.init.C=10.0 \
        +model.init.coef0=0.0001 \
        model.init.degree=4 \
        model.init.max_iter=1000 \
        hydra.sweeper.n_trials=100 \
        hydra.sweeper.direction=minimize \
        +stage=attack \
        attack.init.eps=.001,.001,.01,.1,1 \
        attack.init.eps_step=.0001,.001,.001,.01,.1 \
        attack.init.batch_size=1,10,100 \
        attack.init.max_iter=$max_iter \
        attack.generate.attack_size=100 \
        ++data.sample.train_size=10000 \
        ++data.generate.n_features=100 >| logs/attacks/poly_${i}.log

        i=$(($i+1))
        echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
        HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
        --multirun hydra.sweeper.study_name=attacks \
        +queue=attack_queue \
        model.init.name=sklearn.svm.SVC \
        ++model.init.kernel=rbf \
        model.init.C=10000.0 \
        +model.init.coef0=1.0 \
        model.init.max_iter=1000 \
        hydra.sweeper.n_trials=100 \
        hydra.sweeper.direction=minimize \
        +stage=attack \
        attack.init.eps=.001,.001,.01,.1,1 \
        attack.init.eps_step=.0001,.001,.001,.01,.1 \
        attack.init.batch_size=1,10,100 \
        attack.init.max_iter=$max_iter \
        attack.generate.attack_size=100 \
        ++data.sample.train_size=10000 \
        ++data.generate.n_features=100 >| logs/attacks/rbf_${i}.log
        
        i=$(($i+1))
        echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
        HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
        --multirun hydra.sweeper.study_name=attacks \
        +queue=attack_queue \
        model.init.name=sklearn.svm.SVC \
        model.init.kernel=linear \
        model.init.C=0.01 \
        model.init.max_iter=1000 \
        hydra.sweeper.n_trials=100 \
        hydra.sweeper.direction=minimize \
        +stage=attack \
        attack.init.eps=.001,.001,.01,.1,1 \
        attack.init.eps_step=.0001,.001,.001,.01,.1 \
        attack.init.batch_size=1,10,100 \
        attack.init.max_iter=$max_iter \
        attack.generate.attack_size=100 \
        data.sample.train_size=10000 \
        data.generate.n_features=100 >| logs/attacks/linear_${i}.log
        
        # i=$(($i+1))
        # echo "Running attacks on ${model} with max_iter=${max_iter}-- ${i} of ${TOTAL}."
        # HYDRA_FULL_ERROR=1 python ../../deckard/layers/optimize.py \
        # --multirun hydra.sweeper.study_name=attacks \
        # +queue=attack_queue \
        # model.init.name=sklearn.svm.SVC \
        # model.init.kernel=sigmoid \
        # model.init.C=1.0 \
        # model.init.coef0=0.0001 \
        # model.init.max_iter=1000 \
        # hydra.sweeper.n_trials=100 \
        # hydra.sweeper.direction=minimize \
        # +stage=attack \
        # attack.init.eps=.001,.001,.01,.1,1 \
        # attack.init.eps_step=.0001,.001,.001,.01,.1 \
        # attack.init.batch_size=1,10,100 \
        # attack.init.max_iter=$max_iter \
        # attack.generate.attack_size=100 \
        # data.sample.train_size=10000 \
        # data.generate.n_features=100 >| logs/attacks/sigmoid_${i}.log
    done
done
