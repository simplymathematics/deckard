# N_FEATURES=( 10 100 1000 10000 10000 100000 1000000)
# TRAIN_SIZES=( 10 100 1000 )
TRAIN_SIZES=( 10000 )
N_FEATURES=( 100 )
N_SAMPLES=( 1010000 )
TOTAL=$(( ${#N_FEATURES[@]} * ${#N_SAMPLES[@]} * ${#TRAIN_SIZES[@]} ))
i=$(( 0 ))
mkdir -p logs/models
for train_size in ${TRAIN_SIZES[@]}; do
    for n_samples in ${N_SAMPLES[@]}; do
        for n_features in ${N_FEATURES[@]}; do
            i=$(( i + 1 ))
            echo "Running experiment ${i} of ${TOTAL}"
            # Keeps a meta log of the experiments
            echo "Running experiment ${i} of ${TOTAL}" >> log.txt
            echo "Running experiment with n_features=${n_features} and n_samples=${n_samples} and a train size of ${train_size}" >> log.txt
            # Runs the linear kernel, tries to find the best C value
            HYDRA_FULL_ERROR=1; python -m deckard.layers.optimise  \
            ++data.generate.n_features=$n_features \
            ++data.generate.n_samples=$n_samples \
            ++data.sample.train_size=$train_size \
            model.init.kernel=linear \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            ++hydra.sweeper.study_name=linear_${n_features}_${train_size} "$@" --multirun \
            # Keeps a log of the output for each experiment
            >| logs/models/linear_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "Linear Kernel Done" >> log.txt
            # Runs the poly kernel
            python -m deckard.layers.optimise \
            ++data.generate.n_features=$n_features \
            ++data.generate.n_samples=$n_samples \
            ++data.sample.train_size=$train_size \
            model.init.kernel=rbf \
            +model.init.gamma=scale \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            +model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            ++hydra.sweeper.study_name=rbf_${n_features}_${train_size} "$@" --multirun \
            >| logs/models/rbf_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "RBF Kernel Done" >> log.txt
            # Runs the poly kernel
            python -m deckard.layers.optimise \
            ++data.generate.n_features=$n_features \
            ++data.generate.n_samples=$n_samples \
            ++data.sample.train_size=$train_size \
            model.init.kernel=poly \
            +model.init.degree=1,2,3,4,5 \
            +model.init.gamma=scale \
            +model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000  \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            ++hydra.sweeper.study_name=poly_${n_features}_${train_size} "$@" --multirun \
            >| logs/models/poly_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "Poly Kernel Done" >> log.txt
            echo "Successfully completed experiment ${i} of ${TOTAL}" >> log.txt
        done;
    done;
done;