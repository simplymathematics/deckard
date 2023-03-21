N_FEATURES=( 10 100 1000 10000)
TRAIN_SIZES=( 100 1000 10000 100000 1000000 )
# TRAIN_SIZES=( 2000 )
# N_FEATURES=( 100 )
N_SAMPLES=( 110000 )
TOTAL=$(( ${#N_FEATURES[@]} * ${#N_SAMPLES[@]} * ${#TRAIN_SIZES[@]} ))
i=$(( 0 ))
mkdir -p logs/models
for train_size in ${TRAIN_SIZES[@]}; do
    for n_samples in ${N_SAMPLES[@]}; do
        for n_features in ${N_FEATURES[@]}; do
            i=$(( i + 1 ))
            n_informative=$(( $n_features-1 ))
            echo "Running experiment with n_features=${n_features} and n_samples=${n_samples} and a train size of ${train_size}"
            echo "Running experiment with n_features=${n_features} and n_samples=${n_samples} and a train size of ${train_size}" >| log.txt
            HYDRA_FULL_ERROR=1; python ../../deckard/layers/optimize.py  \
            data.generate.n_features=$n_features \
            data.generate.n_informative=$n_informative \
            data.generate.n_samples=$n_samples \
            data.sample.train_size=$train_size \
            model.init.kernel=linear \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            +stage=models  "$@" --multirun \
            >| logs/models/linear_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "Linear Kernel Done" >> log.txt
            i=$(( i + 1 ))
            python ../../deckard/layers/optimize.py \
            data.generate.n_features=$n_features \
            data.generate.n_informative=$n_informative \
            data.generate.n_samples=$n_samples \
            data.sample.train_size=$train_size \
            model.init.kernel=rbf \
            +model.init.gamma=scale \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            +model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000  \
            +stage=models  "$@" --multirun \
            >| logs/models/rbf_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "RBF Kernel Done" >> log.txt
            i=$(( i + 1 ))
            python ../../deckard/layers/optimize.py \
            data.generate.n_features=$n_features \
            data.generate.n_informative=$n_informative \
            data.generate.n_samples=$n_samples \
            data.sample.train_size=$train_size \
            model.init.kernel=poly \
            model.init.degree=1,2,3,4,5 \
            +model.init.gamma=scale \
            +model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000  \
            model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
            +stage=models  "$@" --multirun \
            >| logs/models/poly_features-${n_features}_samples-${n_samples}_train-${train_size}.log
            echo "Poly Kernel Done" >> log.txt
            echo "Successfully completed experiment ${i} of ${TOTAL}" >> log.txt
        done
    done
done