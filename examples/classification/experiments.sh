N_FEATURES=( 10 100 1000 10000 )
N_SAMPLES=( 10 100 1000 10000 100000 )
i=$(( 0 ))
for n_features in ${N_FEATURES[@]}; do
    for n_samples in ${N_SAMPLES[@]}; do
        n_informative=$(( $n_features - 1 ))
        echo "Running experiment with n_features=${n_features} and n_samples=${n_samples}"
        echo "Running experiment with n_features=${n_features} and n_samples=${n_samples}" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=linear model.init.C=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000 
        echo "Linear Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=rbf +model.init.gamma=auto model.init.C=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000 
        echo "RBF Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=poly +model.init.degree=1,2,3,4,5 +model.init.gamma=auto +model.init.coef0=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000   model.init.C=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000 
        echo "Poly Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=sigmoid +model.init.gamma=auto +model.init.coef0=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000   model.init.C=.000000001,.000001,.001,.01,.1,1,10,100,1000,1000000,1000000000 
        i=$(( $i + 1))
        echo "Sigmoid Kernel Done" >> log.txt
        echo "Successfully completed experiment ${i} of ${#N_FEATURES[@]} * ${#N_SAMPLES[@]}" >> log.txt
    done
done