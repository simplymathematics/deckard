N_FEATURES=( 10 100 1000 10000 )
N_SAMPLES=( 100 1000 10000 )
i=$(( 0 ))
for n_features in ${N_FEATURES[@]}; do
    for n_samples in ${N_SAMPLES[@]}; do
        n_informative=$(( $n_features - 1 ))
        echo "Running experiment with n_features=${n_features} and n_samples=${n_samples}"
        echo "Running experiment with n_features=${n_features} and n_samples=${n_samples}" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=linear 
        echo "Linear Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=rbf +model.init.gamma=scale 
        echo "RBF Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=poly +model.init.degree="range(1, 5)" +model.init.gamma=scale +model.init.coef0="tag(log, range(-1e3, 1e3, 10))" 
        echo "Poly Kernel Done" >> log.txt
        python ../../deckard/layers/optimize.py --multirun data.generate.n_features=$n_features data.generate.n_informative=$n_informative data.generate.n_samples=$n_samples model.init.kernel=sigmoid +model.init.gamma=scale +model.init.coef0="tag(log, range(-1e3, 1e3, 10))" 
        i=$(( $i + 1))
        echo "Sigmoid Kernel Done" >> log.txt
        echo "Successfully completed experiment ${i} of ${#N_FEATURES[@]} * ${#N_SAMPLES[@]}" >> log.txt
    done
done