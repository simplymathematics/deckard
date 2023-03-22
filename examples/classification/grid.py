import subprocess
param_dict = {
    "data.generate.n_features": [10, 100, 1000, 10000],
    "data.generate.n_samples": [110000],
    "data.sample.train_size": [1000000, 100000, 10000, 1000, 100],
    
    "model.init.C": [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 10000],
}

extra_dict = {
    "model.init.gamma": ["scale"],
    "model.init.degree": [2, 3, 4, 5],
    "model.init.coef0": [.0001,.001,.01,.1,1,10,100,1000,10000,10000],
}

big_list = []
for kernel in ["linear", "poly", "rbf"]:
    if str(kernel) == "linear":
        new_dict = param_dict.copy()
        
    elif str(kernel) == "poly":
        sub_dict = {}
        for key in ["model.init.gamma", "model.init.degree", "model.init.coef0"]:
            new_key = "++" + key
            sub_dict.update({new_key: extra_dict[key]})
        new_dict = {**param_dict, **sub_dict}
    elif str(kernel) == "rbf":
        sub_dict = {}
        for key in ["model.init.gamma", "model.init.coef0"]:
            new_key = "++" + key
            sub_dict.update({new_key: extra_dict[key]})
        new_dict = {**param_dict, **sub_dict}
    else:
        raise ValueError(f"Kernel {kernel} not recognized.")
    new_dict['model.init.kernel'] = [kernel]
    for key, value in new_dict.items():
        value = str(value).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        big_list.append(f"{key}={value} ")
    cmd = f"python -m deckard.layers.optimize +stage=models "
    for item in big_list:
        cmd += f" {item}"
    cmd += " --multirun"
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

# for train_size in ${TRAIN_SIZES[@]}; do
#     for n_samples in ${N_SAMPLES[@]}; do
#         for n_features in ${N_FEATURES[@]}; do
#             n_informative=$(( $n_features-1 ))
#             n_redundant=$(( 0 ))
#             n_repeated=$(( 0 ))
#             echo "Running experiment with n_features=${n_features} and n_samples=${n_samples} and a train size of ${train_size}"
#             echo "Running experiment with n_features=${n_features} and n_samples=${n_samples} and a train size of ${train_size}" >| log.txt
#             HYDRA_FULL_ERROR=1; python ../../deckard/layers/optimize.py --multirun \
#             data.generate.n_features=$n_features \
#             data.generate.n_redundant=$n_redundant \
#             data.generate.n_repeated=$n_repeated \
#             data.generate.n_informative=$n_informative \
#             data.generate.n_samples=$n_samples \
#             data.sample.train_size=$train_size \
#             model.init.kernel=linear \
#             model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
#             +stage=models "$@" \
#             >| logs/models/linear_features-${n_features}_samples-${n_samples}_train-${train_size}.log
#             echo "Linear Kernel Done" >> log.txt
#             python ../../deckard/layers/optimize.py --multirun \
#             data.generate.n_features=$n_features \
#             data.generate.n_redundant=$n_redundant \
#             data.generate.n_repeated=$n_repeated \
#             data.generate.n_informative=$n_informative \
#             data.generate.n_samples=$n_samples \
#             data.sample.train_size=$train_size \
#             model.init.kernel=rbf \
#             +model.init.gamma=scale \
#             model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
#             +stage=models "$@" >| logs/models/rbf_features-${n_features}_samples-${n_samples}_train-${train_size}.log
#             echo "RBF Kernel Done" >> log.txt
#             python ../../deckard/layers/optimize.py --multirun \
#             data.generate.n_features=$n_features \
#             data.generate.n_redundant=$n_redundant \
#             data.generate.n_repeated=$n_repeated \
#             data.generate.n_informative=$n_informative \
#             data.generate.n_samples=$n_samples \
#             data.sample.train_size=$train_size \
#             model.init.kernel=poly \
#             model.init.degree=1,2,3,4,5 
#             +model.init.gamma=scale \
#             +model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000  \
#             model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
#             +stage=models "$@" \>| logs/models/poly_features-${n_features}_samples-${n_samples}_train-${train_size}.log
#             echo "Poly Kernel Done" >> log.txt
#             echo "Successfully completed experiment ${i} of ${TOTAL}" >> log.txt
#         done
#     done
# done