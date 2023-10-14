

mkdir -p logs/models
HYDRA_FULL_ERROR=1; python -m deckard.layers.optimise  \
++data.sample.test_size=1000 \
model.init.kernel=linear \
model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
++hydra.sweeper.storage=sqlite:///model.db \
++hydra.sweeper.study_name=linear "$@" --multirun \
>| logs/models/linear.log
echo "Linear Kernel Done" >> model_log.txt
# Runs the poly kernel
python -m deckard.layers.optimise \
++data.sample.test_size=1000 \
model.init.kernel=rbf \
+model.init.gamma=scale \
model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
+model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
++hydra.sweeper.storage=sqlite:///model.db \
++hydra.sweeper.study_name=rbf "$@" --multirun \
>| logs/models/rbf.log
echo "RBF Kernel Done" >> model_log.txt
# Runs the poly kernel
python -m deckard.layers.optimise \
++data.sample.test_size=1000 \
model.init.kernel=poly \
+model.init.degree=1,2,3,4,5 \
+model.init.gamma=scale \
+model.init.coef0=.0001,.001,.01,.1,1,10,100,1000,10000,10000  \
model.init.C=.0001,.001,.01,.1,1,10,100,1000,10000,10000 \
++hydra.sweeper.storage=sqlite:///model.db \
++hydra.sweeper.study_name=poly "$@" --multirun \
>| logs/models/poly.log
echo "Poly Kernel Done" >> model_log.txt
echo "Successfully completed experiment ${i} of ${TOTAL}" >> model_log.txt