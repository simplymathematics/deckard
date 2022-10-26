dvc exp run -S +data.add_noise.X_train=.001,.01,.1,1,10 --queue
dvc exp run -S +data.add_noise.X_test=.001,.01,.1,1,10 --queue
dvc exp run -S +data.classification.n_features=5,10,20,30,50,100 -S +data.classification.n_informative=3,5,10,20,30,50,70,100 -S +data.classification.n_redundant=0,5,10,20,30,50 --queue
dvc exp run -S +model.mtype=linear,logistic -S +model.learning_rate=.000000001,.00000001,.0000001,.000001 -S +model.scale=.001,.01,.1,1,2,3,10 --queue
dvc queue start