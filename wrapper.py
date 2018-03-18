import sample_run as sp
import collect_training_data as ct
import dnn_training as dt
import dnn_run_model as dr

#sample run
sp.sample_run_env()

#function to collect the data
t_data = ct.train_data_collect()

print ("Collect the data and training the model.")
#function to train the NN
mdl = dt.dnn_train(t_data)

print ("Sample runs without the model")
sp.sample_run_env(printstats = True)

print ("Model controlling the cart")
#model controlling the cart.
dr.dnn_run(mdl, games = 5)