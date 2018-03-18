# openai_gym_dnn
Balancing a pole on a cart using Tensorflow TFLearn and OpenAI Gym.

This project is developed in Python and uses TFlearn, OpenAI Gym (https://gym.openai.com/envs/CartPole-v1/)

This project has the following files:
* wrapper.py -- As the name suggests this is a wrapper and it calls functions in various other files. Run this file. 
* sample_run.py -- If you want to see how the env runs, you can run this file. 
* collect_training_data.py -- The function in this runs the env and collects the data for our training. 
* dnn_model.py -- The function in this file defines the Deep Neural Network model. This model was built using TFLearn. This network has 5 hidden layers with 500 nodes in each layer. Activation function used is ReLu.
* dnn_training.py -- The function in this file trains the network using the data collected using collect_training_data.py file. 
* dnn_run_model.py -- The function in this file uses the model trained in dnn_training.py and controls the env. 

The running version of this project can be seen here: https://youtu.be/Juxl8COUPFs

