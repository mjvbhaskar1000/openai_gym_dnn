import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

def dnn_model_def(ipt_len):

    network = input_data(shape=[None, ipt_len, 1], name='input')
    network = fully_connected(network, 500, activation='relu')
    network = fully_connected(network, 500, activation='relu')
    network = fully_connected(network, 500, activation='relu')
    network = fully_connected(network, 500, activation='relu')
    network = fully_connected(network, 500, activation='relu')

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
    dnn = tflearn.DNN(network, tensorboard_dir='log')

    return dnn