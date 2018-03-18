import dnn_model as dm
import numpy as np

epochs = 10
batch_size = 300

def dnn_train(t_data, epochs=10, batch_size=300):

    X = np.array([i[0] for i in t_data]).reshape(-1,len(t_data[0][0]),1)
    y = [i[1] for i in t_data]

    model = dm.dnn_model_def(ipt_len = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=epochs, snapshot_step=batch_size, show_metric=True, run_id='openai_learning')
    return model