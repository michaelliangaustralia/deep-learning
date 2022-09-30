import numpy as np
import IPython # debugging

# NN model hyperparameters
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"}, # TODO why is the input dim 2
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

# Random initialisation of nn layers - weights and biases
def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        params_values['W' + str(idx+1)] = np.random.rand(
            layer['output_dim'], layer['input_dim']
        ) * 0.1 # weight is related to the connections between nodes
        params_values['B' + str(idx+1)] = np.random.rand(
            layer['output_dim'], 1
        ) * 0.1 # bias is related only to the output nodes

    return params_values

# Activation functions
def ReLU(x):
    IPython.embed()
    return x * (x > 0) # (x > 0) evaluates to bool (0 or 1) which implements ReLU




ReLU(1)

