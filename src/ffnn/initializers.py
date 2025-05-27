import numpy as np


def initialize_weights(input_dim, output_dim, method="random_uniform", init_params=None):
    init_params = init_params if init_params is not None else {}

    if method == "zero":
        W = np.zeros((input_dim, output_dim))
        b = np.zeros((1, output_dim))
    elif method == "random_uniform":
        lower = init_params.get('lower', -0.1)
        upper = init_params.get('upper', 0.1)
        seed = init_params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        W = np.random.uniform(lower, upper, (input_dim, output_dim))
        b = np.random.uniform(lower, upper, (1, output_dim))
    elif method == "random_normal":
        mean = init_params.get('mean', 0)
        var = init_params.get('var', 1)
        seed = init_params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        W = np.random.normal(mean, np.sqrt(var), (input_dim, output_dim))
        b = np.random.normal(mean, np.sqrt(var), (1, output_dim))
    elif method == "xavier":
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        seed = init_params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        b = np.zeros((1, output_dim))
    elif method == "he":
        factor = np.sqrt(2.0 / input_dim)
        seed = init_params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        W = np.random.randn(input_dim, output_dim) * factor
        b = np.zeros((1, output_dim))

    else:
        raise ValueError("Unknown initialization method: {}".format(method))

    return W, b