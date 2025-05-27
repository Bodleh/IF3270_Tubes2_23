import numpy as np

def softmax(x):
    """Softmax function for a batch of inputs."""
    # x shape: (batch_size, num_classes)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class EmbeddingScratch:
    def __init__(self, weights):
        self.embedding_matrix = weights[0]

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length)
        Returns: (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding_matrix[x]

class DropoutScratch:
    def __init__(self, rate):
        self.rate = rate
    
    def forward(self, x, training=False):
        # # During inference (training=False), dropout does nothing.
        # if training:
        #     mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        #     return x * mask
        return x

class DenseScratch:
    def __init__(self, weights):
        self.W, self.b = weights

    def forward(self, x):
        """
        x shape: (batch_size, input_features)
        Returns: (batch_size, output_features)
        """
        output = np.dot(x, self.W) + self.b
        return softmax(output)

class LSTMSingleScratch:
    """Batch-compatible forward pass for a unidirectional LSTM."""
    def __init__(self, weights):
        W, U, b = weights
        self.n_units = W.shape[1] // 4
        
        # Kernel weights (for input x)
        self.W_i = W[:, :self.n_units]
        self.W_f = W[:, self.n_units:self.n_units*2]
        self.W_c = W[:, self.n_units*2:self.n_units*3]
        self.W_o = W[:, self.n_units*3:]
        
        # Recurrent weights (for hidden state h)
        self.U_i = U[:, :self.n_units]
        self.U_f = U[:, self.n_units:self.n_units*2]
        self.U_c = U[:, self.n_units*2:self.n_units*3]
        self.U_o = U[:, self.n_units*3:]
        
        # Biases
        self.b_i = b[:self.n_units]
        self.b_f = b[self.n_units:self.n_units*2]
        self.b_c = b[self.n_units*2:self.n_units*3]
        self.b_o = b[self.n_units*3:]

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, embedding_dim)
        Returns: (batch_size, n_units) - the last hidden state for each sequence in the batch.
        """
        batch_size, seq_len, _ = x.shape
        
        h_t = np.zeros((batch_size, self.n_units))
        c_t = np.zeros((batch_size, self.n_units))

        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            f_t = sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            c_tilde = tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)
            c_t = f_t * c_t + i_t * c_tilde
            o_t = sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            h_t = o_t * tanh(c_t)
            
        return h_t

class BidirectionalLSTMSratch:
    """Batch-compatible forward pass for a Bidirectional LSTM."""
    def __init__(self, weights):
        self.lstm_forward = LSTMSingleScratch(weights[:3])
        self.lstm_backward = LSTMSingleScratch(weights[3:])

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, embedding_dim)
        Returns: (batch_size, 2 * n_units)
        """
        h_forward = self.lstm_forward.forward(x)

        x_reversed = np.flip(x, axis=1)
        h_backward = self.lstm_backward.forward(x_reversed)

        return np.concatenate((h_forward, h_backward), axis=1)