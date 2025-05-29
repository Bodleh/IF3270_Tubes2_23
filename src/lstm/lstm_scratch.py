import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

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
        # During inference (training=False), dropout does nothing.
        # Keras dropout layer is also bypassed during model.predict()
        return x

class DenseScratch:
    def __init__(self, weights, activation='relu'):
        self.W, self.b = weights
        self.activation = activation

    def forward(self, x):
        """
        x shape: (batch_size, input_features)
        Returns: (batch_size, output_features)
        """
        output = np.dot(x, self.W) + self.b
        if self.activation == 'softmax':
            return softmax(output)
        elif self.activation == 'relu':
            return relu(output)
        else:
            return relu(output)

class LSTMSingleScratch:
    """Batch-compatible forward pass for a unidirectional LSTM."""
    def __init__(self, weights, return_sequences=False):
        W, U, b = weights
        self.n_units = W.shape[1] // 4
        self.return_sequences = return_sequences
        
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
        Returns: 
            if self.return_sequences: (batch_size, sequence_length, n_units)
            else: (batch_size, n_units) - the last hidden state.
        """
        batch_size, seq_len, _ = x.shape
        
        h_t = np.zeros((batch_size, self.n_units))
        c_t = np.zeros((batch_size, self.n_units))

        if self.return_sequences:
            all_h_t = np.zeros((batch_size, seq_len, self.n_units))

        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            f_t = sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            c_tilde = tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)
            c_t = f_t * c_t + i_t * c_tilde
            o_t = sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            h_t = o_t * tanh(c_t)
            
            if self.return_sequences:
                all_h_t[:, t, :] = h_t
        
        if self.return_sequences:
            return all_h_t
        else:
            return h_t

class BidirectionalLSTMScratch:
    """Batch-compatible forward pass for a Bidirectional LSTM."""
    def __init__(self, weights, return_sequences=False):
        self.return_sequences = return_sequences
        self.lstm_forward = LSTMSingleScratch(weights[:3], return_sequences=self.return_sequences)
        self.lstm_backward = LSTMSingleScratch(weights[3:], return_sequences=self.return_sequences)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, embedding_dim)
        Returns: 
            if self.return_sequences: (batch_size, sequence_length, 2 * n_units)
            else: (batch_size, 2 * n_units)
        """
        h_forward_output = self.lstm_forward.forward(x)

        x_reversed = np.flip(x, axis=1)
        h_backward_output = self.lstm_backward.forward(x_reversed)

        if self.return_sequences:
            h_backward_output_reversed_time = np.flip(h_backward_output, axis=1)
            return np.concatenate((h_forward_output, h_backward_output_reversed_time), axis=2)
        else:
            return np.concatenate((h_forward_output, h_backward_output), axis=1)
    
class NumpyLSTM:
    """
    A class to build a Numpy-based RNN model from a Keras model's weights
    and perform predictions.
    """
    def __init__(self, keras_model):
        """
        Constructs the scratch model from the layers of a trained Keras model.
        """
        self.layers = []
        for layer in keras_model.layers:
            layer_name = layer.name.lower()
            weights = layer.get_weights()
            if 'embedding' in layer_name:
                self.layers.append(EmbeddingScratch(weights))
            elif 'bidirectional' in layer_name:
                rs = layer.return_sequences
                self.layers.append(BidirectionalLSTMScratch(weights, rs))
            elif 'unidirectional' in layer_name:
                rs = layer.return_sequences
                self.layers.append(LSTMSingleScratch(weights, rs))
            elif 'dropout' in layer_name:
                self.layers.append(DropoutScratch(layer.rate))
            elif 'dense' in layer_name:
                self.layers.append(DenseScratch(weights))
            elif 'output' in layer_name:
                self.layers.append(DenseScratch(weights, activation='softmax'))

    def predict(self, X_data, batch_size=64):
        """
        Performs a forward pass on the input data and returns class predictions.
        """
        all_preds = []
        num_samples = len(X_data)

        for i in range(0, num_samples, batch_size):
            x_batch = X_data[i : i + batch_size]
            
            output = x_batch
            for layer in self.layers:
                output = layer.forward(output)
            
            preds_batch = np.argmax(output, axis=1)
            all_preds.append(preds_batch)
            
        return np.concatenate(all_preds)