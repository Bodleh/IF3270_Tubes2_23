import numpy as np
import tensorflow as tf
from typing import List
from keras_model import build_rnn_model


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def tanh(x): return np.tanh(x)


class EmbeddingLayer:
    def __init__(self, W):
        self.W = W

    def forward(self, tokens):
        out = self.W[tokens]
        out[tokens == 0] = 0.0
        return out


class SimpleRNNLayer:
    def __init__(self, Wx, Wh, b, return_sequences: bool):
        self.Wx, self.Wh, self.b = Wx, Wh, b
        self.return_sequences = return_sequences

    def _step(self, x_t, h_prev):
        return tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)

    def forward(self, x):
        B, L, _ = x.shape
        u = self.Wh.shape[0]
        h = np.zeros((B, u), dtype=np.float32)
        if self.return_sequences:
            outputs = np.zeros((B, L, u), dtype=np.float32)

        for t in range(L):
            x_t = x[:, t, :]
            mask = (x_t != 0).any(axis=1, keepdims=True)

            new_h = tanh(x_t @ self.Wx + h @ self.Wh + self.b)
            h = np.where(mask, new_h, h)

            if self.return_sequences:
                outputs[:, t, :] = h

        return outputs if self.return_sequences else h


class BidirectionalSimpleRNN:
    def __init__(self, fwd: SimpleRNNLayer, bwd: SimpleRNNLayer,
                 return_sequences: bool):
        self.fwd, self.bwd = fwd, bwd
        self.return_sequences = return_sequences

    def forward(self, x):
        out_f = self.fwd.forward(x)
        out_b = self.bwd.forward(x[:, ::-1, :])
        if self.return_sequences:
            out_b = out_b[:, ::-1, :]
            return np.concatenate([out_f, out_b], axis=-1)
        else:
            return np.concatenate([out_f, out_b], axis=-1)


class DenseLayer:
    def __init__(self, W, b): self.W, self.b = W, b

    def forward(self, h):
        return softmax(h @ self.W + self.b)


class NumpyRNNModel:
    def __init__(self, weights_h5: str, vect_path: str,
                 num_layers=1, units=64, bidirectional=True,
                 max_len=64):

        vect_model = tf.keras.models.load_model(vect_path)
        self.vector = vect_model.layers[-1]
        vocab_size = self.vector.vocabulary_size()

        keras_model = build_rnn_model(
            vocab_size, num_layers=num_layers,
            units=units, bidirectional=bidirectional,
            max_len=max_len
        )
        keras_model.load_weights(weights_h5)

        self.layers: List = []
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                self.layers.append(
                    EmbeddingLayer(layer.get_weights()[0])
                )

            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                Wx, Wh, b = layer.get_weights()
                self.layers.append(
                    SimpleRNNLayer(Wx, Wh, b, layer.return_sequences)
                )

            elif isinstance(layer, tf.keras.layers.Bidirectional):
                fwd_Wx, fwd_Wh, fwd_b = layer.forward_layer.get_weights()
                bwd_Wx, bwd_Wh, bwd_b = layer.backward_layer.get_weights()
                fwd = SimpleRNNLayer(fwd_Wx, fwd_Wh, fwd_b,
                                     layer.return_sequences)
                bwd = SimpleRNNLayer(bwd_Wx, bwd_Wh, bwd_b,
                                     layer.return_sequences)
                self.layers.append(
                    BidirectionalSimpleRNN(fwd, bwd, layer.return_sequences)
                )

            elif isinstance(layer, tf.keras.layers.Dense):
                Wd, bd = layer.get_weights()
                self.layers.append(DenseLayer(Wd, bd))

        self.max_len = max_len

    def _forward_once(self, token_batch):
        x = token_batch
        for lyr in self.layers:
            x = lyr.forward(x)
        return x

    def predict_proba(self, texts: List[str],
                      batch_size: int | None = None) -> np.ndarray:
        if batch_size is None:
            tokens = self.vector(texts).numpy()
            return self._forward_once(tokens)

        outputs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            tokens = self.vector(chunk).numpy()
            out = self._forward_once(tokens)
            outputs.append(out)
        return np.vstack(outputs)

    def predict(self, texts: List[str], batch_size: int | None = None):
        return self.predict_proba(texts, batch_size).argmax(axis=1)
