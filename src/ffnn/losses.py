import numpy as np


class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError("Loss function belum diimplementasikan!")

    def gradient(self, y_true, y_pred):
        raise NotImplementedError("Gradient belum diimplementasikan!")


class MSELoss(Loss):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


class CategoricalCrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        epsilon = 1e-15  # agar tidak menghasilkan log(0) -> INF
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
