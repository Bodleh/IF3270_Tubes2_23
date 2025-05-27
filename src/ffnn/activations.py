import numpy as np


class ActivationFunction:
    def forward(self, z):
        raise NotImplementedError("Method forward belum diimplementasikan!")

    def backward(self, z):
        raise NotImplementedError("Method backward belum diimplementasikan!")


class LinearActivation(ActivationFunction):
    def forward(self, z):
        return z

    def backward(self, z):
        return np.ones_like(z)


class ReLUActivation(ActivationFunction):
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)


class SigmoidActivation(ActivationFunction):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        sig = self.forward(z)
        return sig * (1 - sig)


class TanhActivation(ActivationFunction):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        return 1 - np.tanh(z) ** 2


class SoftmaxActivation(ActivationFunction):
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)

class LeakyReLUActivation(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, z):
        return np.maximum(self.alpha * z, z)
    
    def backward(self, z):
        dZ = np.ones_like(z)
        dZ[z < 0] = self.alpha
        return dZ

class ELUActivation(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def forward(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))
    
    def backward(self, z):
        return np.where(z > 0, 1, self.alpha * np.exp(z))
