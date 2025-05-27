import numpy as np


class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.dim = dim
        self.eps = eps
        self.scale = np.ones((1, dim))
        self.dscale = np.zeros_like(self.scale)
        
    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + self.eps)
        self.normalized = x / self.rms
        return self.normalized * self.scale
    
    def backward(self, grad_output):
        self.dscale = np.sum(grad_output * self.normalized, axis=0, keepdims=True)
        
        dx = self.scale * grad_output / self.rms
        
        drms = -np.sum(grad_output * self.x * self.scale, axis=1, keepdims=True) / (self.rms ** 2)
        dx_rms = drms * self.x / (self.dim * self.rms)
        
        return dx + dx_rms
    
    def update(self, learning_rate):
        self.scale -= learning_rate * self.dscale