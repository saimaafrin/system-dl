import torch
import pytorch_apis
from gp_apis import gp_Mul, gp_Tpose

class LinearLayer:
    def __init__(self, in_features, out_features, device):
        # Initialize weights and biases with torch tensors
        self.W = torch.rand(out_features, in_features, device=device)
        self.B = torch.rand(out_features, device=device)
        # Store gradients
        self.dW = torch.zeros_like(self.W)
        self.dB = torch.zeros_like(self.B)
        # Store the input
        self.X = None
        self.device = device

    def forward(self, X):
        # Store the input for backward pass
        self.X = X
        # Compute forward pass: Y = XW + B using gp_Mul
        Y = gp_Mul(X, self.W, X.size(0), self.W.size(1), X.size(0), X.size(1), self.W.size(1), self.device)
        Y = Y + self.B
        return Y


    def backward(self, dY):
        # Compute gradients using gp_Mul and gp_Tpose
        dX_T = gp_Tpose(self.X, self.X.size(0), self.X.size(1), self.X.size(0), self.X.size(1), self.device)
        self.dW = gp_Mul(dX_T, dY, dX_T.size(0), dX_T.size(1), dX_T.size(0), dX_T.size(1), dY.size(1), self.device)
        # dB = sum(dY, axis=0)
        self.dB = torch.sum(dY, axis=0)
        # dX = dY @ W.T using gp_Mul and gp_Tpose
        dW_T = gp_Tpose(self.W, self.W.size(0), self.W.size(1), self.W.size(0), self.W.size(1), self.device)
        dX = gp_Mul(dY, dW_T, dY.size(0), dY.size(1), dY.size(0), dY.size(1), dW_T.size(1), self.device)
        return dX


    def update(self, learning_rate):
        # Update weights and biases using gradients
        self.W = self.W - learning_rate * self.dW
        self.B = self.B - learning_rate * self.dB
