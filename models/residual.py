# @source: https://github.com/MishaLaskin/vqvae
# @modified: wujiarong
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - out_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, out_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv1d(in_dim, res_h_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(res_h_dim),
            nn.PReLU(),
            nn.Conv1d(res_h_dim, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - out_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, out_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, out_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        return tensor

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 3000, 512))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(3000, 3000, 1500)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(3000, 3000, 1500, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)