from torch import Tensor, nn
import torch
from .decorators import module

def trunc_normal_(x:Tensor, mean:float=0., std:float=1.) -> Tensor:
    "Truncated normal initialization."
    "From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12"
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def _sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

@module('low','high')
def SigmoidRange(self, x):
    "Sigmoid module with range `(low, high)`"
    return _sigmoid_range(x, self.low, self.high)

class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)

class LinBnDrop(nn.Sequential):
    " Custom Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0.0, act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)
