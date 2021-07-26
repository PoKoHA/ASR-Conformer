import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ResidualConnectionModule(nn.Module):
    """
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """

    def __init__(self, module, module_factor=1.0, input_factor=1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        residual = inputs
        module_output = self.module(inputs)

        output = (module_output * self.module_factor) + (residual * self.input_factor)

        return output


########################
# Decoder
########################
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        # keepdim = dim 유지하지만 1로 바뀜
        # e.g) Tensor[3, 3, 4] --> Tensor[3, 3, 1]
        std = inputs.std(dim=-1, keepdim=True)

        output = (inputs - mean) / (std + self.eps)
        output = self.gamma * output + self.beta

        return output






















































