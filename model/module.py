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

























































