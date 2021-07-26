import torch
import torch.nn as nn
from model.module import LayerNorm

class AddNorm(nn.Module):

    def __init__(self, sublayer, d_model=512):
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            # e.g) MulitHeadAttention 하면 return 값으로 (output, attn_map)
            return self.layer_norm(output[0] + residual), output[1]

        return self.layer_norm(output + residual)

class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model=512, d_ff=2048):
        super(PositionWiseFeedForwardNet, self).__init__()

        # Transormer paper) 원래 Linear 2번이지만 Conv1d kernel=1 로하는 것랑 일치
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs.transpose(1, 2))
        relu = self.relu(conv1)
        conv2 = self.conv2(relu)
        output = conv2.transpose(1, 2)

        return output