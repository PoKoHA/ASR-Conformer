import torch
import torch.nn as nn

from model.activation import Swish
from model.module import Linear

class FeedForwardModule(nn.Module):

    def __init__(self, args, encoder_dim=512, expansion_factor=4, dropout_p=0.1):
        super(FeedForwardModule, self).__init__()
        self.args = args

        self.layer_norm = nn.LayerNorm(encoder_dim)

        self.linear_a = Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.linear_b = Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout_p)
        self.swish = Swish()

    def forward(self, inputs):
        inputs = inputs.cuda(self.args.gpu)

        pre_norm = self.layer_norm(inputs)
        linear_a = self.linear_a(pre_norm)
        swish = self.swish(linear_a)
        dropout_a = self.dropout(swish)

        linear_b = self.linear_b(dropout_a)
        output = self.dropout(linear_b)

        return output

