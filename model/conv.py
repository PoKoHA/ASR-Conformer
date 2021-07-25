import torch
import torch.nn as nn

from model.activation import *

class DepthwiseConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        return output

class PointwiseConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=False):
        super(PointwiseConv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        return output

# Conv Module

class ConformerConvModule(nn.Module):
    """
    구조는 README 참고
    inputs  [batch, time ,dim]
    outputs [batch, time ,dim]
    """

    def __init__(self, args, in_channels, kernel_size=31, expansion_factor=2, dropout_p=.1):
        super(ConformerConvModule, self).__init__()
        assert (kernel_size -1) % 2 == 0, "kenel_size 는 반드시 홀수"
        assert expansion_factor == 2, "2만 지원 가능"
        self.args = args

        self.layernorm = nn.LayerNorm(in_channels)
        self.pointwiseConv1d_a = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.GLU = GLU(dim=1)
        self.depthwiseConv1d = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size -1) // 2)
        # Same Padding
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.Swish = Swish()
        self.pointwiseConv1d_b = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs):
        inputs = inputs.cuda(self.args.gpu)

        layernorm = self.layernorm(inputs)
        layernorm = layernorm.transpose(1, 2)
        point_conv_a = self.pointwiseConv1d_a(layernorm)
        glu = self.GLU(point_conv_a)

        depth_conv = self.depthwiseConv1d(glu)
        batchnrom = self.batchnorm(depth_conv)
        swish = self.Swish(batchnrom)

        point_conv_b = self.pointwiseConv1d_b(swish)
        dropout = self.dropout(point_conv_b)

        output = dropout.transpose(1, 2)

        return output



































