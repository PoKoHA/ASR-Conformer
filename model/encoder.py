import torch
import torch.nn as nn

from model.FFN import FeedForwardModule
from model.attention import MultiHeadedSelfAttentionModule
from model.conv import ConformerConvModule
from model.subsampling import Conv2dSubsampling
from model.module import ResidualConnectionModule, Linear

class ConformerBlock(nn.Module):

    def __init__(
            self,
            args,
            encoder_dim=512,
            n_heads=8,
            FFN_expansion_factor=4,
            conv_expansion_factor=2,
            FFN_dropout=0.1,
            conv_dropout=0.1,
            attn_dropout=0.1,
            kernel_size=31,
            half_step_residual=True,
    ):
        super(ConformerBlock, self).__init__()
        self.args = args

        if half_step_residual:
            self.FFN_residual_factor = 0.5
        else:
            self.FFN_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    args=args,
                    encoder_dim=encoder_dim,
                    expansion_factor=FFN_expansion_factor,
                    dropout_p=FFN_dropout
                ),
                module_factor=self.FFN_residual_factor
            ),# FFN Module
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    args=args,
                    d_model=encoder_dim,
                    n_heads=n_heads,
                    dropout_p=attn_dropout
                ),
            ), # attn module
            ResidualConnectionModule(
                module=ConformerConvModule(
                    args=args,
                    in_channels=encoder_dim,
                    kernel_size=kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout
                ),
            ), # conv module
            ResidualConnectionModule(
                module=FeedForwardModule(
                    args=args,
                    expansion_factor=FFN_expansion_factor,
                    dropout_p=FFN_dropout,
                ), module_factor=self.FFN_residual_factor,
            ),
            nn.LayerNorm(encoder_dim)
        )

    def forward(self, inputs):
        inputs = inputs.cuda(self.args.gpu)
        outputs = self.sequential(inputs)

        return outputs


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            args,
            input_dim=80, # paper
            encoder_dim=512,
            n_layers=17,
            n_heads=8,
            FFN_expansion_factor=4,
            conv_expansion_factor=2,
            input_dropout=0.1,
            FFN_dropout=0.1,
            conv_dropout=0.1,
            attn_dropout=0.1,
            kernel_size=31,
            half_step_residual=True,
    ):
        super(ConformerEncoder, self).__init__()
        self.args = args

        # Processing
        self.conv_subsample = Conv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_linear = Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.input_dropout = nn.Dropout(p=input_dropout)

        # encoder Module
        self.layers = nn.ModuleList([
            ConformerBlock(
                args=args,
                encoder_dim=encoder_dim,
                n_heads=n_heads,
                FFN_expansion_factor=FFN_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                FFN_dropout=FFN_dropout,
                attn_dropout=attn_dropout,
                conv_dropout=conv_dropout,
                kernel_size=kernel_size,
                half_step_residual=half_step_residual,
            ).cuda(args.gpu) for _ in range(n_layers)])


    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p


    def forward(self, inputs, input_lengths):
        # print(" --[Encoder]--")
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_linear(outputs)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs, output_lengths





