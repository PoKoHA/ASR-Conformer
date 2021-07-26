import numpy as np

import torch
import torch.nn as nn

from model.module import Linear

class DecoderRNNT(nn.Module):
    """
     Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size
        ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Returns:
    * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN}

    def __init__(
            self,
            n_classes,
            hidden_dim,
            output_dim,
            n_layers,
            rnn_type='lstm',
            sos_id=2001,
            eos_id=2002,
            dropout_p=0.2):

        super(DecoderRNNT, self).__init__()

        self.hiddn_dim =hidden_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(n_classes, hidden_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )

        self.fc = Linear(hidden_dim, output_dim)

    # todo >>
    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p


    def forward(self, inputs, input_lengths=None, hidden_states=None):
        # print(" --[Decoder]--")
        input_lengths = torch.sort(input_lengths, descending=True)
        inputs_copy = inputs.new_zeros(inputs.size())
        # print("inputs: ", inputs)
        # print("input_lengths:", input_lengths)
        for i1, i2 in enumerate(input_lengths[1]): # ([0, 1, 2, 3, 4, 6, 7, 5])
            inputs_copy[i1][:] = inputs[i2][:]

        # print("inputs: ", inputs_copy)
        embedded = self.embedding(inputs_copy)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded.transpose(0, 1), input_lengths[0].cpu())
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = self.fc(outputs.transpose(0, 1))

        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.fc(outputs)

        return outputs, hidden_states
