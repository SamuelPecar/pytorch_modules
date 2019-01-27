import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# TODO - only LSTM add support for type GRU
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.3, type='LSTM', **kwargs):
        """
            RNN encoder layer.
            Args:
                input_size int: size of input data (i.e. dimension of embeddings)
                hidden_size int: size of hidden state in RNN
                num_layers int: number of layers in RNN
                bidirectional bool: if True RNN will be bidirectional
                dropout float: value of dropout - between layers and also dropout of output
                type str: TODO support for GRU
            """
        super(RNNEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_directions = 2 if self.bidirectional else 1

        self.hidden_size = hidden_size
        self.feature_size = self.num_directions * hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               batch_first=True)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

    def forward(self, embedded, hidden, mask, lengths):
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True)

        lstm_output, lstm_hidden = self.encoder(packed_embedded, hidden)

        padded_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        return self.dropout(padded_output), lstm_hidden
