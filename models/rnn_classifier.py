import torch
import torch.nn as nn

from modules.common.utils import sort_by_lengths

from modules.layers.encoders.rnn_encoder import RNNEncoder
from modules.layers.attention import SelfAttention
from modules.layers.embeddings import ELMo


class RNNClassifier(nn.Module):
    def __init__(self, embed_params, encoder_params, output_dim=2, dropout=0., **kwargs):
        super(RNNClassifier, self).__init__()

        self.embeddings = ELMo(**embed_params)

        self.encoder = RNNEncoder(input_size=self.embeddings.dim, **encoder_params)

        self.attention = SelfAttention(attention_size=self.encoder.feature_size, dropout=dropout)

        self.hidden2out = nn.Linear(self.encoder.feature_size, output_dim)

        self.activation = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, lengths):
        sorted_lengths, sort, unsort = sort_by_lengths(lengths)

        embedded, mask = self.embeddings(x)

        output_encoder, hidden = self.encoder(sort(embedded), hidden=None, mask=sort(mask), lengths=sorted_lengths)
        representations, attentions = self.attention(output_encoder, mask=sort(mask), lengths=sorted_lengths)

        output = self.hidden2out(representations)
        output = self.activation(output)

        return unsort(output)
