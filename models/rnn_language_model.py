import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.layers.encoders.rnn_encoder import RNNEncoder
from modules.common.utils import sort_by_lengths


class RNNLanguageModel(nn.Module):
    def __init__(self, encoder_params, vocab):
        super(RNNLanguageModel, self).__init__()

        self.dropout = nn.Dropout(p=encoder_params['dropout'])

        self.embeddings = nn.Embedding(num_embeddings=len(vocab), embedding_dim=200)
        self.encoder = RNNEncoder(input_size=self.embeddings.embedding_dim, **encoder_params)
        self.decoder = nn.Linear(self.encoder.feature_size, len(vocab))

    def forward(self, input, mask, hidden, lengths):
        sorted_lengths, sort, unsort = sort_by_lengths(lengths)

        embedded = self.embeddings(input)

        output, hidden = self.encoder(sort(embedded), hidden, sort(mask), lengths=sorted_lengths)

        output = self.dropout(unsort(output))

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
