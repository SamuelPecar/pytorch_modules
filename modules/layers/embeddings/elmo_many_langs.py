import torch
import torch.nn as nn
from elmoformanylangs import Embedder
from torch.nn.utils.rnn import pad_sequence

try:
    from config import device
except:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ELMoForManyLangs(nn.Module):
    """
    """

    def __init__(self, embedding_dropout=0., emb_path='./ELMo', **kwargs):
        super(ELMoForManyLangs, self).__init__()

        self.dropout = nn.Dropout(p=embedding_dropout)
        self.embedder = Embedder(emb_path)
        self.embedding_dim = 1024

    def forward(self, sentences):
        embedded = self.embedder.sents2elmo(sentences)

        embedded_list = []
        masks = []

        for e in embedded:
            embedded_list.append(self.dropout(torch.Tensor(e)))
            masks.append(torch.ones(len(e)).long())

        return pad_sequence(embedded_list, batch_first=True).to(device), pad_sequence(masks, batch_first=True).to(device)
