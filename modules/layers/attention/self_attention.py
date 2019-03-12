from torch import nn, torch

from config import device


class SelfAttention(nn.Module):
    """
    Core of the code comes from https://github.com/cbaziotis/ntua-slp-semeval2018/blob/master/modules/nn/attention.py
    """

    def __init__(self, attention_size, dropout=.0):
        super(SelfAttention, self).__init__()

        modules = []
        modules.append(nn.Linear(attention_size, 1))
        modules.append(nn.Tanh())
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask, lengths):
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        masked_scores = scores * mask.float()
        _sums = masked_scores.sum(-1, keepdim=True)
        scores = masked_scores.div(_sums)

        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, scores
