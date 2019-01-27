import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

from config import device


class BERT(nn.Module):
    """
    Proxy for Bert embeddings.
    Args:
        embedding_dropout float: value of dropout
        type str: list of training labels
    """

    def __init__(self, embedding_dropout=0.5, type='bert-large-uncased', **kwargs):
        super(BERT, self).__init__()

        self.word_splitter = BertBasicWordSplitter()
        self.bert_embedder = PretrainedBertEmbedder(type).to(device)
        self.bert_indexer = PretrainedBertIndexer(type)

        self.bert_indexer._added_to_vocabulary = True
        self.bert_indexer._start_piece_ids = []
        self.bert_indexer._end_piece_ids = []

        self.dropout = nn.Dropout(p=embedding_dropout)

        self.dim = self.bert_embedder.get_output_dim()

    def forward(self, sentences):
        embeddings = []
        masks = []

        for sentence in sentences:
            tokens = self.word_splitter.split_words(' '.join(sentence))
            bert_ids = self.bert_indexer.tokens_to_indices(tokens, None, "bert")
            embed = self.bert_embedder(torch.Tensor([bert_ids['bert']]).long().to(device))
            embeddings.append(self.dropout(embed[0]))
            mask = torch.ones(len(bert_ids['bert'])).long()
            masks.append(mask)

        return pad_sequence(embeddings, batch_first=True), pad_sequence(masks, batch_first=True).to(device)
