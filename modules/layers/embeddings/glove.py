import torch
import torch.nn as nn
import numpy as np

from ...common import Vocabulary

from config import device


class GloVe(nn.Module):
    """
    Proxy for GloVe embeddings.
    Args:
        vocab Vocabulary: vocabulary class of input dataset
        file_name str: file of pretrained weigths
        trainable bool: path for local elmo wights file
        embedding_dim int: dimension of embeddings
        embedding_dropout float: value of dropout
    """

    def __init__(self, vocab, file_name=None, trainable=False, embedding_dim=300, embedding_dropout=.0, **kwargs):
        super(GloVe, self).__init__()

        self.dataset_vocab = vocab
        self.embeddings_vocab = Vocabulary()
        self.dim = embedding_dim

        if file_name is not None:
            self.weights = self.load_weights(file_name)

        self.embedding = nn.Embedding(num_embeddings=len(self.embeddings_vocab), embedding_dim=self.dim)
        self.dropout = nn.Dropout(p=embedding_dropout)

        if file_name is not None:
            self.init_weights(trainable)

    def init_weights(self, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.weights), requires_grad=trainable)

    def vectorize(self, batch):
        max_len = max([len(sample) for sample in batch])
        words = torch.zeros((len(batch), max_len), dtype=torch.long).to(device)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long).to(device)

        for i, sample in enumerate(batch):
            for j, word in enumerate(sample):
                if word in self.embeddings_vocab.word2idx:
                    words[i][j] = self.embeddings_vocab.word2idx[word]
                else:
                    words[i][j] = self.embeddings_vocab.word2idx['<UNK>']
                pass
            mask[i, :len(sample)] = 1
        return words, mask

    def load_weights(self, file_name):
        weigths = []
        weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.dim))

        with open(file_name, encoding='UTF-8') as f:
            for line in f:
                values = line.split(' ')
                if values[0] in self.dataset_vocab.word2idx:
                    self.embeddings_vocab.add_word(values[0])
                    weigths.append(np.asarray(values[1:], dtype='float32'))

            for word in self.embeddings_vocab.word2idx:
                if word not in self.dataset_vocab.word2idx:
                    self.embeddings_vocab.add_word(word)
                    weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.dim))

            if "<UNK>" not in self.embeddings_vocab.word2idx:
                self.embeddings_vocab.add_word("<UNK>")
                weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.dim))

            return np.array(weigths, dtype='float32')

    def forward(self, x):

        vectorized, mask = self.vectorize(x)
        embeddings = self.embedding(vectorized)
        embeddings = self.dropout(embeddings)

        return embeddings, mask
