import torch
import torch.nn as nn
import numpy as np

from ...common import Vocabulary

from config import device


class LookUp(nn.Module):
    """
    Proxy for Lookup embeddings with Glove initialization.
    Args:
        vocab Vocabulary: vocabulary class of input dataset
        file_name str: file of pretrained weigths
        trainable bool: path for local elmo wights file
        embedding_dim int: dimension of embeddings
        embedding_dropout float: value of dropout
    """

    def __init__(self, vocab, file_name=None, trainable=False, embedding_dim=300, embedding_dropout=.0, type='glove', **kwargs):
        super(LookUp, self).__init__()

        self.dataset_vocab = vocab
        self.embedding_dim = embedding_dim

        if file_name is not None:
            self.embeddings_vocab = Vocabulary()
            self.weights = self.load_weights(file_name, type)
        else:
            self.embeddings_vocab = self.dataset_vocab

        self.embedding = nn.Embedding(num_embeddings=len(self.embeddings_vocab), embedding_dim=self.embedding_dim)
        self.dropout = nn.Dropout(p=embedding_dropout)
        self.max_len = None

        if file_name is not None:
            self.init_weights(trainable)

    def set_max_len(self, max_len):
        self.max_len = max_len

    def init_weights(self, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.weights), requires_grad=trainable)

    def vectorize(self, batch):
        """
        Coverting array of tokens to array of ids, with a fixed max length and zero padding
        Args:
            text (): list of words
            word2idx (): dictionary of word to ids
        Returns: zero-padded list of ids
        """

        if not self.max_len:
            max_len = max([len(sample) for sample in batch])
        else:
            max_len = self.max_len

        words = torch.zeros((len(batch), max_len), dtype=torch.long).to(device)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long).to(device)

        for i, sample in enumerate(batch):
            for j, word in enumerate(sample):
                words[i][j] = self.embeddings_vocab.word2idx[word]
            mask[i, :len(sample)] = 1
        return words, mask

    def load_weights(self, file_name, type):
        weigths = []
        unknown = 0
        known = 0
        weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.embedding_dim))

        with open(file_name, encoding='UTF-8') as f:
            if type == 'word2vec':
                next(f)
            for line in f:
                values = line.split(' ')
                if values[0] in self.dataset_vocab.word2idx:
                    known += 1
                    self.embeddings_vocab.add_word(values[0])
                    weigths.append(np.asarray(values[1:], dtype='float32'))

            for word in self.dataset_vocab.word2idx:
                if word not in self.embeddings_vocab.word2idx:
                    unknown += 1
                    self.embeddings_vocab.add_word(word)
                    weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.embedding_dim))

            if "<UNK>" not in self.embeddings_vocab.word2idx:
                self.embeddings_vocab.add_word("<UNK>")
                weigths.append(np.random.uniform(low=-0.05, high=0.05, size=self.embedding_dim))

            print(known)
            print(unknown)
            return np.array(weigths, dtype='float32')

    def forward(self, x):

        vectorized, mask = self.vectorize(x)
        embeddings = self.embedding(vectorized)
        embeddings = self.dropout(embeddings)

        return embeddings, mask
