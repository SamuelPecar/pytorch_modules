import operator
import math


class Vocabulary(object):
    """
    Basic class representing dataset vocabulary
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.stats = {}

        self.idx = 0
        self.add_constant('<PAD>')
        self.add_constant('<EOS>')

    def process_tokens(self, text):
        for token in text:
            self.add_word(token)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.stats[word] = 1
        else:
            self.stats[word] += 1

    def add_constant(self, constant):
        self.word2idx[constant] = self.idx
        self.idx2word[self.idx] = constant
        self.idx += 1
        self.stats[constant] = math.inf

    def __len__(self):
        return len(self.word2idx)

    def sort(self):
        self.stats = dict(sorted(self.stats.items(), key=operator.itemgetter(1), reverse=True))

        sorted_word2idx = {}
        sorted_idx = 0

        for key in self.stats:
            sorted_word2idx[key] = sorted_idx
            sorted_idx += 1

        self.word2idx = sorted_word2idx
        self.idx2word = {v: k for k, v in sorted_word2idx.items()}


class ConcatVocabulary(object):
    """
    Class for concating multiple vocabularies
    """

    def __init__(self, vocabularies):
        assert len(vocabularies) > 0, 'vocabularies should not be an empty iterable'

        self.word2idx = {}
        self.idx2word = {}
        self.stats = {}
        self.idx = 0

        self.vocabs = vocabularies
        self.append_vocabularies()

    def append_vocabularies(self):
        for vocab in self.vocabs:
            for word in vocab.word2idx:
                self.add_word(word, vocab.stats[word])

        self.sort()

    def add_word(self, word, count):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.stats[word] = count
        else:
            self.stats[word] += count

    def sort(self):
        self.stats = dict(sorted(self.stats.items(), key=operator.itemgetter(1), reverse=True))

        sorted_word2idx = {}
        sorted_idx = 0

        for key in self.stats:
            sorted_word2idx[key] = sorted_idx
            sorted_idx += 1

        self.word2idx = sorted_word2idx
        self.idx2word = {v: k for k, v in sorted_word2idx.items()}

    def __len__(self):
        return len(self.word2idx)