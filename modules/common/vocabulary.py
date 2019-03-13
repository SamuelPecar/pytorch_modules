import operator


class Vocabulary(object):
    """
    Basic class representing dataset vocabulary
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.stats = {}

        self.idx = 0
        self.add_word('<PAD>')
        self.add_word('<EOS>')

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

    def __len__(self):
        return len(self.word2idx) + 1

    def sort(self):
        self.stats = dict(sorted(self.stats.items(), key=operator.itemgetter(1), reverse=True))

        sorted_word2idx = {}
        sorted_idx = 0

        for key in self.stats:
            sorted_word2idx[key] = sorted_idx
            sorted_idx += 1

        self.word2idx = sorted_word2idx
        self.idx2word = {v: k for k, v in sorted_word2idx.items()}
