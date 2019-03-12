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
