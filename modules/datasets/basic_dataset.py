from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..common import Vocabulary


class BasicDataset(Dataset):
    """
        Class representing basic dataset with progress bar on data loading
    """
    def __init__(self):
        super(BasicDataset, self).__init__()
        self.preprocessing = self.tokenize

    def tokenize(self, text):
        return text.lower().split()

    def process_data(self, data):
        return [self.preprocessing(x) for x in tqdm(data)]

    def process_data_with_vocabulary(self, data):
        _data = []
        vocab = Vocabulary()

        for x in tqdm(data):
            tokens = self.preprocessing(x)
            vocab.process_tokens(tokens)
            _data.append(tokens)
        return _data, vocab
