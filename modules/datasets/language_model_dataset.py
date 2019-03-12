from .basic_dataset import BasicDataset


class LanguageModelDataset(BasicDataset):
    def __init__(self, x, preprocessing=None):
        """
        Class representing dataset for language modeling tasks
        Args:
            x (): list of training samples
            preprocessing (method): method used for preprocessing, if None tokenize from Basic dataset is used
        """
        super(LanguageModelDataset, self).__init__()

        self.preprocessing = self.tokenize if preprocessing is None else preprocessing
        self.data, self.vocab = self.process_data_with_vocabulary(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.data[index]
        targets = inputs[1:]

        length = len(self.data[index]) - 1

        return inputs[:-1], targets, length
