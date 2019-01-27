from .basic_dataset import BasicDataset


class ClassificationDataset(BasicDataset):
    def __init__(self, x, y, preprocessing=None):
        """
        Class representing dataset for classification tasks
        Args:
            x (): list of training samples
            y (): list of training labels
            preprocessing (method): method used for preprocessing, if None tokenize from Basic dataset is used
        """
        super(ClassificationDataset, self).__init__()

        self.preprocessing = self.tokenize if preprocessing is None else preprocessing
        self.data, self.vocab = self.process_data_with_vocabulary(x)
        self.labels = y

    def __getitem__(self, index):
        sample, label = self.data[index], self.labels[index]
        length = len(self.data[index])

        return sample, label, length

    def __len__(self):
        return len(self.data)
