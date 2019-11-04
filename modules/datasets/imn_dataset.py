from .basic_dataset import BasicDataset


class IMNDataset(BasicDataset):
    def __init__(self, x, aspects, polarities, opinions, preprocessing=None):
        """
        Class representing dataset for AE and ABSA tasks
        Args:
            x (): list of training samples
            y (): list of training labels
            preprocessing (method): method used for preprocessing, if None tokenize from Basic dataset is used
        """
        super(IMNDataset, self).__init__()

        self.preprocessing = self.tokenize if preprocessing is None else preprocessing
        self.samples = x
        self.aspects = aspects
        self.polarities = polarities
        self.opinions = opinions

    def __getitem__(self, index):
        sample, aspects, polarities, opinions = self.samples[index], self.aspects[index], self.polarities[index], self.opinions[index]
        length = len(self.samples[index])

        return sample, aspects, polarities, opinions, length

    def __len__(self):
        return len(self.samples)
