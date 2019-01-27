import torch
import numpy as np

from sklearn.utils import compute_class_weight


def class_weigths(y):
    labels = np.unique(y)
    weights = {cls: weight for cls, weight in zip(np.unique(y), compute_class_weight('balanced', labels, y))}

    return torch.FloatTensor([weights[key] for key in sorted(labels)])
