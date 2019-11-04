import numpy as np
import torch


def collate_fn_lm(data):
    inputs, targets, lengths = zip(*data)
    lengths = torch.LongTensor(lengths)
    return inputs, targets, lengths


def collate_fn_cf(data):
    samples, labels, lengths = zip(*data)
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)

    if isinstance(samples[0], np.ndarray):
        samples = torch.LongTensor(samples)
    elif isinstance(samples[0], tuple):
        samples = samples

    return samples, labels, lengths


def collate_fn_IMN(data):
    samples, aspects, polarities, opinions, lengths = zip(*data)

    opinions = torch.LongTensor(opinions)
    lengths = torch.LongTensor(lengths)

    if isinstance(samples[0], np.ndarray):
        samples = torch.LongTensor(samples)
    elif isinstance(samples[0], tuple):
        samples = samples

    return samples, list(aspects), list(polarities), opinions, lengths
