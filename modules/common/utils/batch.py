import torch

def sort_by_lengths(lengths):
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    _, reversed_idx = sorted_idx.sort(descending=True)

    def sort(iterable):
        return iterable[sorted_idx]

    def unsort(iterable):
        return iterable[reversed_idx][sorted_idx][reversed_idx]

    return sorted_lengths, sort, unsort


def vectorize(text, word2idx, device):
    """
    Coverting array of tokens to array of ids, with a fixed max length and zero padding
    Args:
        text (): list of words
        word2idx (): dictionary of word to ids
    Returns: zero-padded list of ids
    """

    max_length = max([len(x) for x in text])

    words = torch.zeros((len(text), max_length), dtype=torch.long).to(device)
    mask = torch.zeros((len(text), max_length), dtype=torch.long).to(device)

    for i, sentence in enumerate(text):
        for j, token in enumerate(sentence):
            words[i][j] = word2idx[token]
        mask[i, :len(sentence)] = 1
    return words, mask