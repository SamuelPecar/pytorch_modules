import pandas as pd


def load_data_file_without_split(data_file, sep='\t', header=0, **kwargs):
    # data = pd.read_csv(data_file, sep=sep, header=header).sample(frac=1).values
    data = pd.read_csv(data_file, sep=sep, header=header).values
    return data


def split_data(data, train_split=None, valid_split=None):
    if not train_split:
        train_split = int(len(data) * 0.8)
    if not valid_split:
        valid_split = int(len(data) * 0.9)

    train = data[:train_split]
    valid = data[train_split:valid_split]
    test = data[valid_split:]

    # return train[:1000], valid[:1000], test[:1000]
    return train, valid, test


def load_data_file(data_file, sep='\t', header=0, **kwargs):
    data = pd.read_csv(data_file, sep=sep, header=header).sample(frac=1).values

    train_split = int(len(data) * 0.9)
    valid_split = int(len(data) * 0.95)

    train = data[:train_split]
    valid = data[train_split:valid_split]
    test = data[valid_split:]

    return train, valid, test


def load_data(train_file, validation_file, test_file, sep=',', header=0, **kwargs):
    # .sample(frac=1)
    train = pd.read_csv(train_file, sep=sep, header=header, quoting=1).values
    valid = pd.read_csv(validation_file, sep=sep, header=header, quoting=1).values
    test = pd.read_csv(test_file, sep=sep, header=header, quoting=1).values

    # return train[:1000], valid[:1000], test[:1000]
    return train, valid, test
