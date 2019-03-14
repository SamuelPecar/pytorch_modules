import pandas as pd


def load_data_file(data_file, sep='\t', header=0, **kwargs):
    data = pd.read_csv(data_file, sep=sep, header=header).sample(frac=1).values

    train_split = int(len(data) * 0.9)
    valid_split = int(len(data) * 0.95)

    train = data[:train_split]
    valid = data[train_split:valid_split]
    test = data[valid_split:]

    return train, valid, test


def load_data(train_file, validation_file, test_file, sep=',', header=0, **kwargs):
    train = pd.read_csv(train_file, sep=sep, header=header, quoting=1).sample(frac=1).values
    valid = pd.read_csv(validation_file, sep=sep, header=header, quoting=1).sample(frac=1).values
    test = pd.read_csv(test_file, sep=sep, header=header, quoting=1).values

    return train, valid, test
