import torch
from torch.utils.data import TensorDataset, DataLoader


def build_vocab_from_text(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_text(text, stoi):
    return [stoi[ch] for ch in text]


def decode_indices(indices, itos):
    return ''.join([itos[i] for i in indices])


def build_dataset(data_indices, context_len):
    X, Y = [], []
    for i in range(context_len, len(data_indices)):
        X.append(data_indices[i - context_len:i])
        Y.append(data_indices[i])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


def make_loaders(X_train, Y_train, X_val, Y_val, batch_size=128, shuffle=True, num_workers=0):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader