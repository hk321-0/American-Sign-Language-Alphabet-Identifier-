'''
Data Loading

Files used from ./asl_data/:
    sign_mnist_train.csv
    sign_mnist_test.csv
'''

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Sign Language MNIST excludes J (index 9) and Z (index 25) because both are
# motion-based signs that can't be captured in a single static image.
# After remap_labels() shifts everything above 9 down by one, these 24 names
# line up with the dense 0..23 label range.
LABEL_NAMES = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("J", "Z")]


def loading_dataset():
    # i/p: Raw data in csv files
    # o/p: shape(X) = (N, 28, 28) uint8, shape(y) = (N,) int64
    train_df = pd.read_csv('asl_data/sign_mnist_train.csv')
    test_df = pd.read_csv('asl_data/sign_mnist_test.csv')

    X_train = train_df.drop(columns=["label"]).to_numpy().astype(np.uint8).reshape(-1, 28, 28)
    y_train = train_df["label"].to_numpy().astype(np.int64)

    X_test = test_df.drop(columns=["label"]).to_numpy().astype(np.uint8).reshape(-1, 28, 28)
    y_test = test_df["label"].to_numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


def remap_labels(y_train, y_test):
    # shifting labels K-Y by 1 since j is excluded from dataset due to motion based sign
    y_train = np.where(y_train > 9, y_train - 1, y_train)
    y_test = np.where(y_test > 9, y_test - 1, y_test)
    return y_train, y_test


def compute_stats(X_train):
    # mean/std of pixel values in [0,1], from TRAINING data only
    mean = np.mean(X_train.astype(np.float32) / 255.0)
    std = np.std(X_train.astype(np.float32) / 255.0)
    return mean, std


class SignsDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = X   # (N, 28, 28) uint8
        self.y = y   # (N,) int64
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # (28,28) uint8 -> (1,28,28) float32, normalized -- done directly with
        # numpy/torch so this file no longer depends on torchvision at all.
        img = self.X[idx].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).unsqueeze(0)
        return img, int(self.y[idx])


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):   # creating train, validation and test loaders
    mean, std = compute_stats(X_train)

    train_loader = DataLoader(SignsDataset(X_train, y_train, mean, std), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SignsDataset(X_val, y_val, mean, std), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SignsDataset(X_test, y_test, mean, std), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
