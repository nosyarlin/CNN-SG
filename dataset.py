import numpy as np
import os
import torch
from collections import defaultdict
from PIL import Image
from shared_funcs import write_to_csv
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import transforms


class ImageDataset(data.Dataset):
    def __init__(self, image_dir, img_size, crop_size, X, y):
        self.image_dir = image_dir
        self.X = X
        self.y = torch.LongTensor([int(i) for i in y])
        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=2),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        fname = os.path.join(self.image_dir, self.X[idx])
        img = Image.open(fname)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.y[idx]

        return self.transforms(img), label


def get_labels(y_fpath: str, delimiter=","):
    """
    Returns dictionary of labels
    Parameters:
        y_fpath: Filepath to csv file with labels
        delimiter: Delimiter used in csv file

    Returns:
        y (dict): Mapping from image filenames to labels
    """
    y = {}
    with open(y_fpath) as f:
        for line in f:
            temp = line.split(delimiter)
            y[temp[0]] = int(temp[1])
    return y


def get_labels_for_images(image_dir: str, y_fpath: str, delimiter=","):
    """
    Get labels for images stored in image_dir
    Parameters:
        image_dir: Filepath to directory with images
        y_fpath: Filepath to csv file with labels
        delimiter: Delimiter used in csv file

    Returns:
        y (dict): Mapping from image filenames to labels
    """
    filenames = os.listdir(image_dir)
    labels = get_labels(y_fpath, delimiter)
    y = {}
    for filename in filenames:
        if filename in labels:
            y[filename] = labels[filename]
    return y


def balance_labels(labels: dict):
    """
    Returns balanced dictionary of labels by randomly sampling the same number
    of observations for each y value

    Parameters:
        labels: Mapping from image filenames to labels

    Returns:
        y (dict): Balanced mapping from image filenames to labels
    """
    # Split file names by labels
    x = defaultdict(list)
    for fname, label in labels.items():
        x[label].append(fname)

    # Build balanced mapping
    unique_labels, counts = np.unique(
        list(labels.values()),
        return_counts=True
    )
    lowest_count = np.min(counts)
    y = {}
    for label in unique_labels:
        samples = np.random.choice(x[label], lowest_count, replace=False)
        for fname in samples:
            y[fname] = labels[fname]
    return y


def get_splits(image_dir: str, y_fpath: str, test_size: float, validation_size: float):
    """
    Splits data into train and test sets
    Parameters:
        y_fpath: Filepath to csv file with labels
        test_size: Percentage of data to be in test set
        validation_size: Percentage of data to be in validation set
    """
    labels = get_labels_for_images(image_dir, y_fpath)
    labels = balance_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(list(labels.keys()),
                                                        list(labels.values()),
                                                        test_size=test_size,
                                                        stratify=list(labels.values()))
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=(
                                                          validation_size / (1.0 - test_size)
                                                      ),
                                                      stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataloader(x, y, batch_size, image_dir, img_size, crop_size):
    return data.DataLoader(ImageDataset(image_dir, img_size, crop_size, x, y),
                           batch_size=batch_size)


if __name__ == '__main__':
    y_fpath = './data/labels.csv'
    image_dir = './data/images'

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        image_dir,
        y_fpath,
        0.1,
        0.25,
    )

    files = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
             'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
    for file, obj in files.items():
        write_to_csv(obj, '{}.csv'.format(file))