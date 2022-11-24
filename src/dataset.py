import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms
from collections import defaultdict
from shared_funcs import write_to_csv
from sklearn.model_selection import train_test_split
from config import (
    SPLITS_DIR, PREPROCESSED_IMAGE_DIR, LABELS_FILEPATH, IMAGE_DIR)


class ImageDataset(data.Dataset):
    def __init__(
        self, image_dir, crop_size,
        X, y, is_train, img_resize, img_size
    ):
        self.image_dir = image_dir
        self.X = X
        self.y = torch.LongTensor([int(i) for i in y])
        if img_resize:
            if not is_train:
                self.transforms = transforms.Compose([
                    transforms.Resize(img_size, 
                        interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda(batch_to_tensor),
                    transforms.Lambda(batch_to_normalize)
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(img_size, 
                        interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.RandomGrayscale(),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
        else:
            if not is_train:
                self.transforms = transforms.Compose([
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda(batch_to_tensor),
                    transforms.Lambda(batch_to_normalize)
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.RandomGrayscale(),
                    transforms.RandomHorizontalFlip(),
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
        # fname = os.path.join(self.image_dir, self.X[idx])
        fname = self.X[idx]
        img = Image.open(fname)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.y[idx]

        return self.transforms(img), label


def batch_to_tensor(crops):
    return torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])


def batch_to_normalize(crops):
    return torch.stack(
        [transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(crop) for crop in crops])


def get_labels(y_fpath: str):
    """
    Returns dictionary of labels from labels.csv

    Parameters:
        y_fpath: Filepath to csv file with labels

    Returns:
        y (dict): Mapping from image filenames to labels
    """
    input = pd.read_csv(y_fpath)
    y = dict(zip(input['FileName'], input['SpeciesCode']))
    
    return y


def get_labels_for_images(image_dir: str, y_fpath: str): 
    """
    Get labels for images stored in image_dir. Ensures 
    that image names in labels.csv is present in the 
    image directory.

    Parameters:
        image_dir: Filepath to directory with images
        y_fpath: Filepath to csv file with labels

    Returns:
        y (dict): Mapping from image filenames to labels
    """
    # Get image file names recursively
    filenames = []
    for _, _, files in os.walk(image_dir):
        for name in files:
            if name.endswith('.jpg'):
                filenames.append(name)

    # Get labels for the images that we have
    labels = get_labels(y_fpath)
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


def get_image_date(filename):
    """
    Reads date of which image was taken from exif data
    """
    path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(path)
    return image.getexif().get(306)


def get_splits(
    image_dir: str, y_fpath: str, test_size: float, validation_size: float
):
    """
    Splits data into train and test sets
    Parameters:
        y_fpath: Filepath to csv file with labels
        test_size: Percentage of data to be in test set
        validation_size: Percentage of data to be in validation set
    """
    labels = get_labels_for_images(image_dir, y_fpath)
    
    # Chop latest test_size for test
    sorted_filenames = sorted(labels.keys(), key=get_image_date)
    num_test_files = round(len(sorted_filenames) * test_size)
    test_filenames = sorted_filenames[-num_test_files:]
    train_labels, test_labels = {}, {}
    for filename in labels:
        if filename in test_filenames:
            test_labels[filename] = labels[filename]
        else:
            train_labels[filename] = labels[filename]
    test_labels = balance_labels(test_labels)
    X_test, y_test = list(test_labels.keys()), list(test_labels.values())

    # Standard train_test_split for train and validation
    train_labels = balance_labels(train_labels)
    X_train, X_val, y_train, y_val = train_test_split(
        list(train_labels.keys()),
        list(train_labels.values()),
        test_size=(validation_size / (1.0 - test_size)),
        stratify=list(train_labels).values()
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataloader(
    x, y, batch_size, image_dir, crop_size,
    is_train, num_workers, img_resize, img_size
):
    if is_train:
        return data.DataLoader(
            ImageDataset(
                image_dir, crop_size, x, y,
                is_train, img_resize, img_size
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    else:
        return data.DataLoader(
            ImageDataset(
                image_dir, crop_size, x, y,
                is_train, img_resize, img_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )


if __name__ == '__main__':
    prop_test = 0.15
    prop_val = 0.15

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        PREPROCESSED_IMAGE_DIR,
        LABELS_FILEPATH,
        prop_test,
        prop_val,
    )

    files = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
             'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
    for file, obj in files.items():
        path = os.path.join(SPLITS_DIR, '{}.csv'.format(file))
        write_to_csv(obj, path)

    print(
        "Dataset has been split with the following proportions:\
        {} train, {} val, {} test".format(
            1 - prop_test - prop_val,
            prop_val,
            prop_test
        )
    )
