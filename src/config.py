import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).parent.parent
IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'images')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
# PREPROCESSED_IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'preprocessed_images')
PREPROCESSED_IMAGE_DIR = 'C:\\temp_for_speed\\CNN-SG\\Big3_train'
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
SPLITS_DIR = os.path.join(ROOT_DIR, 'data', 'splits')

# File paths
LABELS_FILEPATH = os.path.join(ROOT_DIR, 'data', '20220922_big3_train.csv')
MODEL_FILEPATH = os.path.join(MODEL_DIR, 'model.pth')
HYPERPARAMETERS_FILEPATH = os.path.join(RESULTS_DIR, 'hyperparameters.csv')

# Settings for resize.py
IMAGE_SIZE = 360