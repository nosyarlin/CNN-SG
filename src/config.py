import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).parent.parent
IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'images')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
# PREPROCESSED_IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'preprocessed_images')
PREPROCESSED_IMAGE_DIR = 'C:\\temp_for_speed\\CNN-SG\\CCNR_test'
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'big3_20221006')
SPLITS_DIR = os.path.join(ROOT_DIR, 'data', 'splits')

# File paths
LABELS_FILEPATH = os.path.join(ROOT_DIR, 'data', '20220922_ccnr_test.csv')
TRAIN_FILEPATH = os.path.join(SPLITS_DIR, '20220922_big3_train_resized.csv')
VAL_FILEPATH = os.path.join(SPLITS_DIR, '20220922_big3_val_resized.csv')
TEST_FILEPATH = os.path.join(SPLITS_DIR, '20220922_ccnr_test_resized.csv')
MODEL_FILEPATH = None
# MODEL_FILEPATH = os.path.join(MODEL_DIR, 'model.pth')
HYPERPARAMETERS_FILEPATH = os.path.join(RESULTS_DIR, 'hyperparameters.csv')

# Settings for resize.py
IMAGE_SIZE = 360
PARALLEL = True
N_CORES = 7

# Settings for train.py
ARCHI = 'resnet50'
NUM_CLASSES = 3
DROPOUT = 0.01
LEARNING_RATE = 0.0005
BETADIST_ALPHA = 0.9
BETADIST_BETA = 0.99
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-8
EPOCHS = 1
STEP_SIZE = 5
GAMMA = 0.1
BATCH_SIZE = 32
