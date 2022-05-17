from config import LABELS_FILEPATH, IMAGE_DIR
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd


img_datasheet = pd.read_csv(LABELS_FILEPATH)

for filename in tqdm(img_datasheet.FileName):
    try:
        path = os.path.join(IMAGE_DIR, filename)
        img = Image.open(path)
        img.verify()
    except (IOError, SyntaxError):
        print('\nBad file:', filename, '\n')
