import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import LABELS_FILEPATH, IMAGE_DIR


img_datasheet = pd.read_csv(LABELS_FILEPATH)

for filename in tqdm(img_datasheet.FileName):
    try:
        path = os.path.join(IMAGE_DIR, filename)
        img = Image.open(path)
        img.verify()
    except (IOError, SyntaxError):
        print('\nBad file:', filename, '\n')
