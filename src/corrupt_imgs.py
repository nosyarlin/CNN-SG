from PIL import Image
import pandas as pd
from config import LABELS_FILEPATH
from tqdm import tqdm


img_datasheet = pd.read_csv(LABELS_FILEPATH)

for filename in tqdm(img_datasheet.FileName):
    try:
        img = Image.open(filename)
        img.verify()
    except (IOError, SyntaxError):
        print('\nBad file:', filename, '\n')
