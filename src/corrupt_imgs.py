import os
from PIL import Image
import pandas as pd
from config import ROOT_DIR
from tqdm import tqdm


img_datasheet_path = os.path.join(
    ROOT_DIR, 'data', 'splits', 'big4_20210810_train_sheet.csv')
img_datasheet = pd.read_csv(img_datasheet_path)

for filename in tqdm(img_datasheet.FileName):
    try:
        img = Image.open(filename)  # open the image file
        img.verify()  # verify that it is, in fact an image
    except (IOError, SyntaxError):
        # print out the names of corrupt files
        print('\nBad file:', filename, '\n')
