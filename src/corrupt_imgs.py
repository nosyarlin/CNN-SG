import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import LABELS_FILEPATH, IMAGE_DIR


print("Checking for corrupt images now...")
img_datasheet = pd.read_csv(LABELS_FILEPATH)

corrupt_imgs = []
for filename in tqdm(img_datasheet.FileName):
    try:
        path = os.path.join(IMAGE_DIR, filename)
        img = Image.open(path)
        img.verify()
    except (IOError, SyntaxError):
        corrupt_imgs.append(filename)
        print('\nBad file:', filename, '\n')

if corrupt_imgs is not None:
    corrupt_imgs_pd = pd.DataFrame(corrupt_imgs, columns=['FileName'])
    corrupt_imgs_path = os.path.splitext(LABELS_FILEPATH)[0] + '_corrupt_imgs.csv'
    corrupt_imgs_pd.to_csv(corrupt_imgs_path)
    print("File names of corrupt images saved at {}.".format(corrupt_imgs_path))
