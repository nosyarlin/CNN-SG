from config import LABELS_FILEPATH, PREPROCESSED_IMAGE_DIR, IMAGE_DIR
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd


if __name__ == '__main__':
    # Setting Inputs
    img_size = 360

    # Creating the datasheet
    input_sheet = pd.read_csv(LABELS_FILEPATH)
    input_sheet['FileName_resized'] = input_sheet.FileName.map(
        lambda filename: os.path.join(PREPROCESSED_IMAGE_DIR, filename)
    )

    # Resizing and saving the images
    for i in tqdm(range(len(input_sheet))):
        path = os.path.join(IMAGE_DIR, input_sheet.FileName[i])
        image = Image.open(path)
        w, h = image.size

        if w < h:
            new_h = int(img_size * h / w)
            image_resized = image.resize((img_size, new_h))
        else:
            new_w = int(img_size * w / h)
            image_resized = image.resize((new_w, img_size))

        image_resized.save(input_sheet.FileName_resized[i])

    # Saving out the datasheet
    input_sheet['FileName'] = input_sheet['FileName_resized']
    input_sheet = input_sheet.drop(columns='FileName_resized')
    input_sheet.to_csv(index=False, path_or_buf=LABELS_FILEPATH[:len(
        LABELS_FILEPATH) - 4] + '_resized.csv')
