import os
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import concurrent.futures
from config import LABELS_FILEPATH, PREPROCESSED_IMAGE_DIR, IMAGE_DIR
from config import IMAGE_SIZE, PARALLEL, N_CORES


## Global Settings 
# Allow PIL to be tolerant to truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True


## Functions
def resize_img(FileName, FileName_resized):
    path = os.path.join(IMAGE_DIR, FileName)
    image = Image.open(path)
    w, h = image.size

    if w < h:
        new_h = int(IMAGE_SIZE * h / w)
        image_resized = image.resize((IMAGE_SIZE, new_h))
    else:
        new_w = int(IMAGE_SIZE * w / h)
        image_resized = image.resize((new_w, IMAGE_SIZE))

    image_resized.save(FileName_resized)


if __name__ == '__main__':
    # Creating the datasheet
    input_sheet = pd.read_csv(LABELS_FILEPATH)

    if os.path.exists(input_sheet.FileName.iloc[0]): #filename is a full path
        input_sheet['FileName_resized'] = input_sheet.FileName.map(
            lambda filename: os.path.join(PREPROCESSED_IMAGE_DIR, os.path.basename(filename))
        )
    else:
        input_sheet['FileName_resized'] = input_sheet.FileName.map(
            lambda filename: os.path.join(PREPROCESSED_IMAGE_DIR, filename)
        )

    # Resizing and saving the images
    print("Resizing images by making the shorter of width or height {} pixels. " \
        "Aspect ratio of each image is thus maintained.".format(IMAGE_SIZE))
    
    if PARALLEL:
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
            list(tqdm(executor.map(
                resize_img, input_sheet.FileName, input_sheet.FileName_resized),
                total=len(input_sheet)))

    else:
        for i in tqdm(range(len(input_sheet))):
            resize_img(i, input_sheet)
    
    # Saving out the datasheet
    input_sheet['FileName'] = input_sheet['FileName_resized']
    input_sheet = input_sheet.drop(columns='FileName_resized')
    input_sheet.to_csv(index=False, path_or_buf=LABELS_FILEPATH[:len(
        LABELS_FILEPATH) - 4] + '_resized.csv')
