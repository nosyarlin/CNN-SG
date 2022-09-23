import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from config import LABELS_FILEPATH, PREPROCESSED_IMAGE_DIR, IMAGE_DIR, IMAGE_SIZE


def resize_img(i, input_sheet):
    path = os.path.join(IMAGE_DIR, input_sheet.FileName[i])
    image = Image.open(path)
    w, h = image.size

    if w < h:
        new_h = int(IMAGE_SIZE * h / w)
        image_resized = image.resize((IMAGE_SIZE, new_h))
    else:
        new_w = int(IMAGE_SIZE * w / h)
        image_resized = image.resize((new_w, IMAGE_SIZE))

    image_resized.save(input_sheet.FileName_resized[i])


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
    
    results = []
    def callback_func(result):
        results.append(result)
        pbar.update()

    pool = mp.Pool(processes=mp.cpu_count())
    pbar = tqdm(total = len(input_sheet))

    for i in range(len(input_sheet)):
        pool.apply_async(resize_img, args=(i, input_sheet), callback=callback_func)

    pool.close()
    pool.join()
    
    # Saving out the datasheet
    input_sheet['FileName'] = input_sheet['FileName_resized']
    input_sheet = input_sheet.drop(columns='FileName_resized')
    input_sheet.to_csv(index=False, path_or_buf=LABELS_FILEPATH[:len(
        LABELS_FILEPATH) - 4] + '_resized.csv')
