import os
from config import ROOT_DIR
import pandas as pd
from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    # Setting Inputs
    img_datasheet = os.path.join(ROOT_DIR, 'data', 'splits', 'big4_20210810_train_sheet.csv')
    img_save_path = 'C:/_for-temp-data-that-need-SSD-speed/big4_datasets/train_val_sets/train_imgs_resized/'
    img_size = 360

    # Creating the datasheet
    input_sheet = pd.read_csv(img_datasheet)
    input_sheet['FileName_resized'] = img_save_path + input_sheet['Media.Filename'] + '.jpg'

    # Resizing and saving the images
    for i in tqdm(range(len(input_sheet))):
        image = Image.open(input_sheet.FileName[i])
        w, h = image.size

        if w < h:
            new_h = int(img_size*h/w)
            image_resized = image.resize((img_size, new_h))
        else:
            new_w = int(img_size*w/h)
            image_resized = image.resize((new_w, img_size))
        
        image_resized.save(input_sheet.FileName_resized[i])
    
    # Saving out the datasheet
    input_sheet['FileName'] = input_sheet['FileName_resized']
    input_sheet = input_sheet.drop('FileName_resized', 1)
    input_sheet.to_csv(index=False, path_or_buf=img_datasheet[:len(img_datasheet)-4] + '_resized.csv')