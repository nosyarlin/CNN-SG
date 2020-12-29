# Variable control panel
y_fpath = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/Animal-Classification/data/labels.csv'
image_dir = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/Animal-Classification/data/images'
path_to_save_model = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Resnet50/model.pth'
path_to_save_trainval_results = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Resnet50/train_val_results.csv'
path_to_save_test_results = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Resnet50/test_results.csv'

if __name__ == '__main__':
    exec(open("dataset.py").read())
    exec(open("train.py").read())