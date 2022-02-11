import os
import torch
import pandas as pd
from torchsummary import summary
from models import get_model
from shared_funcs import load_checkpoint


## Load ResNet50 hyperparameters and model
resnet50_dir_path = "D:/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Big4/Re_0.7_09/trained_model/"
resnet50_model_path = os.path.join(resnet50_dir_path, 'archi_resnet50_train_acc_0.898_val_acc_0.927_epoch_15.pth')
resnet50_hp_path = os.path.join(resnet50_dir_path, 'hyperparameter_records.csv')

hp = pd.read_csv(resnet50_hp_path)
resnet50_archi = hp.loc[hp['Hyperparameters'] == 'Architecture', 'Values'].item()
resnet50_num_classes = int(hp.loc[hp['Hyperparameters'] == 'NumClasses', 'Values'].item())
resnet50_dropout = float(hp.loc[hp['Hyperparameters'] == 'Dropout', 'Values'].item())
resnet50_crop_size = int(hp.loc[hp['Hyperparameters'] == 'CropSize', 'Values'].item())

resnet50_model, _ = get_model(
    resnet50_archi, resnet50_num_classes, False, False, resnet50_dropout
)
load_checkpoint(resnet50_model_path, resnet50_model)

resnet50_model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Summary of ResNet50 model
summary(resnet50_model, input_size = (3, resnet50_crop_size, resnet50_crop_size)) #input_size = (channel, input_image_width, input_image_height)


## Load MegaDetector model
