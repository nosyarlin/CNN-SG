from config import ROOT_DIR
from dataset import get_dataloader
from models import get_model
from shared_funcs import evaluate_model, load_checkpoint, check_img_size
from torch import nn
import argparse
import os
import pandas as pd
import torch
import sys


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        default=default_image_dir,
        help='Path to the directory containing the images')
    parser.add_argument(
        '--model_path',
        default=default_model_path,
        help='Path to saved model weights')
    parser.add_argument(
        '--xy_test', default=default_xy_test,
        help='Path to xy dataframe')
    parser.add_argument(
        '--path_to_save_results',
        default=default_save_path,
        help='Path to the directory to save the model, hyperparameters and results')
    parser.add_argument(
        '--archi', default=default_archi,
        help='Architecture of the model to be trained. Either inception, resnet50, resnet101, resnet152, wide_resnet50, or mobilenet')
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='Using CPU for processing')
    parser.add_argument(
        '--num_workers', default='4', type=int,
        help='Number of threads to be set for the GPU')
    parser.add_argument(
        '--num_classes', default=default_num_classes, type=int, action='store',
        help='Number of classes to be trained')
    parser.add_argument(
        '--dropout', default=default_dropout, type=float,
        action='store', help='Dropout probablity')
    parser.add_argument(
        '--batch_size', default=default_batch_size, type=int,
        help='Batch size for training')
    parser.add_argument(
        '--img_resize', action='store_true',
        help='Resize each image before training/testing')
    parser.add_argument(
        '--img_size', default=default_img_size, type=int,
        help='Image size for each image')
    parser.add_argument(
        '--crop_size', default=default_crop_size, type=int,
        help='Crop size for each image. Inception v3 expects 299')
    return parser


if __name__ == '__main__':
    saved_model_path = "E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Big4/Re_0.7_09/trained_model/"
    default_model_path = os.path.join(
        saved_model_path, 'archi_resnet50_train_acc_0.898_val_acc_0.927_epoch_15.pth')
    default_save_path = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/CCNR_JB/class_jb_batch4'

    default_image_dir = 'C:/_for-temp-data-that-need-SSD-speed/'
    default_xy_test = os.path.join(
        ROOT_DIR, 'data', 'splits', 'cnn_class_jb_batch4.csv')

    default_hp_path = os.path.join(
        saved_model_path, 'hyperparameter_records.csv')

    # Get hyperparameters of the saved model
    hp = pd.read_csv(default_hp_path)
    default_archi = hp.loc[hp['Hyperparameters']
                           == 'Architecture', 'Values'].item()
    default_num_classes = hp.loc[hp['Hyperparameters']
                                 == 'NumClasses', 'Values'].item()
    default_dropout = hp.loc[hp['Hyperparameters']
                             == 'Dropout', 'Values'].item()
    default_batch_size = hp.loc[hp['Hyperparameters']
                                == 'BatchSize', 'Values'].item()
    default_img_size = hp.loc[hp['Hyperparameters']
                              == 'ImgSize', 'Values'].item()
    default_crop_size = hp.loc[hp['Hyperparameters']
                               == 'CropSize', 'Values'].item()

    parser = get_arg_parser()
    args = parser.parse_args()

    # Check that paths to save results and models exist
    if os.path.exists(args.path_to_save_results) and len(os.listdir(args.path_to_save_results)) == 0:
        print("\nSaving results in " + args.path_to_save_results + "\n")
    else:
        sys.exit(
            "\nError: File path to save results do not exist, or directory is not empty")

    # Get test data
    xy_test = pd.read_csv(args.xy_test)
    if not args.img_resize:
        check_img_size(xy_test.FileName, "testing set", args.img_size)

    test_dl = get_dataloader(
        xy_test.FileName, xy_test.SpeciesCode, args.batch_size, args.image_dir,
        args.crop_size, False, args.num_workers, args.img_resize, args.img_size
    )

    print("Dataset to be used includes {} testing images.".format(
        len(xy_test.FileName)))

    # Create Model
    model, _ = get_model(
        args.archi, args.num_classes, False, False, args.dropout
    )
    load_checkpoint(args.model_path, model)

    # Run data through model
    if not args.use_cpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if torch.backends.cudnn.is_available():
        print("\nUsing {} with cuDNN version {} for testing with {} architecture.".format(
            device, torch.backends.cudnn.version(), args.archi))
    else:
        print("\nUsing {} WITHOUT cuDNN for testing with {} architecture.".format(
            device, args.archi))

    loss_func = nn.CrossEntropyLoss()
    test_acc, test_loss, probabilities = evaluate_model(
        model, test_dl, loss_func, device, 'Testing'
    )
    print("\nTesting complete. Test acc: {}, Test loss: {}\n".format(
        test_acc, test_loss))

    # Saving results, probabilities and hyperparameters
    probabilities = probabilities.T.tolist()
    test_probs_df = pd.DataFrame({
        'file_name': xy_test.FileName,
        'prob_empty': probabilities[0],
        'prob_human': probabilities[1],
        'prob_animal': probabilities[2]}
    )
    test_probs_df.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.path_to_save_results,
            'test_probabilities.csv'
        )
    )

    test_results_df = pd.DataFrame({'Acc': [test_acc], 'Loss': [test_loss]})
    test_results_df.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.path_to_save_results,
            'test_results.csv'
        )
    )

    hp_names = (
        "TrainedModel", "ModelID", "TestSet", "TestSetSize")
    hp_values = (
        args.model_path, args.path_to_save_results.split("/")[-1], args.xy_test.split("\\")[-1], len(xy_test.FileName))

    hp2 = pd.DataFrame(
        {'Hyperparameters': hp_names, 'Values': hp_values})
    hp_records = hp.append(hp2)
    hp_records.to_csv(index=False, path_or_buf=os.path.join(
        args.path_to_save_results, 'testing_hyperparameter_records.csv'))
