from config import ROOT_DIR
from dataset import get_dataloader
from models import get_model
from shared_funcs import read_csv, evaluate_model
from torch import nn
import argparse
import os
import pandas as pd
import torch


def get_arg_parser():
    default_model_path = os.path.join(ROOT_DIR, 'models', 'model.pth')

    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        default='C:/_for-temp-data-that-need-SSD-speed/ProjectMast_FYP_Media',
        help='Path to the directory containing the images'
    )
    parser.add_argument(
        '--model_path',
        default=default_model_path,
        help='Path to saved model weights'
    )
    parser.add_argument(
        '--path_to_save_results',
        default='E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Inception_FYP/AllLayer_propTrain=0.7/run_5/',
        help='Path to the directory to save the model, hyperparameters and results'
    )
    parser.add_argument(
        '--archi', default='inception',
        help='Architecture of the model to be trained. Either inception, resnet50, resnet101, resnet152, wide_resnet50, or mobilenet')
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='Using CPU for processing')
    parser.add_argument(
        '--num_classes', default='3', type=int, action='store',
        help='Number of classes to be trained')
    parser.add_argument(
        '--dropout', default='0.1', type=float,
        action='store', help='Dropout probablity')
    parser.add_argument(
        '--batch_size', default='32', type=int,
        help='Batch size for training')
    parser.add_argument(
        '--img_size', default='360', type=int,
        help='Image size for each image')
    parser.add_argument(
        '--crop_size', default='299', type=int,
        help='Crop size for each image. Inception v3 expects 299')


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    # Get test data
    splits_dir = os.path.join(ROOT_DIR, 'data', 'splits')
    X_test = read_csv(os.path.join(splits_dir, 'X_test.csv'))
    y_test = read_csv(os.path.join(splits_dir, 'y_test.csv'))
    test_dl = get_dataloader(
        X_test, y_test, args.batch_size, args.image_dir,
        args.img_size, args.crop_size, False
    )

    # Create Model
    model, _ = get_model(
        args.archi, args.num_classes, False, False, args.dropout
    )
    model.load_state_dict(torch.load(args.model_path))

    # Run data through model
    if not args.use_cpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loss_func = nn.CrossEntropyLoss()
    test_acc, test_loss, probabilities = evaluate_model(
        model, test_dl, loss_func, device, 'Testing'
    )

    # Saving results and probabilities
    probabilities = probabilities.T.tolist()
    test_probs_df = pd.DataFrame({
        'file_name': X_test,
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
