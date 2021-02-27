from config import ROOT_DIR
from dataset import get_dataloader
from models import get_model
from shared_funcs import evaluate_model, load_checkpoint, read_csv
from torch import nn
import argparse
import os
import pandas as pd
import torch


def get_arg_parser():
    default_model_path = "E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Resnet50_FYP/AllLayer_propTrain=0.7/run_2/model.pth"
    default_X_test = os.path.join(ROOT_DIR, 'data', 'splits', 'X_test_sbwr_phase1.csv')
    default_y_test = os.path.join(ROOT_DIR, 'data', 'splits', 'Y_test_sbwr_phase1.csv')
    default_save_path = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/SBWR_Phase1/Train_2021-01-25_32c9a04cdece4e4cacdb89f5d2c3f542'

    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        default='C:/_for-temp-data-that-need-SSD-speed/SBWR_20191127-20200120/',
        help='Path to the directory containing the images'
    )
    parser.add_argument(
        '--model_path',
        default=default_model_path,
        help='Path to saved model weights'
    )
    parser.add_argument(
        '--X_test', default=default_X_test,
        help='Path to X data'
    )
    parser.add_argument(
        '--y_test', default=default_y_test,
        help='Path to y data'
    )
    parser.add_argument(
        '--path_to_save_results',
        default=default_save_path,
        help='Path to the directory to save the model, hyperparameters and results'
    )
    parser.add_argument(
        '--archi', default='resnet50',
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
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    # Get test data
    X_test = read_csv(args.X_test)
    y_test = read_csv(args.y_test)
    test_dl = get_dataloader(
        X_test, y_test, args.batch_size, args.image_dir,
        args.img_size, args.crop_size, False
    )

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

    # Saving results and probabilities
    probabilities = probabilities.T.tolist()
    test_probs_df = pd.DataFrame({
        'file_name': args.X_test,
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
