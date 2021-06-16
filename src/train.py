from clearml import Task
from config import ROOT_DIR
from dataset import get_dataloader
from datetime import date
from hanging_threads import start_monitoring
from models import get_model
from shared_funcs import (
    read_csv, write_to_csv, train_validate, evaluate_model,
    save_checkpoint, load_checkpoint
)
from torch import nn, optim
import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        # default='C:/_for-temp-data-that-need-SSD-speed/ProjectMast_FYP_Media',
        default='C:/_for-temp-data-that-need-SSD-speed/SBWR_20191127-20200120',
        help='Path to the directory containing the images'
    )
    parser.add_argument(
        '--path_to_save_results',
        # default='E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Resnet50_FYP/AllLayer_propTrain=0.7/run_9',
        default='E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/SBWR_Phase1/Mo_0.7_06/extra',
        help='Path to the directory to save the model, hyperparameters and results'
    )
    parser.add_argument(
        '--model_path',
        # default=None,
        default='E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/SBWR_Phase1/Mo_0.7_06/FT_6Wks/archi_mobilenet_train_acc_0.759_val_acc_0.862_epoch_7.pth',
        help='Path to saved model weights. If this is set, we will use the provided weights as the starting point for training'
    )
    parser.add_argument(
        '--skip_test', action='store_true',
        help='Set if testing should be skipped')
    parser.add_argument(
        '--archi', default='mobilenet',
        help='Architecture of the model to be trained. Either inception, resnet50, resnet101, resnet152, wide_resnet50, or mobilenet')
    parser.add_argument(
        '--no_pretraining', action='store_true',
        help='Set if you want the model to be trained from scratch')
    parser.add_argument(
        '--train_only_classifier', action='store_true',
        help='Set if we train classification layer only')
    parser.add_argument(
        '--use_cpu', action='store_true',
        help='Using CPU for processing')
    parser.add_argument(
        '--num_classes', default='3', type=int,
        action='store', help='Number of classes to be trained')
    parser.add_argument(
        '--dropout', default='0.01', type=float,
        action='store', help='Dropout probablity')
    parser.add_argument(
        '--lr', default='0.0005', type=float, help='The learning rate')
    parser.add_argument(
        '--betadist_alpha', default=0.9, type=float,
        help='The alpha value controlling the shape of the beta distribution for the Adam optimiser')
    parser.add_argument(
        '--betadist_beta', default=0.99, type=float,
        help='The beta value controlling the shape of the beta distribution for the Adam optimiser')
    parser.add_argument(
        '--eps', default='1e-8', type=float,
        help='Epsilon value for Adam optimiser')
    parser.add_argument(
        '--weight_decay', default='1e-8', type=float,
        help='Weight decay for Adam optimiser')
    parser.add_argument(
        '--epochs', default='5', type=int,
        help='Number of epochs to be run for training')
    parser.add_argument(
        '--step_size', default='5', type=int,
        help='Step size')
    parser.add_argument(
        '--gamma', default='0.1', type=float,
        help='Gamma value for optimiser')
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
    # Monitoring for hangs
    # monitoring_thread = start_monitoring(seconds_frozen=60)

    # Connecting to the clearml dashboard
    task = Task.init(project_name="Nosyarlin", task_name="Train_" + date.today().strftime('%Y-%m-%d'),
                     task_type=Task.TaskTypes.training)

    # Set hyperparameters
    parser = get_arg_parser()
    args = parser.parse_args()

    # Check that paths to save results and models exist
    if os.path.exists(args.path_to_save_results) and len(os.listdir(args.path_to_save_results)) == 0:
        print("\nSaving results in " + args.path_to_save_results)
    else:
        sys.exit(
            "\nError: File path to save results do not exist, or directory is not empty")

    # Read data
    splits_dir = os.path.join(ROOT_DIR, 'data', 'splits')
    
    # X_train = read_csv(os.path.join(splits_dir, 'X_train.csv'))
    # y_train = read_csv(os.path.join(splits_dir, 'y_train.csv'))
    # X_val = read_csv(os.path.join(splits_dir, 'X_val.csv'))
    # y_val = read_csv(os.path.join(splits_dir, 'y_val.csv'))
    # X_test = read_csv(os.path.join(splits_dir, 'X_test.csv'))
    # y_test = read_csv(os.path.join(splits_dir, 'y_test.csv'))
    
    X_train = read_csv(os.path.join(splits_dir, 'X_train_sbwr_phase1.csv'))
    y_train = read_csv(os.path.join(splits_dir, 'Y_train_sbwr_phase1.csv'))
    X_val = read_csv(os.path.join(splits_dir, 'X_val_sbwr_phase1.csv'))
    y_val = read_csv(os.path.join(splits_dir, 'Y_val_sbwr_phase1.csv'))
    X_test = read_csv(os.path.join(splits_dir, 'X_test_sbwr_phase1.csv'))
    y_test = read_csv(os.path.join(splits_dir, 'Y_test_sbwr_phase1.csv'))
    
    train_dl = get_dataloader(
        X_train, y_train, args.batch_size, args.image_dir,
        args.img_size, args.crop_size, True
    )
    val_dl = get_dataloader(
        X_val, y_val, args.batch_size, args.image_dir,
        args.img_size, args.crop_size, False
    )
    test_dl = get_dataloader(
        X_test, y_test, args.batch_size, args.image_dir,
        args.img_size, args.crop_size, False
    )

    print("\nDataset to be used includes {} training images, {} validation images and {} testing images.".format(
        len(X_train), len(X_val), len(X_test)))
    print("Number of empty:humans:animals in training, validation and testing sets respectively is: {}:{}:{}; {}:{}:{}; {}:{}:{}\n".format(
        y_train.count("0"), y_train.count("1"), y_train.count("2"),
        y_val.count("0"), y_val.count("1"), y_val.count("2"),
        y_test.count("0"), y_test.count("1"), y_test.count("2")))
    if not args.skip_test:
        print('Testing will be conducted\n')
    else:
        print('Testing will NOT be conducted\n')

    # Extract hyperparameters if further training from a pre-trained model 
    if args.model_path is not None:
        hp_path = os.path.join(os.path.dirname(args.model_path), 'hyperparameter_records.csv')
        hp = pd.read_csv(hp_path)
        args.archi = hp.loc[hp['Hyperparameters'] == 'Architecture', 'Values'].item()
        args.weight_decay = float(hp.loc[hp['Hyperparameters'] == 'WeightDecay', 'Values'].item())
        args.dropout = float(hp.loc[hp['Hyperparameters'] == 'Dropout', 'Values'].item())

    # Output hyperparameters for recording purposes
    hp_names = (
        "SkipTest", "ModelPath", "LearningRate", "BetaDist_alpha", "BetaDist_beta", "Eps",
        "WeightDecay", "Epochs", "StepSize", "Gamma", "BatchSize", "ImgSize",
        "CropSize", "Architecture", "NumClasses", "TrainOnlyClassifier", "Dropout",
        "NoPretraining", "NumTrainImages", "NumValImages", "NumTestImages")
    hp_values = (
        args.skip_test, args.model_path, args.lr, args.betadist_alpha, args.betadist_beta,
        args.eps, args.weight_decay, args.epochs, args.step_size, args.gamma,
        args.batch_size, args.img_size, args.crop_size, args.archi,
        args.num_classes, args.train_only_classifier, args.dropout,
        args.no_pretraining, len(X_train), len(X_val), len(X_test))

    hp_records = pd.DataFrame(
        {'Hyperparameters': hp_names, 'Values': hp_values})
    hp_records.to_csv(index=False, path_or_buf=os.path.join(
        args.path_to_save_results, 'hyperparameter_records.csv'))

    print(hp_records)

    # Build model
    model, parameters = get_model(
        args.archi, args.num_classes, not args.train_only_classifier,
        not args.no_pretraining, args.dropout
    )

    # Prepare for training
    if not args.use_cpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if torch.backends.cudnn.is_available():
        print("\nUsing {} with cuDNN version {} for training with {} architecture.".format(
            device, torch.backends.cudnn.version(), args.archi))
    else:
        print("\nUsing {} WITHOUT cuDNN for training with {} architecture.".format(
            device, args.archi))

    betas = (args.betadist_alpha, args.betadist_beta)
    optimizer = optim.Adam(
        parameters,
        lr=args.lr,
        betas=betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    loss_func = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )
    if args.model_path is not None:
        load_checkpoint(args.model_path, model, optimizer, scheduler)

    # Train and validate
    weights, train_loss, train_acc, val_loss, val_acc, train_val_results = train_validate(
        args.epochs, model, optimizer, scheduler, loss_func,
        train_dl, val_dl, device, args.archi,
        args.path_to_save_results
    )

    results_dir = os.path.join(ROOT_DIR, 'results')
    write_to_csv(train_loss, os.path.join(results_dir, 'train_loss.csv'))
    write_to_csv(train_acc, os.path.join(results_dir, 'train_acc.csv'))
    write_to_csv(val_loss, os.path.join(results_dir, 'val_loss.csv'))
    write_to_csv(val_acc, os.path.join(results_dir, 'val_acc.csv'))
    train_val_results.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.path_to_save_results,
            'train_val_results.csv')
    )

    # Test
    if args.skip_test:
        print("\nTesting will not be conducted. Exiting now.")
        sys.exit()  # requires exit code 0 for clearml to detect successful termination

    print("Training and validation complete. Starting testing now.")
    model.load_state_dict(weights)
    test_acc, test_loss, probabilities = evaluate_model(
        model, test_dl, loss_func, device, 'Testing')
    print("Test acc: {}, Test loss: {}".format(test_acc, test_loss))

    # Saving results and probabilities
    probabilities = probabilities.T.tolist()
    test_probs_df = pd.DataFrame({
        'file_name': X_test,
        'prob_empty': probabilities[0],
        'prob_human': probabilities[1],
        'prob_animal': probabilities[2]}
    )
    test_probs_df.to_csv(index=False, path_or_buf=os.path.join(
        args.path_to_save_results, 'test_probabilities.csv'))

    test_results_df = pd.DataFrame({'Acc': [test_acc], 'Loss': [test_loss]})
    test_results_df.to_csv(index=False, path_or_buf=os.path.join(
        args.path_to_save_results, 'test_results.csv'))
