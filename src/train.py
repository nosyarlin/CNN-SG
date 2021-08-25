from clearml import Task
from config import ROOT_DIR
from dataset import get_dataloader
from datetime import date
from hanging_threads import start_monitoring
from models import get_model
from shared_funcs import (
    read_csv, write_to_csv, train_validate, evaluate_model,
    save_checkpoint, load_checkpoint, check_img_size
)
from torch import nn, optim
import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch
import yappi


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Process Command-line Arguments')
    parser.add_argument(
        '--image_dir',
        default=default_image_dir,
        help='Path to the directory containing the images'
    )
    parser.add_argument(
        '--save_results_path',
        default=default_save_results_path,
        help='Path to the directory to save the model, hyperparameters and results'
    )
    parser.add_argument(
        '--model_path',
        default=default_model_path,
        help='Path to saved model weights. If this is set, we will use the provided weights as the starting point for training. If none, no further training will be conducted.'
    )
    parser.add_argument(
        '--xy_train', default=default_xy_train,
        help='Path to xy dataframe that will be used for training. Should contain two columns, "FileName" and "SpeciesCode".'
    )
    parser.add_argument(
        '--xy_val', default=default_xy_val,
        help='Path to xy dataframe that will be used for validation. Should contain two columns, "FileName" and "SpeciesCode".'
    )
    parser.add_argument(
        '--xy_test', default=default_xy_test,
        help='Path to xy dataframe that will be used for testing. Should contain two columns, "FileName" and "SpeciesCode".'
    )
    parser.add_argument(
        '--skip_test', action='store_true',
        help='Set if testing should be skipped'
    )
    parser.add_argument(
        '--archi', default=default_archi,
        help='Architecture of the model to be trained. Either inception, resnet50, resnet101, resnet152, wide_resnet50, or mobilenet'
    )
    parser.add_argument(
        '--no_pretraining', action='store_true',
        help='Set if you want the model to be trained from scratch'
    )
    parser.add_argument(
        '--train_only_classifier', action='store_true',
        help='Set if we train classification layer only'
    )
    parser.add_argument(
        '--num_workers', default=default_num_workers, type=int,
        help='Number of threads to be set for the GPU'
    )
    parser.add_argument(
        '--use_cpu', action='store_true',
        help='Using CPU for processing'
    )
    parser.add_argument(
        '--num_classes', default='3', type=int,
        action='store', help='Number of classes to be trained'
    )
    parser.add_argument(
        '--dropout', default=default_dropout, type=float,
        action='store', help='Dropout probablity'
    )
    parser.add_argument(
        '--lr', default='0.0005', type=float, help='The learning rate'
    )
    parser.add_argument(
        '--betadist_alpha', default=0.9, type=float,
        help='The alpha value controlling the shape of the beta distribution for the Adam optimiser'
    )
    parser.add_argument(
        '--betadist_beta', default=0.99, type=float,
        help='The beta value controlling the shape of the beta distribution for the Adam optimiser'
    )
    parser.add_argument(
        '--eps', default='1e-8', type=float,
        help='Epsilon value for Adam optimiser'
    )
    parser.add_argument(
        '--weight_decay', default=default_weight_decay, type=float,
        help='Weight decay for Adam optimiser'
    )
    parser.add_argument(
        '--epochs', default=default_epochs, type=int,
        help='Number of epochs to be run for training'
    )
    parser.add_argument(
        '--step_size', default='5', type=int,
        help='Step size'
    )
    parser.add_argument(
        '--gamma', default='0.1', type=float,
        help='Gamma value for optimiser'
    )
    parser.add_argument(
        '--batch_size', default='32', type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--img_size', default='360', type=int,
        help='Image size for each image'
    )
    parser.add_argument(
        '--crop_size', default='299', type=int,
        help='Crop size for each image. Inception v3 expects 299'
    )

    return parser


if __name__ == '__main__':
    # Monitoring for hangs and profiler
    #monitoring_thread = start_monitoring(seconds_frozen=60)

    # Connecting to the clearml dashboard
    task = Task.init(project_name="Nosyarlin", task_name="Train_" + date.today().strftime('%Y-%m-%d'),
                     task_type=Task.TaskTypes.training, reuse_last_task_id=False)

    # Set hyperparameters
    default_image_dir = 'C:/_for-temp-data-that-need-SSD-speed/'
    default_save_results_path = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Big4/Mo_0.7_09/trained_model_2' 
    default_model_path = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Big4/Mo_0.7_09/trained_model/archi_mobilenet_train_acc_0.899_val_acc_0.93_epoch_12.pth'

    default_xy_train = os.path.join(ROOT_DIR, 'data', 'splits', 'big4_20210810_train_sheet_resized.csv')
    default_xy_val = os.path.join(ROOT_DIR, 'data', 'splits', 'big4_20210810_val_sheet_resized.csv')
    default_xy_test = os.path.join(ROOT_DIR, 'data', 'splits', 'big4_20210810_test_sheet_resized.csv')

    default_archi = 'mobilenet' #Either inception, resnet50, resnet101, resnet152, wide_resnet50, or mobilenet
    default_dropout = '0.0001'
    default_weight_decay = '1e-8'
    default_epochs = '2' 
    default_num_workers='5'

    parser = get_arg_parser()
    args = parser.parse_args()

    # Check that paths to save results and models exist
    if os.path.exists(args.save_results_path) and len(os.listdir(args.save_results_path)) == 0:
        print("\nSaving results in " + args.save_results_path)
    else:
        sys.exit(
            "\nError: File path to save results do not exist, or directory is not empty")

    # Read data
    xy_train = pd.read_csv(args.xy_train)
    xy_val = pd.read_csv(args.xy_val)
    xy_test = pd.read_csv(args.xy_test)

    # Check the image size for the first image
    check_img_size(xy_train.FileName, "training set", args.img_size)
    check_img_size(xy_val.FileName, "validation set", args.img_size)
    check_img_size(xy_test.FileName, "testing set", args.img_size)

    train_dl = get_dataloader(
        xy_train.FileName, xy_train.SpeciesCode, args.batch_size, args.image_dir,
        args.crop_size, True, args.num_workers
    )
    val_dl = get_dataloader(
        xy_val.FileName, xy_val.SpeciesCode, args.batch_size, args.image_dir,
        args.crop_size, False, args.num_workers
    )
    test_dl = get_dataloader(
        xy_test.FileName, xy_test.SpeciesCode, args.batch_size, args.image_dir,
        args.crop_size, False, args.num_workers
    )

    print("\nDataset to be used includes {} training images, {} validation images and {} testing images.".format(
        len(xy_train.FileName), len(xy_val.FileName), len(xy_test.FileName)))
    print("Number of empty:humans:animals in training, validation and testing sets respectively is: {}:{}:{}; {}:{}:{}; {}:{}:{}\n".format(
        len(xy_train[xy_train.SpeciesCode == 0]), len(xy_train[xy_train.SpeciesCode == 1]), len(xy_train[xy_train.SpeciesCode == 2]),
        len(xy_val[xy_val.SpeciesCode == 0]), len(xy_val[xy_val.SpeciesCode == 1]), len(xy_val[xy_val.SpeciesCode == 2]),
        len(xy_test[xy_test.SpeciesCode == 0]), len(xy_test[xy_test.SpeciesCode == 1]), len(xy_test[xy_test.SpeciesCode == 2])))

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
        "SkipTest", "NumWorkers", "ModelPath", "LearningRate", "BetaDist_alpha", "BetaDist_beta", "Eps",
        "WeightDecay", "Epochs", "StepSize", "Gamma", "BatchSize", "ImgSize",
        "CropSize", "Architecture", "NumClasses", "TrainOnlyClassifier", "Dropout",
        "NoPretraining", "NumTrainImages", "NumValImages", "NumTestImages")
    hp_values = (
        args.skip_test, args.num_workers, args.model_path, args.lr, args.betadist_alpha, args.betadist_beta,
        args.eps, args.weight_decay, args.epochs, args.step_size, args.gamma,
        args.batch_size, args.img_size, args.crop_size, args.archi,
        args.num_classes, args.train_only_classifier, args.dropout,
        args.no_pretraining, len(xy_train.FileName), len(xy_val.FileName), len(xy_test.FileName))

    hp_records = pd.DataFrame(
        {'Hyperparameters': hp_names, 'Values': hp_values})
    hp_records.to_csv(index=False, path_or_buf=os.path.join(
        args.save_results_path, 'hyperparameter_records.csv'))

    print(hp_records.to_string())

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
    
    #Profiler
    yappi.set_clock_type("WALL")
    yappi.start(profile_threads = True)

    # Train and validate
    weights, train_loss, train_acc, val_loss, val_acc, train_val_results = train_validate(
        args.epochs, model, optimizer, scheduler, loss_func,
        train_dl, val_dl, device, args.archi,
        args.save_results_path
    )

    results_dir = os.path.join(ROOT_DIR, 'results')
    write_to_csv(train_loss, os.path.join(results_dir, 'train_loss.csv'))
    write_to_csv(train_acc, os.path.join(results_dir, 'train_acc.csv'))
    write_to_csv(val_loss, os.path.join(results_dir, 'val_loss.csv'))
    write_to_csv(val_acc, os.path.join(results_dir, 'val_acc.csv'))
    train_val_results.to_csv(
        index=False,
        path_or_buf=os.path.join(
            args.save_results_path,
            'train_val_results.csv')
    )

    #Profiler
    yappi.stop()
    stats = yappi.get_func_stats().sort(sort_type = "ttot", sort_order = "desc")
    stats_file = os.path.join(args.save_results_path, 'yappi_function_stats.prof')
    stats.save(stats_file, "pstat")
    #stats.print_all()

    #threads = yappi.get_thread_stats()
    #threads.print_all()

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
        'file_name': xy_test.FileName,
        'prob_empty': probabilities[0],
        'prob_human': probabilities[1],
        'prob_animal': probabilities[2]}
    )
    test_probs_df.to_csv(index=False, path_or_buf=os.path.join(
        args.save_results_path, 'test_probabilities.csv'))

    test_results_df = pd.DataFrame({'Acc': [test_acc], 'Loss': [test_loss]})
    test_results_df.to_csv(index=False, path_or_buf=os.path.join(
        args.save_results_path, 'test_results.csv'))