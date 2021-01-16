import sys
import os.path
from os import path
from datetime import date

import csv
import numpy as np
import torch
from torch import nn, optim
import pandas as pd
from clearml import Task, Logger
import argparse

from dataset import get_dataloader
from models import get_model
from shared_funcs import read_csv, write_to_csv


def evaluate_model(model, dl, loss_func, device, logger_title):
    model.eval()
    with torch.no_grad():
        losses = []
        total_count = 0
        total_correct = 0

        probabilities = []

        j = 0 #Starting the iteration count

        for X, y in dl:
            j += 1 #Adding one iteration count per batch

            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_func(logits, y)
            losses.append(loss.item())

            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)

            softmax = nn.Softmax(1)
            probabilities.append(softmax(logits))

            Logger.current_logger().report_scalar(
                logger_title, "loss", iteration= j, value= loss.item())
            Logger.current_logger().report_scalar(
                logger_title, "accuracy", iteration= j, value= (total_correct / total_count))

    probabilities = torch.cat(probabilities, 0)

    return total_correct / total_count, np.mean(losses), probabilities.tolist()


def train_model(model, dl, loss_func, optimizer, device, archi, epoch):
    model.train()
    train_loss = []
    total_count = 0
    total_correct = 0

    j = 0 #Starting the iteration count 

    for X, y in dl:
        j += 1 #Adding one iteration count per batch

        model.zero_grad()
        X, y = X.to(device), y.to(device)

        # Inception gives two outputs
        if archi == "inception":
            logits, aux_logits = model(X)
            loss1 = loss_func(logits, y)
            loss2 = loss_func(aux_logits, y)
            loss = loss1 + 0.4 * loss2
        else:
            logits = model(X)
            loss = loss_func(logits, y)
        train_loss.append(loss.item())

        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        total_count += y.size(0)

        loss.backward()
        optimizer.step()

        Logger.current_logger().report_scalar(
            "Training", "loss", iteration=((epoch+1) * j), value= loss.item())
        Logger.current_logger().report_scalar(
            "Training", "accuracy", iteration=((epoch+1) * j), value= (total_correct / total_count))

    return total_correct / total_count, np.mean(train_loss)


def train_validate(
        epochs, model, optimizer, scheduler, loss_func, train_dl,
        val_dl, device, archi, path_to_save_results):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = -1
    best_weights = None

    print("Starting the training now.")
    print("Training for {} epochs".format(epochs))

    for epoch in range(epochs):
        # Train
        acc_train, loss_train = train_model(
            model, train_dl, loss_func, optimizer, device, archi, epoch)
        train_acc.append(acc_train)
        train_loss.append(loss_train)       
        scheduler.step()

        # Validate
        acc_val, loss_val, probabilities = evaluate_model(model, val_dl, loss_func, device, 'Validation (Most recent epoch)')
        val_acc.append(acc_val)
        val_loss.append(loss_val)
        
        # Save model if improved
        if not best_weights or val_acc[-1] > best_val_acc:
            best_weights = model.state_dict()
            torch.save(best_weights, os.path.join(path_to_save_results, 'model.pth'))
            best_val_acc = val_acc[-1]
        else:
            print("Model has not improved, and will not be saved.\n")

        # Logging the results in clearml
        Logger.current_logger().report_scalar(
            "Training and Validation", "Train accuracy", iteration= epoch+1, value= acc_train)
        Logger.current_logger().report_scalar(
            "Training and Validation", "Train loss", iteration= epoch+1, value= loss_train)
        Logger.current_logger().report_scalar(
            "Training and Validation", "Val accuracy", iteration= epoch+1, value= acc_val)
        Logger.current_logger().report_scalar(
            "Training and Validation", "Val loss", iteration= epoch+1, value= loss_val)

        print("Epoch: {} of {}".format(epoch + 1, epochs))
        print("Validation acc: {}, Validation loss: {}"
              .format(acc_val, loss_val))

    # Saving them into datasets
    train_val_results = pd.DataFrame({'Epoch': list(range(
        1, epochs + 1)), 'TrainAcc': train_acc, 'TrainLoss': train_loss, 'ValAcc': val_acc, 'ValLoss': val_loss})

    return best_weights, train_loss, train_acc, val_loss, val_acc, train_val_results


if __name__ == '__main__':
    # Connecting to the clearml dashboard 
    task = Task.init(project_name="Nosyarlin", task_name= "Train_"+ date.today().strftime('%Y-%m-%d'),
        task_type= Task.TaskTypes.training)

    # Set hyperparameters
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')
    parser.add_argument('--image_dir', default= None, action= 'store', help= 'Path to the directory containing the images')
    parser.add_argument('--path_to_save_results', default= None, action= 'store', help= 'Path to the directory to save the model, hyperparameters and results')
    parser.add_argument('--run_test', default= True, type= bool, action= 'store', help= 'Determine if testing should be conducted')
    parser.add_argument('--archi', default= 'resnet50', action='store', help='Architecture of the model to be trained. Either inception, resnet50, wide_resnet50, or mobilenet')
    parser.add_argument('--pretrained', default= True, type= bool, action='store', help='Choose if the model to be trained should be a pretrained model from pytorch')
    parser.add_argument('--train_all_weights', default= True, type= bool, action='store', help='True: train all weights across all layers; False: train classification layer only')
    parser.add_argument('--use_data_augmentation', default= True, type= bool, action='store', help='Using data augmentation')
    parser.add_argument('--use_gpu', default= True, type= bool, action='store', help='Using GPU for processing')
    parser.add_argument('--num_classes', default= '3', type= int, action='store', help='Number of classes to be trained')
    parser.add_argument('--lr', default= '0.001', type= float, action='store', help='The learning rate')
    parser.add_argument('--betadist_alpha', default= 0.9, type= float, action='store', help='The alpha value controlling the shape of the beta distribution for the Adam optimiser')
    parser.add_argument('--betadist_beta', default= 0.99, type= float, action='store', help='The beta value controlling the shape of the beta distribution for the Adam optimiser')
    parser.add_argument('--eps', default= '1e-8', type= float, action='store', help='Epsilon value for Adam optimiser')
    parser.add_argument('--weight_decay', default= '0', type= float, action='store', help='Weight decay for Adam optimiser')
    parser.add_argument('--epochs', default= '25', type= int, action='store', help='Number of epochs to be run for training')
    parser.add_argument('--step_size', default= '5', type= int, action='store', help='Step size')
    parser.add_argument('--gamma', default= '0.1', type= float, action='store', help='Gamma value for optimiser')
    parser.add_argument('--batch_size', default= '32', type= int, action='store', help='Batch size for training')
    parser.add_argument('--img_size', default= '360', type= int, action='store', help='Image size for each image')
    parser.add_argument('--crop_size', default= '299', type= int, action='store', help='Crop size for each image. Inception v3 expects 299')

    args = parser.parse_args([
        '--image_dir', 'C:/_for-temp-data-that-need-SSD-speed/ProjectMast_FYP_Media',
        '--path_to_save_results', 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Test/',
        '--run_test', 'False',
        '--archi', 'mobilenet',
        '--epochs', '15',
        '--lr', '0.001',
        '--betadist_alpha', '0.9',
        '--betadist_beta', '0.99', 
        '--batch_size', '32', 
        '--weight_decay', '0'
        ])

    # Check that paths to save results and models exist 
    if path.exists(args.path_to_save_results):
        print("\nSaving results in " + args.path_to_save_results)
    else:
        sys.exit("\nError: File path to save results do not exist")

    # Read data
    X_train = read_csv('X_train.csv')
    y_train = read_csv('y_train.csv')
    X_val = read_csv('X_val.csv')
    y_val = read_csv('y_val.csv')
    X_test = read_csv('X_test.csv')
    y_test = read_csv('y_test.csv')
    train_dl = get_dataloader(
        X_train, y_train, args.batch_size, args.image_dir, args.img_size, args.crop_size, args.use_data_augmentation
    )
    val_dl = get_dataloader(
        X_val, y_val, args.batch_size, args.image_dir, args.img_size, args.crop_size, False
    )
    test_dl = get_dataloader(
        X_test, y_test, args.batch_size, args.image_dir, args.img_size, args.crop_size, False
    )

    print("Dataset to be used includes {} training images, {} validation images and {} testing images.".format(len(X_train), len(X_val), len(X_test)))
    print("Number of empty:humans:animals in training, validation and testing sets respectively is: {}:{}:{}; {}:{}:{}; {}:{}:{}\n".format(
        y_train.count("0"),y_train.count("1"),y_train.count("2"),
        y_val.count("0"),y_val.count("1"),y_val.count("2"),
        y_test.count("0"),y_test.count("1"),y_test.count("2")))
    if args.run_test:
        print('Testing will be conducted.')
    else:
        print('Testing will NOT be conducted')
        
    # Output hyperparameters for recording purposes
    hp_names = (
        "Run_test", "LearningRate", "BetaDist_alpha", "BetaDist_beta", "Eps", "WeightDecay", "Epochs", "StepSize", "Gamma", "BatchSize", "ImgSize", "CropSize",
        "Architecture", "NumClasses", "UseDataAugmentation", "TrainAllWeights", "Pretrained",
        "NumTrainImages", "NumValImages", "NumTestImages")
    hp_values = (
        args.run_test, args.lr, args.betadist_alpha, args.betadist_beta, args.eps, args.weight_decay, args.epochs, args.step_size, args.gamma, args.batch_size, args.img_size, args.crop_size, 
        args.archi, args.num_classes, args.use_data_augmentation, args.train_all_weights, args.pretrained,
        len(X_train), len(X_val), len(X_test))

    hp_records = pd.DataFrame({'Hyperparameters': hp_names, 'Values': hp_values})
    hp_records.to_csv(index=False, path_or_buf= os.path.join(args.path_to_save_results, 'hyperparameter_records.csv'))

    # Build model
    model, parameters = get_model(
        args.archi, args.num_classes, args.train_all_weights, args.pretrained)

    # Prepare for training
    if args.use_gpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using {} for training with {} architecture.".format(device, args.archi))

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

    # Train and validate
    weights, train_loss, train_acc, val_loss, val_acc, train_val_results= train_validate(
        args.epochs, model, optimizer, scheduler, loss_func,
        train_dl, val_dl, device, args.archi, 
        args.path_to_save_results
    )

    write_to_csv(train_loss, 'train_loss.csv')
    write_to_csv(train_acc, 'train_acc.csv')
    write_to_csv(val_loss, 'val_loss.csv')
    write_to_csv(val_acc, 'val_acc.csv')
    train_val_results.to_csv(
        index=False, path_or_buf = os.path.join(args.path_to_save_results, 'train_val_results.csv'))

    # Test
    if args.run_test:
        print("Training and validation complete. Starting testing now.")
        
        test_acc_data = []
        test_loss_data = []        
        model.load_state_dict(weights)
        test_acc, test_loss, probabilities = evaluate_model(model, test_dl, loss_func, device, 'Testing')
        print("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        test_acc_data.append(test_acc)
        test_loss_data.append(test_loss)

        # Saving results and probabilities into datasets
        test_results = pd.DataFrame({'Acc': test_acc_data, 'Loss': test_loss_data})
        test_results.to_csv(index=False, path_or_buf = os.path.join(args.path_to_save_results, 'test_results.csv'))
        
        with open(os.path.join(args.path_to_save_results, 'test_probabilities.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["prob_empty", "prob_human", "prob_animal"])
            for row in probabilities:
                writer.writerow(row)
    else:
        print("Testing will not be conducted")