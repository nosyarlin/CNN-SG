import csv
import numpy as np
import torch
from dataset import get_dataloader
from models import get_model
from shared_funcs import read_csv, write_to_csv
from torch import nn, optim
import pandas as pd
import os.path
from os import path
import sys


def evaluate_model(model, dl, loss_func, device):
    model.eval()
    with torch.no_grad():
        losses = []
        total_count = 0
        total_correct = 0

        probabilities = []

        for X, y in dl:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_func(logits, y)
            losses.append(loss.item())

            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)

            softmax = nn.Softmax(1)
            probabilities.append(softmax(logits))
    probabilities = torch.cat(probabilities, 0)

    return total_correct / total_count, np.mean(losses), probabilities.tolist()


def train_model(model, dl, loss_func, optimizer, device, archi):
    model.train()
    train_loss = []
    total_count = 0
    total_correct = 0
    for X, y in dl:
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

    return total_correct / total_count, np.mean(train_loss)


def train_validate_test(
        epochs, model, optimizer, scheduler, loss_func, train_dl,
        val_dl, test_dl, device, archi, path_to_save_results):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_acc_data = []
    test_loss_data = []
    best_val_acc = -1
    best_weights = None

    print("Starting the training now.")
    print("Training for {} epochs".format(epochs))

    for epoch in range(epochs):
        # Train
        acc, loss = train_model(
            model, train_dl, loss_func, optimizer, device, archi)
        train_loss.append(loss)
        train_acc.append(acc)
        scheduler.step()

        # Validate
        acc, loss, probabilities = evaluate_model(model, val_dl, loss_func, device)
        val_acc.append(acc)
        val_loss.append(loss)

        print("Epoch: {} of {}".format(epoch + 1, epochs))
        print("Validation acc: {}, Validation loss: {}"
              .format(acc, loss))

        # Save model if improved
        if not best_weights or val_acc[-1] > best_val_acc:
            best_weights = model.state_dict()
            torch.save(best_weights, path_to_save_results+'model.pth')
            best_val_acc = val_acc[-1]
            print("\n")
        else:
            print("Model has not improved, and will not be saved.\n")

    # Test
    print("Training and validation complete. Starting testing now.")
    model.load_state_dict(best_weights)
    test_acc, test_loss, probabilities = evaluate_model(model, test_dl, loss_func, device)
    print("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
    test_acc_data.append(test_acc)
    test_loss_data.append(test_loss)

    # Saving them into datasets
    test_results = pd.DataFrame({'Acc': test_acc_data, 'Loss': test_loss_data})
    train_val_results = pd.DataFrame({'Epoch': list(range(
        1, epochs + 1)), 'TrainAcc': train_acc, 'TrainLoss': train_loss, 'ValAcc': val_acc, 'ValLoss': val_loss})

    return best_weights, train_loss, train_acc, val_loss, val_acc, test_results, train_val_results, probabilities
    
    
if __name__ == '__main__':
    # Set hyperparameters
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0
    epochs = 1
    step_size = 5
    gamma = 0.1
    batch_size = 32
    img_size = 360 #inception 360
    crop_size = 299  # Inception v3 expects 299

    archi = 'resnet50' #either inception, resnet50, wide_resnet50, or mobilenet
    num_classes = 3
    use_gpu = True
    use_data_augmentation = True
    train_all_weights = False
    pretrained = True

    image_dir = 'C:/_for-temp-data-that-need-SSD-speed/ProjectMast_FYP_Media'
    path_to_save_results = 'E:/JoejynDocuments/CNN_Animal_ID/Nosyarlin/SBWR_BTNR_CCNR/Results/Test/' #must end with /

    # Check that paths to save results and models exist 
    if path.exists(path_to_save_results):
        print("\nSaving results in "+path_to_save_results)
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
        X_train, y_train, batch_size, image_dir, img_size, crop_size, use_data_augmentation
    )
    val_dl = get_dataloader(
        X_val, y_val, batch_size, image_dir, img_size, crop_size, False
    )
    test_dl = get_dataloader(
        X_test, y_test, batch_size, image_dir, img_size, crop_size, False
    )

    print("Dataset to be used includes {} training images, {} validation images and {} testing images.".format(len(X_train), len(X_val), len(X_test)))
    print("Number of empty:humans:animals in training, validation and testing sets respectively is: {}:{}:{}; {}:{}:{}; {}:{}:{}\n".format(
        y_train.count("0"),y_train.count("1"),y_train.count("2"),
        y_val.count("0"),y_val.count("1"),y_val.count("2"),
        y_test.count("0"),y_test.count("1"),y_test.count("2")))

    # Build model
    model, parameters = get_model(
        archi, num_classes, train_all_weights, pretrained)

    # Prepare for training
    if use_gpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using {} for training with {} architecture.".format(device, archi))

    optimizer = optim.Adam(
        parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    loss_func = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # Train, validate, test
    weights, train_loss, train_acc, val_loss, val_acc, test_results, train_val_results, probabilities = train_validate_test(
        epochs, model, optimizer, scheduler, loss_func,
        train_dl, val_dl, test_dl, device, archi, 
        path_to_save_results
    )

    # Save results
    write_to_csv(train_loss, 'train_loss.csv')
    write_to_csv(train_acc, 'train_acc.csv')
    write_to_csv(val_loss, 'val_loss.csv')
    write_to_csv(val_acc, 'val_acc.csv')

    train_val_results.to_csv(
        index=False, path_or_buf=path_to_save_results+'train_val_results.csv')
    test_results.to_csv(index=False, path_or_buf=path_to_save_results+'test_results.csv')

    # Output probablities for test data
    with open(path_to_save_results+'test_probabilities.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["prob_empty", "prob_human", "prob_animal"])
        for row in probabilities:
            writer.writerow(row)

    # Output hyperparameters for recording purposes
    hp_names = (
        "LearningRate", "Betas", "Eps", "WeightDecay", "Epochs", "StepSize", "Gamma", "BatchSize", "ImgSize", "CropSize",
        "Architecture", "NumClasses", "UseDataAugmentation", "TrainAllWeights", "Pretrained",
        "NumTrainImages", "NumValImages", "NumTestImages")
    hp_values = (
        lr, betas, eps, weight_decay, epochs, step_size, gamma, batch_size, img_size, crop_size, 
        archi, num_classes, use_data_augmentation, train_all_weights, pretrained,
        len(X_train), len(X_val), len(X_test))

    hp_records = pd.DataFrame({'Hyperparameters': hp_names, 'Values': hp_values})
    hp_records.to_csv(index=False, path_or_buf=path_to_save_results+'hyperparameter_records.csv')