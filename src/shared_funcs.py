from torch import nn
from torch.cuda.amp import autocast, GradScaler
import csv
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import sys


def write_to_csv(obj, fname):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(obj)


def read_csv(fname):
    out = []
    with open(fname) as f:
        reader = csv.reader(f)
        for line in reader:
            out.append(line)
    return [item for sublist in out for item in sublist]


def evaluate_model(model, dl, loss_func, device, logger_title):
    model.eval()
    with torch.no_grad():
        losses = []
        total_count = 0
        total_correct = 0

        probabilities = []

        j = 0  # Starting the iteration count

        for X, y in tqdm(dl):
            j += 1  # Adding one iteration count per batch

            X, y = X.to(device), y.to(device)

            X = X.view(-1, X.size(2), X.size(3), X.size(4))
            logits = model(X)
            logits = logits.view(-1, 5, logits.size(1)).mean(dim=1)
            loss = loss_func(logits, y)
            losses.append(loss.item())

            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)

            softmax = nn.Softmax(1)
            probabilities.append(softmax(logits))

        probabilities = torch.cat(probabilities, 0).cpu().numpy()

    return total_correct / total_count, np.mean(losses), probabilities


def train_model(model, dl, loss_func, optimizer, device, archi, epoch):
    model.train()
    train_loss = []
    total_count = 0
    total_correct = 0
    scaler = GradScaler()

    j = 0  # Starting the iteration count

    for X, y in tqdm(dl):
        j += 1  # Adding one iteration count per batch

        model.zero_grad()
        X, y = X.to(device), y.to(device)

        # auto cast to fp16 or fp32
        with autocast():
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

        # Scales loss, perform backprop and unscale
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
        acc_val, loss_val, _ = evaluate_model(
            model, val_dl, loss_func, device, 'Validation (Most recent epoch)')
        val_acc.append(acc_val)
        val_loss.append(loss_val)

        print("Epoch: {} of {} completed.".format(epoch + 1, epochs))
        print("Validation acc: {}, Validation loss: {}\n"
              .format(acc_val, loss_val))

        # Save model if improved
        if not best_weights or val_acc[-1] > best_val_acc:
            best_weights = model.state_dict()
            best_val_acc = val_acc[-1]
            save_checkpoint(
                os.path.join(
                    path_to_save_results,
                    'archi_{}_train_acc_{}_val_acc_{}_epoch_{}.pth'.format(
                        archi,
                        np.round(train_acc[-1], 3),
                        np.round(val_acc[-1], 3),
                        epoch + 1
                    )
                ),
                model, optimizer, scheduler)
        else:
            print(
                "Model trained in epoch {} has not improved, \
                and will not be saved.\n".format(
                    epoch + 1
                )
            )

    # Saving results into a dataframe
    train_val_results = pd.DataFrame({
        'Epoch': list(range(1, epochs + 1)),
        'TrainAcc': train_acc,
        'TrainLoss': train_loss,
        'ValAcc': val_acc,
        'ValLoss': val_loss
    })

    return best_weights, train_loss, train_acc,\
        val_loss, val_acc, train_val_results


def save_checkpoint(path, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])


def check_img_size(data, set_name, img_size):
    image = Image.open(data[1])
    w, h = image.size

    if not(w == img_size or h == img_size):
        sys.exit(
            "\nError: The first image in {} is not \
            the correct size of {}".format(
                set_name,
                img_size
            )
        )
