import csv
import torch
from dataset import get_dataloader
from models import get_model
from shared_funcs import read_csv
from torch import nn


def get_probabilities(model, dl, device):
    model.eval()
    with torch.no_grad():
        probabilities = []
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            softmax = nn.Softmax(1)
            probabilities.append(softmax(logits))
    probabilities = torch.cat(probabilities, 0)
    return probabilities.tolist()


if __name__ == '__main__':
    batch_size = 32
    img_size = 256
    crop_size = 224  # smallest is 224

    archi = 'resnet50'
    num_classes = 3
    use_gpu = False

    image_dir = './data/images'
    model_path = './models/model.pth'
    probabilities_path = './test_probabilities.csv'

    # Get data
    X_test = read_csv('X_test.csv')
    y_test = read_csv('y_test.csv')
    test_dl = get_dataloader(
        X_test, y_test, batch_size, image_dir, img_size, crop_size, False
    )

    # Prepare model
    model, parameters = get_model(
        archi, num_classes, False, False)
    model.load_state_dict(torch.load(model_path))

    # Get probabilities
    if use_gpu:
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    probabilities = get_probabilities(model, test_dl, device)

    with open(probabilities_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in probabilities:
            writer.writerow(row)
