from torchvision import models
from torch import nn


def get_model(
        name: str, num_classes: int, train_all_weights: bool,
        pretrained: bool):
    """
    Returns CNN model and trainable parameters

    Parameters:
        name: Name of architecture. mobilenet/resnet50
    """
    if name == 'resnet50':
        return _get_resnet(name, num_classes, train_all_weights, pretrained)
    elif name == 'wide_resnet50':
        return _get_resnet(name, num_classes, train_all_weights, pretrained)
    else:
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        if train_all_weights:
            parameters = model.parameters()
        else:
            parameters = model.classifier.parameters()

    return model, parameters


def _get_resnet(name: str, num_classes: int, train_all_weights: bool,
                pretrained: bool):
    # Get model
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        model = models.wide_resnet50_2(pretrained=pretrained)

    # Reset classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Get params
    if train_all_weights:
        parameters = model.parameters()
    else:
        parameters = model.fc.parameters()

    return model, parameters
