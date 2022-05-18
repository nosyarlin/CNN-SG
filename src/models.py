import sys
from torch import nn
from torchvision import models


def get_model(
        name: str, num_classes: int, train_all_weights: bool,
        pretrained: bool, dropout=0):
    """
    Returns CNN model and trainable parameters

    Parameters:
        name: Name of architecture. mobilenet/resnet50
    """
    if name == 'resnet50':
        return _get_resnet(
            name, num_classes, train_all_weights, pretrained, dropout
        )
    elif name == 'resnet101':
        return _get_resnet(
            name, num_classes, train_all_weights, pretrained, dropout
        )
    elif name == 'resnet152':
        return _get_resnet(
            name, num_classes, train_all_weights, pretrained, dropout
        )
    elif name == 'wide_resnet50':
        return _get_resnet(
            name, num_classes, train_all_weights, pretrained, dropout
        )
    elif name == 'inception':
        return _get_inception(
            num_classes, train_all_weights, pretrained, dropout
        )
    elif name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.last_channel, num_classes),
        )
        if train_all_weights:
            parameters = model.parameters()
        else:
            parameters = model.classifier.parameters()
    else:
        sys.exit("\nError: Please enter a valid architecture")

    return model, parameters


def _get_inception(
        num_classes: int, train_all_weights: bool,
        pretrained: bool, dropout: float):

    model = models.inception_v3(pretrained=pretrained)

    # Handle the auxilary net
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes)
    )

    # Handle the primary net
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes)
    )

    # Get params
    if train_all_weights:
        parameters = model.parameters()
    else:
        parameters = model.fc.parameters()

    return model, parameters


def _get_resnet(
        name: str, num_classes: int, train_all_weights: bool,
        pretrained: bool, dropout: float):

    # Get model
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        model = models.wide_resnet50_2(pretrained=pretrained)

    # Reset classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes)
    )

    # Get params
    if train_all_weights:
        parameters = model.parameters()
    else:
        parameters = model.fc.parameters()

    return model, parameters
