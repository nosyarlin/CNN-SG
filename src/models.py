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
        return _get_mobilenet(
            num_classes, train_all_weights, pretrained, dropout
        )
    else:
        sys.exit("\nError: Please enter a valid architecture")


def _get_mobilenet(
        num_classes: int, train_all_weights: bool,
        pretrained: bool, dropout: float):
    
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.last_channel, num_classes),
    )
    if train_all_weights:
        parameters = model.parameters()
    else:
        parameters = model.classifier.parameters()
    
    return model, parameters


def _get_inception(
        num_classes: int, train_all_weights: bool,
        pretrained: bool, dropout: float):

    if pretrained:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    else:
        model = models.inception_v3(weights=None)

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
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
    elif name == 'resnet101':
        if pretrained:
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            model = models.resnet101(weights=None)
    elif name == 'resnet152':
        if pretrained:
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            model = models.resnet152(weights=None)
    elif name == 'wide_resnet50':
        if pretrained:
            model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            model = models.wide_resnet50_2(weights=None)
    else:
        sys.exit("\nError: Please input a valid resnet model.")

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
