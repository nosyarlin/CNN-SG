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
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if train_all_weights:
            parameters = model.parameters()
        else:
            parameters = model.fc.parameters()

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
