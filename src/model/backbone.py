import logging

from torch import nn
from torchvision.models import resnet50, resnet18

from src.configuration import Config


def get_backbone(config: Config):
    backbone_name = config.model.backbone.name
    logging.info('Initialising backbone: %s', backbone_name)

    if backbone_name == 'resnet50':
        backbone = resnet50()
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet18':
        backbone = resnet18()
        backbone.fc = nn.Identity()
    else:
        raise NotImplementedError(f'Backbone {backbone_name} not supported.')

    return backbone
