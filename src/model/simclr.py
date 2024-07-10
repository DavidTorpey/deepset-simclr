import logging

from lightly.models.modules import SimCLRProjectionHead
from torch import nn

from src.configuration import Config
from src.model.backbone import get_backbone


class SimCLR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)

        self.projection_head = SimCLRProjectionHead(
            list(self.backbone.parameters())[-1].shape[0],
            config.model.hidden_dim,
            config.model.proj_dim
        )

    def forward(self, x):
        x = self.backbone(x)
        z = self.projection_head(x)
        return z


class DeepSet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.deepset = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, instance_representations):
        aggregated_features = instance_representations.sum(1)

        return self.deepset(aggregated_features)


class DeepSetSimCLR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)

        encoder_dim = list(self.backbone.parameters())[-1].shape[0]
        self.projection_head = SimCLRProjectionHead(
            encoder_dim,
            config.model.hidden_dim,
            config.model.proj_dim
        )

        if config.model.dedicated_deepset_mlp:
            logging.info('Using a dedicated MLP for the DeepSet...')
            self.deepset = DeepSet(encoder_dim)

        self.config = config

    def forward(self, x):
        #  x: [batch, num_crops, 3, H, W]
        a, b, c, d, e = x.shape

        #  features: [batch, num_crops, encoder_dim]
        features = self.backbone(x.view(a * b, c, d, e)).view(a, b, -1)

        #  set_representations: [batch, encoder_dim]
        if self.config.model.dedicated_deepset_mlp:
            set_representations = self.deepset(features)
        else:
            set_representations = features.sum(1)

        #  z: [batch, proj_dim]
        z = self.projection_head(set_representations)

        return z
