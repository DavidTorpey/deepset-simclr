from dataclasses import dataclass, field

import yaml
from dacite import from_dict


@dataclass
class General:
    output_dir: str
    log_to_wandb: bool = False
    checkpoint_freq: int = 10


@dataclass
class Data:
    dataset: str
    dataset_root: str
    train_aug: str = 'simclr'
    val_aug: str = 'simclr'
    image_size: int = 224

    # For NLST only
    dataset_type: str = 'per_scan'  # ["per_scan", "deepset"]
    sample_width: float = 0.1
    # For "per_scan"
    min_required_samples: int = 5
    # For "deepset"
    set_size: int = 5


@dataclass
class Optimisation:
    device: str = 'cpu'
    lr: float = 3e-4
    weight_decay: float = 0.0
    workers: int = 4
    batch_size: int = 4
    epochs: int = 1
    warmup_epochs: int = 1
    optimiser: str = 'adam'


@dataclass
class Backbone:
    name: str = 'resnet50'


@dataclass
class Model:
    type: str = 'simclr'  # ["simclr", "deepset"]
    backbone: Backbone = field(default_factory=lambda: Backbone())
    hidden_dim: int = 2048
    proj_dim: int = 128
    dedicated_deepset_mlp: bool = False


@dataclass
class Config:
    general: General
    data: Data
    model: Model = field(default_factory=lambda: Model())
    optim: Optimisation = field(default_factory=lambda: Optimisation())


def load_config(config_path: str) -> Config:
    with open(config_path) as file:
        data = yaml.safe_load(file)
    return from_dict(Config, data)
