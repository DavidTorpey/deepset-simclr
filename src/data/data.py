from torch.utils.data import DataLoader

from src.configuration import Config
from src.data.augmentations import get_transform
from src.data.dummy_dataset import get_dummy_dataset


def get_loaders(config: Config):
    dataset = config.data.dataset

    train_transform = get_transform(config.data.train_aug, config)
    val_transform = get_transform(config.data.val_aug, config)

    if dataset == 'dummy':
        train_dataset, val_dataset = get_dummy_dataset(config, train_transform, val_transform)
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported')

    train_loader = DataLoader(
        train_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=True, num_workers=config.optim.workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=False, num_workers=config.optim.workers, pin_memory=True
    )

    return train_loader, val_loader
