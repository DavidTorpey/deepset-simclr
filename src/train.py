import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import asdict

import torch
from torch.backends import cudnn

from src.configuration import Config, load_config
from src.constants import CONFIG_FILE_NAME, LOG_FILE_NAME
from src.data.data import get_loaders
from src.model.model import get_model
from src.optimisation import get_optimiser
from src.persistence import restore_from_checkpoint
from src.trainer import Trainer
from src.utl import mkdir, cosine_scheduler


def train(config: Config):
    cudnn.benchmark = True

    if torch.cuda.is_available() and config.optim.device == 'cpu':
        logging.warning('CUDA is available, but chosen to run on CPU.')

    train_loader, val_loader = get_loaders(config)

    model = get_model(config).to(config.optim.device)

    optimiser = get_optimiser(model.parameters(), config)

    lr_schedule = cosine_scheduler(
        config.optim.lr, 0, config.optim.epochs, len(train_loader), config.optim.warmup_epochs
    )

    to_restore = restore_from_checkpoint(
        config, model, optimiser
    )
    start_epoch = to_restore['epoch']

    trainer = Trainer(model, optimiser, lr_schedule, config)

    trainer.train(train_loader, val_loader, start_epoch)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config_path)

    mkdir(config.general.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.general.output_dir, LOG_FILE_NAME)),
            logging.StreamHandler()
        ]
    )

    with open(os.path.join(config.general.output_dir, CONFIG_FILE_NAME), 'w') as handle:
        json.dump(asdict(config), handle, indent=2)

    train(config)


if __name__ == '__main__':
    main()
