import logging

from src.configuration import Config
from src.model.simclr import SimCLR, DeepSetSimCLR


def get_model(config: Config):
    logging.info('Initialising model: %s', config.model.type)
    if config.model.type == 'simclr':
        model = SimCLR(config)
    elif config.model.type == 'deepset':
        model = DeepSetSimCLR(config)
    else:
        raise NotImplementedError(f'Model not supported: {config.model.type}')

    return model
