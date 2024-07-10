import logging

from torch.optim import Adam, SGD

from src.configuration import Config


def get_optimiser(parameters, config: Config):
    optimiser_name = config.optim.optimiser

    logging.info('Initialising optimiser: %s', optimiser_name)

    if optimiser_name == 'adam':
        optimiser = Adam(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    elif optimiser_name == 'sgd':
        optimiser = SGD(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
            momentum=0.9
        )
    else:
        raise NotImplementedError(f'Optimiser {optimiser_name} not supported.')

    return optimiser
