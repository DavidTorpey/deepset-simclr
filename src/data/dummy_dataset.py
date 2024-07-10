import logging
import os

from src.configuration import Config
from src.data.dataset import SimCLRNLSTPerScanAugDataset, SimCLRNLSTDeepSetDataset


def get_dummy_dataset(config: Config, train_transform, val_transform):
    """
    We define fake, dummy paths here for both the scans and the images in the scans.
    When you use this repo on your own dataset, you should replace these with real path
    to actual scan folders and images.
    """

    dummy_train_scans = [
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan1'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan2'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan3'),
    ]

    dummy_train_paths = [
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan1', 'image1.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan1', 'image2.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan2', 'image1.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan3', 'image1.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan3', 'image2.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan3', 'image3.jpg'),
    ]

    dummy_val_scans = [
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan4'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan5'),
    ]

    dummy_val_paths = [
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan4', 'image1.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan4', 'image2.jpg'),
        os.path.join(config.data.dataset_root, 'fake', 'path', 'to', 'scan5', 'image1.jpg'),
    ]

    if config.data.dataset_type == 'per_scan':
        logging.info('Initialising per-scan dataset')
        train_dataset = SimCLRNLSTPerScanAugDataset(
            dummy_train_paths, dummy_train_scans, train_transform, config
        )

        val_dataset = SimCLRNLSTPerScanAugDataset(
            dummy_val_paths, dummy_val_scans, val_transform, config
        )
    elif config.data.dataset_type == 'deepset':
        logging.info('Initialising Deep Set dataset')
        train_dataset = SimCLRNLSTDeepSetDataset(
            dummy_train_paths, dummy_train_scans, config
        )

        val_dataset = SimCLRNLSTDeepSetDataset(
            dummy_val_paths, dummy_val_scans, config
        )
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    logging.info(
        'Initialised dummy dataset: Train=%s, Val=%s',
        len(train_dataset), len(val_dataset)
    )

    return train_dataset, val_dataset
