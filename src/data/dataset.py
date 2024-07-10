import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.configuration import Config
from src.data.augmentations import SimCLRTransformNoRandomResizedCrop, DeepSetTransform


class SimCLRNLSTPerScanAugDataset(Dataset):
    """
    Normal SimCLR will generate a positive pair of views by generating 2
    views from the same image.

    Here we allow a positive pair to come from any 2 images in within the same scan.
    """
    def __init__(self, paths, scans, transform, config: Config):
        self.paths = paths
        self.scans = scans
        self.transform = transform
        self.config = config
        self.min_required_samples = config.data.min_required_samples

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        chosen_scan = np.random.choice(self.scans)
        paths_for_scan = glob(os.path.join(chosen_scan, '*.png'))
        if len(paths_for_scan) < self.min_required_samples:
            # Per-image view generation
            path = np.random.choice(paths_for_scan)
            return self.transform(Image.open(str(path)).convert('RGB'))
        else:
            # Per-scan view generation
            paths_for_scan = sorted(paths_for_scan, key=lambda x: int(os.path.splitext(x)[0].split('-')[-1]))

            width = int(np.ceil(self.config.data.sample_width * len(paths_for_scan)))
            i = random.randint(0, len(paths_for_scan) - width - 1)
            j = i + width

            chosen_paths = [paths_for_scan[i], paths_for_scan[j]]

            image1_from_scan = Image.open(str(chosen_paths[0])).convert('RGB')
            image2_from_scan = Image.open(str(chosen_paths[1])).convert('RGB')

            return self.transform(image1_from_scan)[0], self.transform(image2_from_scan)[0]
        

class SimCLRNLSTDeepSetDataset(Dataset):
    def __init__(self, paths, scans, config: Config):
        self.paths = paths
        self.scans = scans
        self.config = config

        self.simclr_tfm_no_rrc = SimCLRTransformNoRandomResizedCrop(config)
        self.deepset_tfm = DeepSetTransform(
            config,
            normalise=False,
            random_resized_crop=True,
            horizontal_flip=False,
            colour_jitter=False,
            grayscale=False,
            gaussian_blur=False
        )

    def __len__(self):
        return len(self.paths)

    def generate_image_set(self, paths_for_scan):
        width = int(np.ceil(self.config.data.sample_width * len(paths_for_scan)))

        if (len(paths_for_scan) - width - 1) < 0:
            i = 0
            j = 0
        else:
            i = random.randint(0, len(paths_for_scan) - width - 1)
            j = i + width

        idx = np.linspace(i, j, self.config.data.set_size).astype(int)
        images_in_set = paths_for_scan[idx]

        image_set = [Image.open(e).convert('RGB') for e in images_in_set]
        image_set = self.deepset_tfm(image_set)

        image_set = torch.cat([
            self.simclr_tfm_no_rrc(image).unsqueeze(0)
            for image in image_set
        ], dim=0)

        return image_set

    def __getitem__(self, item):
        chosen_scan = np.random.choice(self.scans)
        paths_for_scan = glob(os.path.join(chosen_scan, '*.png'))
        paths_for_scan = np.array(sorted(paths_for_scan, key=lambda x: int(os.path.splitext(x)[0].split('-')[-1])))

        image_set1 = self.generate_image_set(paths_for_scan)
        image_set2 = self.generate_image_set(paths_for_scan)

        return image_set1, image_set2
