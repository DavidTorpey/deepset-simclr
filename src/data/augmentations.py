import logging
from typing import List

import numpy as np
from PIL import Image
from lightly.transforms import SimCLRTransform, GaussianBlur
import torchvision.transforms as T
import torchvision.transforms.functional as F

from src.configuration import Config
from src.constants import MEAN, STD


class DeepSetTransform:
    def __init__(
            self, config: Config,
            normalise=True,
            random_resized_crop=True,
            horizontal_flip=True,
            colour_jitter=True,
            grayscale=True,
            gaussian_blur=True,
    ):
        self.config = config

        self.final_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

        self.normalise = normalise
        self.random_resized_crop = random_resized_crop
        self.horizontal_flip = horizontal_flip
        self.colour_jitter = colour_jitter
        self.grayscale = grayscale
        self.gaussian_blur = gaussian_blur

    def perform_random_resized_crop(self, imgs):
        size = (self.config.data.image_size, self.config.data.image_size)

        rrc = T.RandomResizedCrop(size=self.config.data.image_size, scale=(0.08, 1.0))

        i, j, h, w = rrc.get_params(imgs[0], scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))

        return_imgs = []
        for img in imgs:
            img = F.resized_crop(img, i, j, h, w, size, F.InterpolationMode.BILINEAR, antialias="warn")
            return_imgs.append(img)

        return return_imgs

    def perform_horizon_flip(self, imgs):
        if 0.5 < np.random.uniform(0, 1):
            rhf = T.RandomHorizontalFlip(p=1.0)
            return_imgs = []
            for img in imgs:
                return_imgs.append(rhf(img))
            return return_imgs
        else:
            return imgs

    def perform_random_grayscale(self, imgs):
        if 0.2 < np.random.uniform(0, 1):
            rg = T.RandomGrayscale(p=1.0)
            return_imgs = []
            for img in imgs:
                return_imgs.append(rg(img))
            return return_imgs
        else:
            return imgs

    def perform_gaussian_blur(self, imgs):
        if 0.5 < np.random.uniform(0, 1):
            sigma = np.random.uniform(0.1, 2)
            gb = GaussianBlur(kernel_size=None, sigmas=(sigma, sigma), prob=1.0)
            return_imgs = []
            for img in imgs:
                return_imgs.append(gb(img))
            return return_imgs
        else:
            return imgs

    def perform_colour_jitter(self, imgs):
        if 0.2 < np.random.uniform(0, 1):
            return imgs

        brightness = 0.8
        contrast = 0.8
        saturation = 0.8
        hue = 0.2
        cj = T.ColorJitter(brightness, contrast, saturation, hue)

        brightness = cj._check_input(brightness, "brightness")
        contrast = cj._check_input(contrast, "contrast")
        saturation = cj._check_input(saturation, "saturation")
        hue = cj._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = cj.get_params(
            brightness, contrast, saturation, hue
        )

        return_imgs = []
        for img in imgs:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
            return_imgs.append(img)

        return return_imgs

    def __call__(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        if self.random_resized_crop:
            pil_images = self.perform_random_resized_crop(pil_images)

        if self.horizontal_flip:
            pil_images = self.perform_horizon_flip(pil_images)

        if self.colour_jitter:
            pil_images = self.perform_colour_jitter(pil_images)

        if self.grayscale:
            pil_images = self.perform_random_grayscale(pil_images)

        if self.gaussian_blur:
            pil_images = self.perform_gaussian_blur(pil_images)

        if self.normalise:
            return [self.final_transform(e) for e in pil_images]

        return pil_images


class SimCLRTransformNoRandomResizedCrop:
    def __init__(self, config: Config):
        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        )

        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=None, sigmas=(0.1, 2), prob=0.5),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __call__(self, x):
        return self.transform(x)


def get_transform(transform_name, config: Config):
    logging.info('Initialising transform: %s', transform_name)

    if transform_name == 'simclr':
        tfm = SimCLRTransform(
            input_size=config.data.image_size,
            normalize={"mean": MEAN, "std": STD}
        )
    elif transform_name == 'deepset':
        tfm = DeepSetTransform(config)
    else:
        raise NotImplementedError(f'Transform {transform_name} not supported.')

    return tfm
