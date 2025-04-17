"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import cv2
import torch


import random
import numpy as np
from pathlib import Path
from itertools import chain
from munch import Munch
from PIL import Image

from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def transform_train(contour_image, target_height=256, target_width=512):
    # Rotatation
    angle = np.random.uniform(-5, 5)
    center = (contour_image.shape[1] // 2, contour_image.shape[0] // 2)  # (x, y)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotation flag
    flip_flag = False # np.random.rand() < 0.5
    # Apply rotation to the first image to compute the new dimensions (should remain same as original for cv2.warpAffine)
    # Use cv2.BORDER_REFLECT to avoid black borders.
    def transform_single(image):
        # 1. Random Rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = rotated.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(rotated, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 3. Random horizontal flip (50% chance) using the same decision for all images.
        # (We generate the flip flag once outside, so here we assume that variable is defined)
        # if flip_flag:
        #     resized = np.fliplr(resized)
        
        # 4. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        
        return cropped

    cropped = transform_single(contour_image)
    
    # 5. 将像素值归一化，并反转（1.0 - value），再扩展通道维度
    img = 1.0 - (cropped.astype(np.float32) / 255.0)
    img = np.expand_dims(img, axis=0)  # 形状变为 (1, H, W)
    tensor = torch.from_numpy(img)
    return tensor


def transform_test(image, target_height=256, target_width=512):
    original_height, original_width = image.shape
    scale = target_height / original_height
    new_width = int(original_width * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
    width_left = (resized.shape[1] - target_width) // 2
    cropped = resized[:, width_left:width_left + target_width]
    
    img = 1.0 - (cropped.astype(np.float32) / 255.0)
    img = np.expand_dims(img, axis=0)  # 形状变为 (1, H, W)
    
    tensor = torch.from_numpy(img)
    return tensor


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG', 'tiff']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=transform_test):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('L')
        img = self.transform(np.array(img))
        return {'image': img, 'filename':str(os.path.basename(fname))}

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('L')
        img2 = Image.open(fname2).convert('L')
        img = self.transform(np.array(img))
        img2 = self.transform(np.array(img2))
        return img, img2, label

    def __len__(self):
        return len(self.targets)


class CustomImageFolder(ImageFolder):
    """
    """
    def __init__(self, root, transform, target_height=256, target_width=512, loader=None, is_valid_file=None):
        super(CustomImageFolder, self).__init__(root, loader=loader, is_valid_file=is_valid_file)
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('L')
        sample = sample.convert('L')
        sample_np = np.array(sample)
        tensor = self.transform(sample_np, 
                                 target_height=self.target_height, 
                                 target_width=self.target_width)
        return tensor, target


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)
    
    if which == 'source':
        dataset = CustomImageFolder(root, transform=transform_train)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform=transform_train)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=False, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    
    dataset = DefaultDataset(root, transform=transform_test)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    dataset = CustomImageFolder(root, transform=transform_test)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})