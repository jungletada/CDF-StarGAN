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
from torchvision.datasets import ImageFolder


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def listdir(dname):
    fnames = list(chain(
        *[list(Path(dname).rglob('*.' + ext))  
          for ext in ['png', 'jpg', 'jpeg', 'JPG', 'tiff']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root):
        self.samples = listdir(root)
        self.samples.sort()
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        image = torch.from_numpy(np.array(Image.open(fname).convert('L')))
        image = image / 255. * 2. - 1. #  [0, 255] -> [-1, 1]
        image = image.unsqueeze(0)
        return {'image': image, 'filename':str(os.path.basename(fname))}

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root):
        self.samples, self.targets = self._make_dataset(root)

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
        image1 = torch.from_numpy(np.array(Image.open(fname).convert('L')))
        image1 = image1 / 255. * 2. - 1. #  [0, 255] -> [-1, 1]
        image1 = image1.unsqueeze(0)

        image2 = torch.from_numpy(np.array(Image.open(fname2).convert('L')))
        image2 = image2 / 255. * 2. - 1. #  [0, 255] -> [-1, 1]
        image2 = image2.unsqueeze(0)
        return image1, image2, label

    def __len__(self):
        return len(self.targets)


class CustomImageFolder(ImageFolder):
    def __init__(self, root, loader=None, is_valid_file=None):
        super(CustomImageFolder, self).__init__(
            root, loader=loader, is_valid_file=is_valid_file)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = torch.from_numpy(np.array(Image.open(path).convert('L')))
        image = image / 255. * 2. - 1. #  [0, 255] -> [-1, 1]
        image = image.unsqueeze(0)
        return image, target


def get_train_loader(root, which='source', batch_size=8, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)
    
    if which == 'source':
        dataset = CustomImageFolder(root)
    elif which == 'reference':
        dataset = ReferenceDataset(root)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)


def get_eval_loader(root, batch_size=32, shuffle=True, num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    dataset = DefaultDataset(root)
    return data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last)


def get_test_loader(root, batch_size=32, shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    dataset = CustomImageFolder(root)
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