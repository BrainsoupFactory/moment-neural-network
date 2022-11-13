# -*- coding: utf-8 -*-
from typing import Callable

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def classic_mnist_loader(data_dir: str, train_batch: int = 128, test_batch: int = 100,
                         transform_train: Callable = transforms.Compose([transforms.RandomCrop(28, padding=2),
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))]),
                         transform_test: Callable = transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,))]),
                         **kwargs):
    train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, **kwargs)

    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False, **kwargs)

    return train_loader, test_loader