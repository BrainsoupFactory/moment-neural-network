# -*- coding: utf-8 -*-
import os
from typing import Callable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def cub_bird_dataloader(data_dir: str, train_batch: int = 32, test_batch: int = 32,
                        transform_train: Callable = transforms.Compose([
                            transforms.Resize(448),
                            transforms.CenterCrop(448),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
                        transform_test: Callable = transforms.Compose([
                            transforms.Resize(448),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ]), num_workers: int = 4, pin_memory: bool = True, dataset_only: bool = False, **kwargs):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_set = ImageFolder(root=train_dir, transform=transform_train, **kwargs)
    test_set = ImageFolder(root=test_dir, transform=transform_test, **kwargs)
    if dataset_only:
        return train_set, test_set
    else:
        train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)

        test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

        return train_loader, test_loader