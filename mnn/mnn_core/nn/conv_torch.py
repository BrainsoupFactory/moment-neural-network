# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import Tensor


class MomentConv2d(torch.nn.Module):
    __constants__ = ["in_channels", "out_channels", "kernel_size", "stride"]

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv2d(1, 1, kernel_size, stride=stride, bias=False) for _ in range(out_channels)
        )

    def forward(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, in_channels, height, width = mean.shape
        mean_flat = mean.reshape(batch_size * in_channels, 1, height, width)
        output_mean = None

        for index, convolution in enumerate(self.convolutions):
            convolved = convolution(mean_flat)
            if output_mean is None:
                output_mean = mean.new_empty(batch_size, self.out_channels, convolved.shape[2], convolved.shape[3])
            convolved = convolved.reshape(batch_size, in_channels, convolved.shape[2], convolved.shape[3])
            output_mean[:, index, :, :] = torch.sum(convolved, dim=1)

        batch_size, in_channels, height, width, _, _ = covariance.shape
        covariance_flat = covariance.reshape(batch_size * in_channels * height * width, 1, height, width)
        output_covariance = None

        for index, convolution in enumerate(self.convolutions):
            convolved = convolution(covariance_flat)
            output_height = convolved.shape[2]
            output_width = convolved.shape[3]
            if output_covariance is None:
                output_covariance = covariance.new_empty(
                    batch_size,
                    self.out_channels,
                    output_height,
                    output_width,
                    output_height,
                    output_width,
                )

            convolved = convolved.reshape(batch_size, in_channels, height, width, output_height, output_width)
            convolved = torch.permute(convolved, (0, 1, 4, 5, 2, 3))
            convolved = convolved.reshape(batch_size * in_channels * output_height * output_width, 1, height, width)
            convolved = convolution(convolved)
            convolved = convolved.reshape(batch_size, in_channels, output_height, output_width, output_height, output_width)
            convolved = torch.permute(convolved, (0, 1, 4, 5, 2, 3))
            output_covariance[:, index, :, :, :, :] = torch.sum(convolved, dim=1)

        return output_mean, output_covariance
