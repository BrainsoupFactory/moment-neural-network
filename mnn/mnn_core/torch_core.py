# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import torch

from .torch_dawson import DawsonFirstOrder, DawsonSecondOrder


@dataclass
class MNNParameters:
    excitatory_inhibitory_ratio: float = 0.0
    membrane_conductance: float = 0.05
    refractory_time: float = 5.0
    threshold_voltage: float = 20.0
    resting_voltage: float = 0.0
    eps: float = 1e-5
    cut_off: float = 10.0
    ignore_refractory_time: bool = True
    degree: int = 100

    @property
    def correction_factor(self) -> float:
        return 2 / sqrt(2 * self.membrane_conductance)

    @property
    def special_factor(self) -> float:
        return 4 / sqrt(2 * pi * self.membrane_conductance) * 0.8862269251743827

    def reset(self) -> None:
        self.excitatory_inhibitory_ratio = 0.0
        self.membrane_conductance = 0.05
        self.refractory_time = 5.0
        self.threshold_voltage = 20.0
        self.resting_voltage = 0.0
        self.eps = 1e-5
        self.cut_off = 10.0
        self.ignore_refractory_time = True
        self.degree = 100


class MNNCore:
    def __init__(self, parameters: MNNParameters | None = None) -> None:
        self.parameters = parameters if parameters is not None else MNNParameters()
        self.first_order_dawson = DawsonFirstOrder()
        self.second_order_dawson = DawsonSecondOrder()

    @property
    def threshold(self) -> float:
        return self.parameters.threshold_voltage * self.parameters.membrane_conductance

    @property
    def sqrt_conductance(self) -> float:
        return sqrt(self.parameters.membrane_conductance)

    def compute_bounds(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_std = std > 0
        denominator = self.sqrt_conductance * std + (~positive_std).to(dtype=std.dtype)
        upper_bound = (self.threshold - mean) / denominator
        lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean) / denominator
        return upper_bound, lower_bound, positive_std

    def forward_mean(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        output_mean = torch.zeros_like(mean)
        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            mean_interval = (
                2
                / self.parameters.membrane_conductance
                * (self.first_order_dawson.integral(upper_bound) - self.first_order_dawson.integral(lower_bound))
            )
            output_mean[active] = 1 / (mean_interval + self.parameters.refractory_time)

        suprathreshold = (~positive_std) & (mean > self.threshold)
        if torch.any(suprathreshold):
            output_mean[suprathreshold] = 1 / (
                self.parameters.refractory_time
                - 1
                / self.parameters.membrane_conductance
                * torch.log(1 - self.threshold / mean[suprathreshold])
            )
        return output_mean

    def backward_mean(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        grad_mean_mean = torch.zeros_like(mean)
        grad_mean_std = torch.zeros_like(mean)

        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            upper_g = self.first_order_dawson.evaluate(upper_bound)
            lower_g = self.first_order_dawson.evaluate(lower_bound)
            delta_g = upper_g - lower_g
            scale = output_mean[active] * output_mean[active] / std[active]
            grad_mean_mean[active] = (
                scale * delta_g * 2 / self.parameters.membrane_conductance / self.sqrt_conductance
            )
            grad_mean_std[active] = (
                scale
                * (upper_g * upper_bound - lower_g * lower_bound)
                * 2
                / self.parameters.membrane_conductance
            )

        suprathreshold = (~positive_std) & (mean > self.threshold)
        if torch.any(suprathreshold):
            grad_mean_mean[suprathreshold] = (
                self.parameters.threshold_voltage
                * output_mean[suprathreshold]
                * output_mean[suprathreshold]
                / mean[suprathreshold]
                / (mean[suprathreshold] - self.threshold)
            )
        return grad_mean_mean, grad_mean_std

    def forward_std(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
    ) -> torch.Tensor:
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        fano_factor = torch.zeros_like(mean)

        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            variance_interval = (
                8
                / self.parameters.membrane_conductance
                / self.parameters.membrane_conductance
                * (self.second_order_dawson.integral(upper_bound) - self.second_order_dawson.integral(lower_bound))
            )
            fano_factor[active] = variance_interval * output_mean[active] * output_mean[active]

        fano_factor[~positive_std] = (mean[~positive_std] < self.threshold).to(dtype=mean.dtype)
        return torch.sqrt(fano_factor * output_mean)

    def backward_std(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        grad_std_mean = torch.zeros_like(mean)
        grad_std_std = torch.zeros_like(mean)

        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            upper_g = self.first_order_dawson.evaluate(upper_bound)
            lower_g = self.first_order_dawson.evaluate(lower_bound)
            upper_h = self.second_order_dawson.evaluate(upper_bound)
            lower_h = self.second_order_dawson.evaluate(lower_bound)
            delta_g = upper_g - lower_g
            delta_h = upper_h - lower_h
            delta_integral_h = self.second_order_dawson.integral(upper_bound) - self.second_order_dawson.integral(lower_bound)
            ratio = output_std[active] / std[active]
            grad_std_mean[active] = (
                3
                / self.parameters.membrane_conductance
                / self.sqrt_conductance
                * ratio
                * output_mean[active]
                * delta_g
                - 0.5 / self.sqrt_conductance * ratio * delta_h / delta_integral_h
            )
            grad_std_std[active] = (
                3
                / self.parameters.membrane_conductance
                * ratio
                * output_mean[active]
                * (upper_g * upper_bound - lower_g * lower_bound)
                - 0.5
                * ratio
                * (upper_h * upper_bound - lower_h * lower_bound)
                / delta_integral_h
            )

        suprathreshold = (~positive_std) & (mean > self.threshold)
        if torch.any(suprathreshold):
            grad_std_std[suprathreshold] = (
                1
                / sqrt(2 * self.parameters.membrane_conductance)
                * output_mean[suprathreshold].pow(1.5)
                * torch.sqrt(
                    1 / (1 - mean[suprathreshold]).pow(2)
                    - 1 / mean[suprathreshold].pow(2)
                )
            )
        return grad_std_mean, grad_std_std

    def forward_chi(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> torch.Tensor:
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        chi = torch.zeros_like(mean)

        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            delta_g = self.first_order_dawson.evaluate(upper_bound) - self.first_order_dawson.evaluate(lower_bound)
            chi[active] = (
                output_mean[active]
                * output_mean[active]
                / output_std[active]
                * delta_g
                * 2
                / self.parameters.membrane_conductance
                / self.sqrt_conductance
            )

        suprathreshold = (~positive_std) & (mean > self.threshold)
        if torch.any(suprathreshold):
            chi[suprathreshold] = (
                sqrt(2 / self.parameters.membrane_conductance)
                / torch.sqrt(
                    self.parameters.refractory_time
                    - 1
                    / self.parameters.membrane_conductance
                    * torch.log(1 - 1 / mean[suprathreshold])
                )
                / torch.sqrt(2 * mean[suprathreshold] - 1)
            )
        return chi

    def backward_chi(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
        chi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grad_mean_mean, grad_mean_std = self.backward_mean(mean, std, output_mean)
        positive_std = std > 0
        active = positive_std & ((self.threshold - mean) < self.parameters.cut_off * self.sqrt_conductance * std)
        grad_chi_mean = torch.zeros_like(mean)
        grad_chi_std = torch.zeros_like(mean)

        if torch.any(active):
            temp = std[active] * self.sqrt_conductance
            upper_bound = (self.threshold - mean[active]) / temp
            lower_bound = (self.parameters.resting_voltage * self.parameters.membrane_conductance - mean[active]) / temp
            upper_g = self.first_order_dawson.evaluate(upper_bound)
            lower_g = self.first_order_dawson.evaluate(lower_bound)
            upper_h = self.second_order_dawson.evaluate(upper_bound)
            lower_h = self.second_order_dawson.evaluate(lower_bound)
            delta_g = upper_g - lower_g
            delta_h = upper_h - lower_h
            delta_integral_h = self.second_order_dawson.integral(upper_bound) - self.second_order_dawson.integral(lower_bound)
            temp_g = upper_g * upper_bound - lower_g * lower_bound
            temp_h = upper_h * upper_bound - lower_h * lower_bound
            chi_active = chi[active]

            grad_chi_mean[active] = (
                0.5 * chi_active / output_mean[active] * grad_mean_mean[active]
                - sqrt(2)
                / self.parameters.membrane_conductance
                * torch.sqrt(output_mean[active] / delta_integral_h)
                * temp_g
                / std[active]
                + chi_active * delta_h / delta_integral_h / 2 / self.sqrt_conductance / std[active]
            )

            second_temp_g = 2 * (upper_g * upper_bound * upper_bound - lower_g * lower_bound * lower_bound)
            second_temp_g = second_temp_g + upper_bound - lower_bound
            grad_chi_std[active] = chi_active * (
                0.5 / output_mean[active] * grad_mean_std[active]
                - (second_temp_g / delta_g) / std[active]
                + 0.5 / std[active] / delta_integral_h * temp_h
            )

        suprathreshold = (~positive_std) & (mean > self.threshold)
        if torch.any(suprathreshold):
            temp = 2 * mean[suprathreshold] / self.threshold - 1
            grad_chi_mean[suprathreshold] = (
                1
                / sqrt(2 * self.parameters.membrane_conductance)
                / torch.sqrt(output_mean[suprathreshold] * temp)
                * grad_mean_mean[suprathreshold]
                - sqrt(2 / self.parameters.membrane_conductance)
                / self.threshold
                * torch.sqrt(output_mean[suprathreshold])
                * temp.pow(-1.5)
            )
        return grad_chi_mean, grad_chi_std

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_mean = self.forward_mean(mean, std)
        output_std = self.forward_std(mean, std, output_mean)
        chi = self.forward_chi(mean, std, output_mean, output_std)
        return output_mean, output_std, chi

    def backward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
        chi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_mean_mean, grad_mean_std = self.backward_mean(mean, std, output_mean)
        grad_std_mean, grad_std_std = self.backward_std(mean, std, output_mean, output_std)
        grad_chi_mean, grad_chi_std = self.backward_chi(mean, std, output_mean, chi)
        return grad_mean_mean, grad_mean_std, grad_std_mean, grad_std_std, grad_chi_mean, grad_chi_std

    def get_parameter(self, name: str):
        return getattr(self.parameters, name)

    def set_parameter(self, name: str, value) -> None:
        setattr(self.parameters, name, value)

    def reset_parameters(self) -> None:
        self.parameters.reset()
