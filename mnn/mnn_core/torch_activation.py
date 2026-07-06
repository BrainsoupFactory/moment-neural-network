# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import torch

from .torch_core import MNNCore


_core = MNNCore()

_PARAMETER_ALIASES = {
    "L": "membrane_conductance",
    "vol_th": "threshold_voltage",
    "vol_rest": "resting_voltage",
    "t_ref": "refractory_time",
    "ratio": "excitatory_inhibitory_ratio",
    "ignore_t_ref": "ignore_refractory_time",
}


def _canonical_parameter_name(name: str) -> str:
    return _PARAMETER_ALIASES.get(name, name)


def _set_diagonal(tensor: torch.Tensor, value: float) -> torch.Tensor:
    result = tensor.clone()
    torch.diagonal(result, dim1=-1, dim2=-2).fill_(value)
    return result


class MNNActivationWithCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, mean: torch.Tensor, std: torch.Tensor, correlation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = mean.dtype
        mean64 = mean.to(dtype=torch.float64)
        std64 = std.to(dtype=torch.float64)
        correlation64 = correlation.to(dtype=torch.float64)
        output_mean64, output_std64, chi64 = _core.forward(mean64, std64)
        chi_outer = torch.matmul(chi64.unsqueeze(-1), chi64.unsqueeze(-2))
        output_correlation64 = _set_diagonal(correlation64 * chi_outer, 1.0)
        ctx.input_dtype = input_dtype
        ctx.correlation_dtype = correlation.dtype
        ctx.save_for_backward(mean64, std64, correlation64, output_mean64, output_std64, chi64)
        return (
            output_mean64.to(dtype=input_dtype),
            output_std64.to(dtype=input_dtype),
            output_correlation64.to(dtype=correlation.dtype),
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_grad, std_grad, correlation_grad = grad_outputs
        mean, std, correlation, output_mean, output_std, chi = ctx.saved_tensors
        mean_grad = mean_grad.to(dtype=torch.float64)
        std_grad = std_grad.to(dtype=torch.float64)
        correlation_grad = correlation_grad.to(dtype=torch.float64)
        (
            grad_mean_mean,
            grad_mean_std,
            grad_std_mean,
            grad_std_std,
            grad_chi_mean,
            grad_chi_std,
        ) = _core.backward(mean, std, output_mean, output_std, chi)

        grad_chi_mean = torch.clamp(grad_chi_mean, -1, 1)
        grad_chi_std = torch.clamp(grad_chi_std, -1, 1)

        off_diagonal_correlation = _set_diagonal(correlation, 0.0)
        correlation_contribution = correlation_grad * off_diagonal_correlation
        correlation_contribution = torch.matmul(chi.unsqueeze(-2), correlation_contribution).squeeze(-2) * 2
        correlation_grad_mean = correlation_contribution * grad_chi_mean
        correlation_grad_std = correlation_contribution * grad_chi_std

        mean_input_grad = mean_grad * grad_mean_mean + std_grad * grad_std_mean + correlation_grad_mean
        std_input_grad = mean_grad * grad_mean_std + std_grad * grad_std_std + correlation_grad_std
        correlation_input_grad = _set_diagonal(
            torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)) * correlation_grad,
            0.0,
        )
        return (
            mean_input_grad.to(dtype=ctx.input_dtype),
            std_input_grad.to(dtype=ctx.input_dtype),
            correlation_input_grad.to(dtype=ctx.correlation_dtype),
        )


class MNNActivationWithoutCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = mean.dtype
        mean64 = mean.to(dtype=torch.float64)
        std64 = std.to(dtype=torch.float64)
        output_mean64 = _core.forward_mean(mean64, std64)
        output_std64 = _core.forward_std(mean64, std64, output_mean64)
        ctx.input_dtype = input_dtype
        ctx.save_for_backward(mean64, std64, output_mean64, output_std64)
        return output_mean64.to(dtype=input_dtype), output_std64.to(dtype=input_dtype)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_grad, std_grad = grad_outputs
        mean, std, output_mean, output_std = ctx.saved_tensors
        mean_grad = mean_grad.to(dtype=torch.float64)
        std_grad = std_grad.to(dtype=torch.float64)
        grad_mean_mean, grad_mean_std = _core.backward_mean(mean, std, output_mean)
        grad_std_mean, grad_std_std = _core.backward_std(mean, std, output_mean, output_std)
        mean_input_grad = mean_grad * grad_mean_mean + std_grad * grad_std_mean
        std_input_grad = mean_grad * grad_mean_std + std_grad * grad_std_std
        return mean_input_grad.to(dtype=ctx.input_dtype), std_input_grad.to(dtype=ctx.input_dtype)


class ConstantCurrentActivation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        current: torch.Tensor,
        threshold_voltage: float = 20.0,
        membrane_conductance: float = 0.05,
        refractory_time: float = 5.0,
    ) -> torch.Tensor:
        threshold = threshold_voltage * membrane_conductance
        active = current > threshold
        output = torch.zeros_like(current)
        output[active] = 1 / (
            refractory_time
            - 1 / membrane_conductance * torch.log(1 - threshold / current[active])
        )
        ctx.threshold_voltage = threshold_voltage
        ctx.threshold = threshold
        ctx.save_for_backward(current, output, active)
        return output

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        current, output, active = ctx.saved_tensors
        output_grad = torch.zeros_like(grad)
        output_grad[active] = (
            ctx.threshold_voltage
            * output[active]
            * output[active]
            / current[active]
            / (current[active] - ctx.threshold)
            * grad[active]
        )
        return output_grad, None, None, None


def get_core_parameter(name: str):
    return _core.get_parameter(_canonical_parameter_name(name))


def set_core_parameter(name: str, value) -> None:
    _core.set_parameter(_canonical_parameter_name(name), value)


def reset_core_parameters() -> None:
    _core.reset_parameters()


mnn_activation_with_correlation = MNNActivationWithCorrelation.apply
mnn_activation_without_correlation = MNNActivationWithoutCorrelation.apply
constant_current_activation = ConstantCurrentActivation.apply
