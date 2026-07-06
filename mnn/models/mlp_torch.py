# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..mnn_core.nn.activation_torch import ConstantCurrentActivationTorch, MomentActivation
from ..mnn_core.nn.custom_batch_norm_torch import CustomMomentBatchNorm1d
from ..mnn_core.nn.ensemble_torch import MomentBlock
from ..mnn_core.nn.linear_torch import MomentLinear, MomentLinearNoCorrelation
from ..mnn_core.nn import functional_torch as functional


def _sparse_degree_for_layer(sparse_degree: Optional[int | Sequence[int]], index: int) -> Optional[int]:
    if sparse_degree is None:
        return None
    if isinstance(sparse_degree, int):
        return sparse_degree
    return sparse_degree[index]


def _forward_moment_layers(
    inputs: Tuple[Tensor, Tensor],
    layers: torch.nn.ModuleList,
    decoder: Optional[torch.nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    mean, covariance = inputs
    for module in layers:
        mean, covariance = module(mean, covariance)
    if decoder is not None:
        mean, covariance = decoder(mean, covariance)
    return mean, covariance


def decoding_layer(
    num_neurons: int,
    num_classes: Optional[int],
    *,
    bias: bool = True,
    use_mean: bool = True,
    use_covariance: bool = False,
    signal_correlation: bool = False,
) -> Optional[torch.nn.Module]:
    if not use_mean and not use_covariance:
        raise ValueError("use_mean and use_covariance cannot both be False")
    if not use_covariance:
        return None
    covariance_dims = int(num_neurons * (num_neurons + 1) / 2)
    in_features = covariance_dims if (signal_correlation or not use_mean) else covariance_dims + num_neurons
    if num_classes is None:
        return torch.nn.Identity()
    return torch.nn.Linear(in_features, num_classes, bias=bias)


def decode_moments(
    mean: Tensor,
    covariance: Tensor,
    *,
    use_mean: bool = True,
    use_covariance: bool = False,
    signal_correlation: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    if not use_mean and not use_covariance:
        raise ValueError("use_mean and use_covariance cannot both be False")
    if not use_covariance:
        return mean, covariance
    if not use_mean:
        return functional.triu_vec(covariance).squeeze(-1)
    if signal_correlation:
        signal = functional.compute_signal_correlation(mean, covariance)
        return functional.triu_vec(signal).squeeze(-1)
    covariance_features = functional.triu_vec(covariance).squeeze(-1)
    return torch.cat([mean, covariance_features], dim=-1)


class SpikeMomentMlp(torch.nn.Module):
    def __init__(
        self,
        structure: Sequence[int],
        num_classes: Optional[int] = 10,
        *,
        use_mean: bool = True,
        use_covariance: bool = False,
        signal_correlation: bool = False,
        sparse_degree: Optional[int | Sequence[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.structure = list(structure)
        self.num_classes = num_classes
        self.use_mean = use_mean
        self.use_covariance = use_covariance
        self.signal_correlation = signal_correlation
        self.layers = torch.nn.ModuleList()
        if len(self.structure) >= 2:
            for index, (in_features, out_features) in enumerate(zip(self.structure[:-1], self.structure[1:])):
                self.layers.append(
                    MomentBlock(
                        in_features,
                        out_features,
                        sparse_degree=_sparse_degree_for_layer(sparse_degree, index),
                        **kwargs,
                    )
                )
        if num_classes is None:
            self.predict = None
        else:
            self.predict = MomentBlock(self.structure[-1], num_classes, **kwargs)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
        mean, covariance = _forward_moment_layers(inputs, self.layers, self.predict)
        return decode_moments(
            mean,
            covariance,
            use_mean=self.use_mean,
            use_covariance=self.use_covariance,
            signal_correlation=self.signal_correlation,
        )


class MomentMlp(torch.nn.Module):
    def __init__(
        self,
        structure: Sequence[int],
        num_classes: Optional[int] = 10,
        *,
        norm_bias_variance: bool = False,
        predict_bias: bool = False,
        predict_bias_variance: bool = False,
        use_mean: bool = True,
        use_covariance: bool = False,
        special_init: bool = True,
        dropout: Optional[float] = None,
        momentum: float = 0.9,
        eps: float = 1e-5,
        record_mean_variance: bool = False,
        signal_correlation: bool = False,
        linear_bias: bool = False,
        linear_bias_variance: bool = False,
        sparse_degree: Optional[int | Sequence[int]] = None,
        params_for_criterion: bool = False,
        **legacy_aliases,
    ) -> None:
        super().__init__()
        norm_bias_variance = legacy_aliases.pop("bn_bias_var", norm_bias_variance)
        linear_bias = legacy_aliases.pop("ln_bias", linear_bias)
        linear_bias_variance = legacy_aliases.pop("ln_bias_var", linear_bias_variance)
        record_mean_variance = legacy_aliases.pop("record_bn_mean_var", record_mean_variance)
        if legacy_aliases:
            unknown = ", ".join(sorted(legacy_aliases))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.structure = list(structure)
        self.num_classes = num_classes
        self.use_mean = use_mean
        self.use_covariance = use_covariance
        self.signal_correlation = signal_correlation
        self.layers = torch.nn.ModuleList()
        for index, (in_features, out_features) in enumerate(zip(self.structure[:-1], self.structure[1:])):
            self.layers.append(
                MomentLinear(
                    in_features,
                    out_features,
                    bias=linear_bias,
                    bias_variance=linear_bias_variance,
                    dropout=dropout,
                    sparse_degree=_sparse_degree_for_layer(sparse_degree, index),
                )
            )
            self.layers.append(
                CustomMomentBatchNorm1d(
                    out_features,
                    bias_variance=norm_bias_variance,
                    special_init=special_init,
                    momentum=momentum,
                    eps=eps,
                    record_mean_variance=record_mean_variance,
                )
            )
            self.layers.append(MomentActivation())

        self.decoder = decoding_layer(
            self.structure[-1],
            num_classes,
            bias=predict_bias,
            use_mean=use_mean,
            use_covariance=use_covariance,
            signal_correlation=signal_correlation,
        )
        if self.decoder is None:
            self.predict = None if num_classes is None else MomentLinear(
                self.structure[-1],
                num_classes,
                bias=predict_bias,
                bias_variance=predict_bias_variance,
            )
        else:
            self.predict = self.decoder
        if params_for_criterion and num_classes is not None:
            self.register_parameter("criterion_params", torch.nn.Parameter(torch.ones(num_classes - 1)))

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
        mean, covariance = _forward_moment_layers(inputs, self.layers)
        decoded = decode_moments(
            mean,
            covariance,
            use_mean=self.use_mean,
            use_covariance=self.use_covariance,
            signal_correlation=self.signal_correlation,
        )
        if self.predict is None:
            return decoded
        if isinstance(decoded, tuple):
            return self.predict(*decoded)
        return self.predict(decoded)


class MomentMlpNoCorrelation(torch.nn.Module):
    def __init__(
        self,
        structure: Sequence[int],
        num_classes: int = 10,
        *,
        norm_bias_variance: bool = False,
        predict_bias: bool = False,
        predict_bias_variance: bool = False,
        linear_bias: bool = False,
        linear_bias_variance: bool = False,
        record_mean_variance: bool = False,
        special_init: bool = True,
        momentum: float = 0.9,
        eps: float = 1e-5,
        **legacy_aliases,
    ) -> None:
        super().__init__()
        norm_bias_variance = legacy_aliases.pop("bn_bias_var", norm_bias_variance)
        linear_bias = legacy_aliases.pop("ln_bias", linear_bias)
        linear_bias_variance = legacy_aliases.pop("ln_bias_var", linear_bias_variance)
        record_mean_variance = legacy_aliases.pop("record_bn_mean_var", record_mean_variance)
        if legacy_aliases:
            unknown = ", ".join(sorted(legacy_aliases))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.structure = list(structure)
        layers: list[torch.nn.Module] = []
        for in_features, out_features in zip(self.structure[:-1], self.structure[1:]):
            layers.append(MomentLinearNoCorrelation(in_features, out_features, bias=linear_bias, bias_variance=linear_bias_variance))
            layers.append(
                CustomMomentBatchNorm1d(
                    out_features,
                    eps=eps,
                    momentum=momentum,
                    bias_variance=norm_bias_variance,
                    special_init=special_init,
                    record_mean_variance=record_mean_variance,
                )
            )
            layers.append(MomentActivation())
        self.layers = torch.nn.Sequential(*layers)
        self.predict = MomentLinearNoCorrelation(
            self.structure[-1],
            num_classes,
            bias=predict_bias,
            bias_variance=predict_bias_variance,
        )

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, covariance = functional.parse_input(args)
        mean, covariance = self.layers((mean, covariance))
        return self.predict(mean, covariance)


class MomentRateMlp(torch.nn.Module):
    def __init__(
        self,
        structure: Sequence[int],
        num_classes: int = 10,
        *,
        predict_bias: bool = True,
        momentum: float = 0.9,
        eps: float = 1e-5,
        linear_bias: bool = False,
        threshold_voltage: float = 20.0,
        membrane_conductance: float = 0.05,
        refractory_time: float = 5.0,
        batch_norm_init: bool = True,
        **legacy_aliases,
    ) -> None:
        super().__init__()
        threshold_voltage = legacy_aliases.pop("V_th", threshold_voltage)
        membrane_conductance = legacy_aliases.pop("L", membrane_conductance)
        refractory_time = legacy_aliases.pop("T_ref", refractory_time)
        batch_norm_init = legacy_aliases.pop("bn_init", batch_norm_init)
        linear_bias = legacy_aliases.pop("ln_bias", linear_bias)
        if legacy_aliases:
            unknown = ", ".join(sorted(legacy_aliases))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.structure = list(structure)
        layers: list[torch.nn.Module] = []
        for in_features, out_features in zip(self.structure[:-1], self.structure[1:]):
            layers.append(torch.nn.Linear(in_features, out_features, bias=linear_bias))
            layers.append(torch.nn.BatchNorm1d(out_features, eps=eps, momentum=momentum))
            layers.append(
                ConstantCurrentActivationTorch(
                    threshold_voltage=threshold_voltage,
                    membrane_conductance=membrane_conductance,
                    refractory_time=refractory_time,
                )
            )
        self.layers = torch.nn.Sequential(*layers)
        self.predict = torch.nn.Linear(self.structure[-1], num_classes, bias=predict_bias)
        if batch_norm_init:
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.weight.fill_(2.5)
                        module.bias.fill_(2.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.predict(self.layers(x))


class AnnMlpTorch(torch.nn.Module):
    def __init__(
        self,
        structure: Sequence[int],
        num_classes: int = 10,
        *,
        use_batch_norm: bool = True,
        predict_bias: bool = True,
        activation: str = "relu",
        **legacy_aliases,
    ) -> None:
        super().__init__()
        use_batch_norm = legacy_aliases.pop("need_bn", use_batch_norm)
        activation = legacy_aliases.pop("activation_func", activation)
        if legacy_aliases:
            unknown = ", ".join(sorted(legacy_aliases))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.structure = list(structure)
        layers: list[torch.nn.Module] = []
        for in_features, out_features in zip(self.structure[:-1], self.structure[1:]):
            layers.append(torch.nn.Linear(in_features, out_features, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(out_features))
            if activation == "gelu":
                layers.append(torch.nn.GELU())
            elif activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            else:
                layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        self.predict = torch.nn.Linear(self.structure[-1], num_classes, bias=predict_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.predict(self.layers(x))
