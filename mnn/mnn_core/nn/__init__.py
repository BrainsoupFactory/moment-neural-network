from __future__ import annotations

from importlib import import_module

from . import functional
from . import functional_torch
from .activation_torch import ConstantCurrentActivationTorch, MomentActivation
from .batch_norm_torch import MomentBatchNorm1d, MomentBatchNorm1dNoCorrelation
from .conv_torch import MomentConv2d
from .criterion_torch import CrossEntropyOnMeanTorch, FidelityLossTorch
from .criterion_torch import GaussianSamplingCrossEntropyLossTorch, GaussianSamplingPredictTorch
from .criterion_torch import LabelSmoothingTorch, LikelihoodMSETorch, MSEOnMeanTorch
from .criterion_torch import SampleBasedEarthMoverLossTorch
from .custom_batch_norm_torch import CustomMomentBatchNorm1d
from .ensemble_torch import MomentBlock, MomentBlockNoCorrelation
from .linear_torch import Identity, MomentLinear, MomentLinearNoCorrelation
from .pooling_torch import MomentPooling


_LEGACY_EXPORTS = {
    "LinearDuo": (".linear", "LinearDuo"),
    "LinearNoRho": (".linear", "LinearNoRho"),
    "BatchNorm1dDuo": (".batch_norm", "BatchNorm1dDuo"),
    "BatchNorm1dNoRho": (".batch_norm", "BatchNorm1dNoRho"),
    "OriginMnnActivation": (".activation", "OriginMnnActivation"),
    "ConstantCurrentActivation": (".activation", "ConstantCurrentActivation"),
    "CustomBatchNorm1D": (".custom_batch_norm", "CustomBatchNorm1D"),
    "EnsembleLinearDuo": (".ensemble", "EnsembleLinearDuo"),
    "EnsembleLinearNoRho": (".ensemble", "EnsembleLinearNoRho"),
    "LabelSmoothing": (".criterion", "LabelSmoothing"),
    "CrossEntropyOnMean": (".criterion", "CrossEntropyOnMean"),
    "MSEOnMean": (".criterion", "MSEOnMean"),
    "LikelihoodMSE": (".criterion", "LikelihoodMSE"),
    "GaussianSamplingCrossEntropyLoss": (".criterion", "GaussianSamplingCrossEntropyLoss"),
    "GaussianSamplingPredict": (".criterion", "GaussianSamplingPredict"),
    "SampleBasedEarthMoverLoss": (".criterion", "SampleBasedEarthMoverLoss"),
    "FidelityLoss": (".criterion", "FidelityLoss"),
    "MnnPooling": (".pooling", "MnnPooling"),
}


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        module_name, attr_name = _LEGACY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "functional",
    "functional_torch",
    "MomentActivation",
    "ConstantCurrentActivationTorch",
    "MomentLinear",
    "MomentLinearNoCorrelation",
    "Identity",
    "MomentBatchNorm1d",
    "MomentBatchNorm1dNoCorrelation",
    "CustomMomentBatchNorm1d",
    "MomentBlock",
    "MomentBlockNoCorrelation",
    "MomentPooling",
    "MomentConv2d",
    "LabelSmoothingTorch",
    "CrossEntropyOnMeanTorch",
    "MSEOnMeanTorch",
    "LikelihoodMSETorch",
    "GaussianSamplingCrossEntropyLossTorch",
    "GaussianSamplingPredictTorch",
    "SampleBasedEarthMoverLossTorch",
    "FidelityLossTorch",
    *_LEGACY_EXPORTS.keys(),
]
