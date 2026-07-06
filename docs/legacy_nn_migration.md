# Legacy `mnn_core.nn` Migration

The original `mnn.mnn_core.nn` classes are legacy APIs. They remain importable for compatibility, but they depend on the older activation path that routes through `mnn_pytorch.py`, `mnn_utils.py`, and `fast_dawson.py`. These legacy APIs are deprecated and are expected to be removed in a future release.

Use the Torch-first APIs exported from `mnn.mnn_core.nn` for new code. The new APIs call `torch_activation.py`, `torch_core.py`, and `torch_dawson.py`, keep tensors on their current device, and avoid NumPy/SciPy computation paths.

## Migration Table

| Legacy API | Torch-first API |
| --- | --- |
| `OriginMnnActivation` | `MomentActivation` |
| `ConstantCurrentActivation` | `ConstantCurrentActivationTorch` |
| `LinearDuo` | `MomentLinear` |
| `LinearNoRho` | `MomentLinearNoCorrelation` |
| `BatchNorm1dDuo` | `MomentBatchNorm1d` |
| `BatchNorm1dNoRho` | `MomentBatchNorm1dNoCorrelation` |
| `CustomBatchNorm1D` | `CustomMomentBatchNorm1d` |
| `EnsembleLinearDuo` | `MomentBlock` |
| `EnsembleLinearNoRho` | `MomentBlockNoCorrelation` |
| `MnnPooling` | `MomentPooling` |
| `MomentConv2d` from `mconv2d.py` | `MomentConv2d` from `conv_torch.py` |
| `LabelSmoothing` | `LabelSmoothingTorch` |
| `CrossEntropyOnMean` | `CrossEntropyOnMeanTorch` |
| `MSEOnMean` | `MSEOnMeanTorch` |
| `LikelihoodMSE` | `LikelihoodMSETorch` |
| `GaussianSamplingCrossEntropyLoss` | `GaussianSamplingCrossEntropyLossTorch` |
| `GaussianSamplingPredict` | `GaussianSamplingPredictTorch` |
| `SampleBasedEarthMoverLoss` | `SampleBasedEarthMoverLossTorch` |
| `FidelityLoss` | `FidelityLossTorch` |

## Example

```python
from mnn.mnn_core.nn import MomentActivation, MomentLinear, CustomMomentBatchNorm1d

layer = MomentLinear(784, 100)
norm = CustomMomentBatchNorm1d(100)
activation = MomentActivation()
```

Legacy names are still available from `mnn.mnn_core.nn`, but new code should not depend on them.
