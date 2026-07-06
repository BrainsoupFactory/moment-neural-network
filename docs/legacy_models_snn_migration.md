# Legacy `models` and `snn` Migration

The original `mnn.models` and `mnn.snn` classes remain importable for compatibility, but they are legacy APIs. New code should use the Torch-first APIs added alongside them.

The new APIs use the Pythonic `mnn.mnn_core.nn` Torch modules and avoid the old NumPy/SciPy activation and moment-helper path. Legacy Loihi/NxSDK deployment code in `mnn.snn.snn2loihi` is a hardware boundary and is not part of the Torch-first refactor.

Use these modules for new code:

- `mnn.models.mlp_torch`
- `mnn.models.cnn_torch`
- `mnn.snn.base.*_torch`
- `mnn.snn.mnn_to_snn_torch`
- `mnn.snn.functional_torch`

## Models

| Legacy API | Torch-first API |
| --- | --- |
| `MnnMlp` | `MomentMlp` |
| `MnnMlpNoRho` | `MomentMlpNoCorrelation` |
| `MnnMlpMeanOnly` | `MomentRateMlp` |
| `SnnMlp` | `SpikeMomentMlp` |
| `AnnMlp` | `AnnMlpTorch` |
| `GeneralCnnPool` | `CnnWithPoolingClassifier` |

## SNN

| Legacy API | Torch-first API |
| --- | --- |
| `LIFNeurons` | `LifNeurons` |
| `SpikeMonitor` | `SpikeMonitorTorch` |
| `NeuronProbe` | `NeuronProbeTorch` |
| `GaussianCurrentGenerator` | `GaussianCurrentSource` |
| `PoissonSpikeGenerator` | `PoissonSpikeSource` |
| `MnnMlpTrans` | `MomentMlpToSnn` |
| `SnnMlpTrans` | `SpikeMomentMlpToSnn` |
| `MnnMlpMeanOnlyTrans` | `RateMomentMlpToSnn` |
| `convert_parameters` | `convert_moment_parameters` |

## SNN Validation Helpers

| Legacy API | Torch-first API |
| --- | --- |
| `MnnSnnValidate` | `MomentSnnValidator` |
| `CustomMnnMlp` | `DirectReadoutMomentMlpToSnn` |
| `sample_poisson_spike` | `sample_poisson_spikes` |
| `sparse_spike_train_statistics` | `sparse_spike_train_statistics` in `functional_torch` |
| `snn_validation` | `snn_validation` in `functional_torch` |

## Notes

- New model APIs use clearer names such as `num_classes`, `use_covariance`, `linear_bias`, and `norm_bias_variance`.
- New SNN current sources sample correlated Gaussian currents with Torch tensors on the input device.
- Probe and save/export helpers may explicitly move data to CPU because they are collection boundaries, not core computation paths.
- `mnn.snn.functional_torch` provides the Torch-first validation workflow and should be preferred over `mnn.snn.functional` for new simulation scripts.
- `mnn.snn.snn2loihi` remains legacy because NxSDK, HDF5 export, and plotting workflows require NumPy-facing interfaces.
