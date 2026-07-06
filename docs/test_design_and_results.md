# Test Design and Results

This document describes the test design, main test cases, and most recent local results for the current `tests/` suite. The suite focuses on numerical agreement between the new Torch-first/Pythonic APIs and the legacy APIs, package export compatibility, device and dtype preservation, static dependency constraints, and CPU/GPU timing samples.

## Test Run Metadata

| Field | Value |
| --- | --- |
| Test date | 2026-07-06 |
| Package version under test | `0.2.0` |
| Legacy comparison baseline | `0.1.0` API behavior retained in legacy modules |
| Command | `python -m unittest discover -s tests -p "test_*.py" -v` |
| CUDA availability | Not available in the most recent run |

## Version Comparison

The current tests are designed around a compatibility comparison between the legacy `0.1.0` API behavior and the new `0.2.0` API surface.

| Area | `0.1.0` Legacy API | `0.2.0` New API | Test coverage |
| --- | --- | --- | --- |
| Core MNN math | `fast_dawson.py`, `mnn_utils.py`, `mnn_pytorch.py`; NumPy/SciPy-backed CPU paths appear in core calculations | `torch_dawson.py`, `torch_core.py`, `torch_activation.py`; Torch-first Tensor paths | Numerical forward/backward agreement, autograd gradients, CPU/GPU timing hooks |
| NN layers | Legacy names such as `OriginMnnActivation`, `LinearDuo`, `CustomBatchNorm1D` | Pythonic names such as `MomentActivation`, `MomentLinear`, `CustomMomentBatchNorm1d`, `MomentBlock` | Layer-by-layer output/gradient agreement and static dependency checks |
| Models | Legacy MLP/ANN/SNN-style wrappers import legacy NN components | `MomentMlp`, `MomentRateMlp`, `SpikeMomentMlp`, `AnnMlpTorch`, `CnnWithPoolingClassifier` use new Torch-first NN APIs | Forward agreement, export checks, CPU benchmark, CUDA device-preservation check |
| SNN | Legacy simulation and conversion helpers, with Loihi path kept as hardware boundary | Torch-first current sources, neurons, monitors/probes, validators, and MNN-to-SNN conversion helpers | Shape/statistics agreement, device/dtype checks, conversion smoke tests |
| Training tools | Public helpers split across `general_prepare.py`, `functional.py`, and `general_train.py` | Single-file `mnn.utils.training_tools_api` entrypoint | Config/helper parity, minimal dataloader/optimizer/criterion setup, training-loop smoke test |

## How to Run

Use the project `mnn` conda environment:

```bash
conda activate mnn
python -m unittest discover -s tests -p "test_*.py" -v
```

The most recent run did not detect CUDA, so CUDA-specific tests were skipped through `unittest.skipUnless(torch.cuda.is_available())`.

## Result Summary

Most recent full run:

```text
Ran 63 tests in 0.518s

OK (skipped=5)
```

The skipped tests were all CUDA-only cases:

- `test_gpu_torch_activation_timing_against_vanilla_cpu`
- `test_gpu_torch_core_timing_against_vanilla_cpu`
- `test_gpu_torch_dawson_timing_against_vanilla_cpu`
- `test_cuda_model_forward_stays_on_gpu`
- `test_correlated_gaussian_current_stays_on_cuda`

## `test_torch_mnn_core.py`

This file validates mathematical agreement between the new core modules `torch_dawson.py`, `torch_core.py`, `torch_activation.py` and the legacy CPU/NumPy/SciPy path. It also records timing samples for the Torch implementation on CPU/GPU against vanilla CPU.

### Dawson Accuracy

Cases:

- `test_first_order_dawson_matches_vanilla`
- `test_second_order_dawson_matches_vanilla`

Design:

- Builds 129 `float64` sample points on `[-8, 8]`.
- Uses legacy `Dawson1.dawson1`, `Dawson1.int_fast`, `Dawson2.dawson2`, and `Dawson2.int_fast`.
- Uses new `DawsonFirstOrder.evaluate()`, `DawsonFirstOrder.integral()`, `DawsonSecondOrder.evaluate()`, and `DawsonSecondOrder.integral()`.
- Compares maximum absolute error and maximum relative error. Some integral and second-order Dawson checks primarily rely on relative error because the approximations can produce very large intermediate magnitudes.

Coverage intent:

- Verifies that the fixed Chebyshev coefficients in the new modules align with legacy output.
- Verifies that the `torch.Tensor` input path does not depend on NumPy/SciPy computation.

### Core Forward/Backward Accuracy

Cases:

- `test_compute_bounds_matches_vanilla`
- `test_forward_methods_match_vanilla`
- `test_backward_methods_match_vanilla`

Design:

- Builds two-dimensional `mean/std` tensors that include zero-standard-deviation, low-mean, and high-mean branches.
- Compares against legacy `Mnn_Core_Func.compute_bound`, `forward_fast_*`, `backward_fast_*`, `fast_forward`, and `fast_backward`.
- Maps those calls to new `MNNCore.compute_bounds()`, `forward_mean()`, `forward_std()`, `forward_chi()`, `backward_mean()`, `backward_std()`, `backward_chi()`, `forward()`, and `backward()`.
- Additionally checks that the `compute_bounds` mask exactly matches the legacy mask.

Coverage intent:

- Verifies compatibility across piecewise branches, masks, zero-variance boundaries, and aggregate forward/backward interfaces.
- Ensures the new function decomposition does not change the legacy formulas.

### Activation Accuracy and Autograd

Cases:

- `test_activation_without_correlation_matches_vanilla_forward_and_backward`
- `test_activation_with_correlation_matches_vanilla_forward_and_backward`
- `test_activation_float32_cutoff_region_uses_stable_internal_precision`

Design:

- Compares legacy `mnn_activate_no_rho` with new `mnn_activation_without_correlation`.
- Compares legacy `mnn_activate_trio` with new `mnn_activation_with_correlation`.
- Compares every forward output tensor.
- Sums all outputs and runs backward, then compares gradients for `mean`, `std`, and `correlation`.
- Exercises a float32 cutoff-region input where the activation internally promotes calculations to float64, then verifies finite forward outputs, finite gradients, and float32 output dtype preservation.

Coverage intent:

- Verifies that the new Torch-only custom autograd path matches legacy forward and backward behavior.
- Ensures correlation-matrix diagonal handling keeps the legacy behavior.

### Benchmarks

Cases:

- `test_cpu_torch_dawson_timing_against_vanilla_cpu`
- `test_cpu_torch_core_timing_against_vanilla_cpu`
- `test_cpu_torch_activation_timing_against_vanilla_cpu`
- CUDA counterparts when CUDA is available

Design:

- Uses 2048 `float64` points and averages over 10 repeats.
- CPU tests compare vanilla CPU and Torch CPU.
- GPU tests compare vanilla CPU and Torch CUDA, synchronizing CUDA after each call.
- Benchmark tests still assert numerical agreement, so they do not measure speed without correctness.

Most recent CPU results:

| Case | Vanilla CPU | Torch CPU |
| --- | ---: | ---: |
| `activation_without_correlation` | 0.000483s/call | 0.001550s/call |
| `activation_with_correlation` | 0.003923s/call | 0.006208s/call |
| `nn.MomentActivation` | N/A | 0.014487s/call |
| `core.forward` | 0.000475s/call | 0.001525s/call |
| `core.backward` | 0.000665s/call | 0.003703s/call |
| `dawson_first.evaluate` | 0.000018s/call | 0.000017s/call |
| `dawson_first.integral` | 0.000079s/call | 0.000442s/call |
| `dawson_second.evaluate` | 0.000142s/call | 0.000599s/call |
| `dawson_second.integral` | 0.000123s/call | 0.000653s/call |

Notes:

- On the current CPU, the Torch implementation is usually slower than the NumPy/SciPy legacy path, but it preserves Tensor, dtype, and GPU semantics.
- `dawson_first.evaluate` was slightly faster in Torch in this CPU sample.
- GPU timing should be refreshed in a CUDA-enabled environment.

## `test_torch_nn_api.py`

This file validates the new Pythonic Torch API under `mnn.mnn_core.nn`, including package exports, numerical agreement with legacy layers, and best-practice structural constraints.

### Import and Static Dependency Tests

Cases:

- `test_new_api_exports_from_nn_package`
- `test_new_modules_do_not_import_legacy_core_modules`

Design:

- Verifies that package-level exports in `mnn.mnn_core.nn` point to the new classes, including `MomentActivation`, `MomentLinear`, `MomentBatchNorm1d`, `CustomMomentBatchNorm1d`, `MomentBlock`, and `MomentPooling`.
- Uses `inspect.getsource` to statically check that new `*_torch.py` modules do not contain `mnn_pytorch`, `mnn_utils`, or `fast_dawson`.

Coverage intent:

- Ensures the explicit new API is importable from the package entrypoint.
- Prevents the new NN layers from routing back through legacy core implementations.

### Layer Accuracy Tests

Cases:

- `test_activation_matches_legacy_forward_and_backward`
- `test_constant_current_activation_matches_legacy`
- `test_linear_matches_legacy`
- `test_linear_no_correlation_matches_legacy`
- `test_batch_norm_matches_legacy_eval`
- `test_batch_norm_no_correlation_matches_legacy_eval`
- `test_custom_batch_norm_matches_legacy_eval`
- `test_pooling_matches_legacy`
- `test_criterion_matches_legacy`
- `test_ensemble_matches_legacy_eval`
- `test_ensemble_no_correlation_matches_legacy_eval`

Design:

- Builds `float64` mean/covariance/std inputs.
- Instantiates legacy and new modules, then copies legacy weights, biases, running statistics, and variance-bias parameters into the new modules under `torch.no_grad()`.
- Uses eval-mode batch norm and ensemble tests to avoid training-statistics differences.
- Activation tests also compare backward gradients.

Coverage intent:

- Verifies that new Pythonic class names and internal implementations preserve legacy layer mathematics.
- Covers both correlation and no-correlation paths.
- Covers loss, pooling, batch norm, linear, activation, and ensemble layer behavior.

### Best-Practice Tests

Cases:

- `test_sparse_mask_is_buffer_and_uses_torch_shape`
- `test_moment_conv_uses_module_list`
- `test_moment_conv_forward_shapes`

Design:

- Checks that `sparse_mask` is registered as a buffer, has the same shape as `weight`, and has the expected number of sparse connections per output neuron.
- Checks that convolution submodules are registered through `torch.nn.ModuleList`, so PyTorch can track their parameters.
- Checks mean and covariance output shapes for convolution forward.

Coverage intent:

- Guards against unregistered parameters/buffers, NumPy-based sparse masks, and shape mismatches in the new implementation.

## `test_torch_models_api.py`

This file validates the new Torch-first model API in `mnn.models`, including MLP, rate-only, spike-style, and ANN wrappers.

### Export and Static Dependency Tests

Cases:

- `test_exports_new_model_api`
- `test_new_model_sources_do_not_use_legacy_nn_api`

Design:

- Verifies package-level exports for `MomentMlp`, `SpikeMomentMlp`, `MomentRateMlp`, and `AnnMlpTorch`.
- Statically checks that `models.mlp_torch` and `models.cnn_torch` do not reference legacy NN names: `LinearDuo`, `OriginMnnActivation`, `CustomBatchNorm1D`, or legacy `mnn_core.nn.functional`.

Coverage intent:

- Ensures the new models API depends on new NN layers rather than indirectly reusing legacy implementations.

### Forward Compatibility Tests

Cases:

- `test_moment_mlp_matches_legacy_forward`
- `test_spike_moment_mlp_matches_legacy_forward`
- `test_rate_mlp_matches_legacy_forward`
- `test_ann_mlp_torch_shape_matches_legacy`

Design:

- Copies legacy model weights and batch norm state into the new MNN, SNN-style, and mean-only/rate MLP models.
- Uses fixed-seed `float64` mean/covariance inputs or ordinary Tensor inputs.
- Compares forward outputs numerically or by shape, depending on the model type.

Coverage intent:

- Confirms that the new model construction preserves legacy model composition semantics.
- Keeps the ANN wrapper check scoped to output shape, preventing ordinary Tensor models from accidentally adopting moment-tuple semantics.

### Benchmarks and CUDA

Cases:

- `test_cpu_benchmark_model_api`
- `test_cuda_model_forward_stays_on_gpu`

Design:

- CPU benchmark compares legacy `MnnMlp` and new `MomentMlp`, averaging over 3 repeats.
- CUDA test verifies that `MomentMlp` inputs and outputs stay on GPU when CUDA is available.

Most recent CPU results:

| Case | Time |
| --- | ---: |
| legacy CPU `MnnMlp` | 0.000655s/call |
| Torch CPU `MomentMlp` | 0.001283s/call |

## `test_torch_snn_api.py`

This file validates the new Torch-first API under `mnn.snn` and `mnn.snn.base`, including current sources, Poisson spike sources, LIF neurons, monitors/probes, top-level functional helpers, and MNN-to-SNN conversion.

### Export and Static Dependency Tests

Cases:

- `test_exports_new_snn_api`
- `test_new_snn_sources_do_not_use_numpy_or_legacy_core`
- `test_top_level_functional_torch_exports`

Design:

- Verifies package-level exports for new APIs such as `LifNeurons`, `SpikeMonitorTorch`, `GaussianCurrentSource`, `PoissonSpikeSource`, and `MomentMlpToSnn`.
- Statically checks that new SNN Torch modules do not contain `import numpy`, `import scipy`, `.numpy()`, `from_numpy`, `.data`, or legacy core dependencies.
- Verifies that `snn.functional_torch` validators and helpers are also exported at the top level.

Coverage intent:

- Ensures the core calculation path of the new SNN implementation is Torch-first.
- Prevents hidden CPU/NumPy fallback in simulation or conversion code.

### Functional Helper Tests

Cases:

- `test_sample_shape_matches_legacy_sample_size`
- `test_top_level_poisson_sample_matches_legacy_shape`
- `test_top_level_sparse_statistics_matches_legacy`
- `test_condition_modifier_uses_torch_correlation`

Design:

- Checks `sample_shape` against legacy sample-size semantics.
- Checks Poisson sampling shape and dtype against the legacy helper.
- Compares sparse spike train statistics with legacy `sparse_spike_train_statistics`.
- Checks that `MomentSnnValidator.modify_value_by_condition` zeros the mean and preserves covariance diagonals under the `corr_only` condition.

Coverage intent:

- Covers the parallel refactor of the top-level `snn/functional.py` API.
- Confirms that statistical helpers and condition-modification logic preserve legacy semantics.

### Current Source and Device Tests

Cases:

- `test_independent_gaussian_current_shape_dtype_device`
- `test_correlated_gaussian_current_uses_torch_device`
- `test_correlated_gaussian_current_stays_on_cuda`
- `test_poisson_source_shape_matches_legacy`

Design:

- Independent Gaussian current checks output shape, dtype, and device.
- Correlated Gaussian current uses the covariance/Cholesky path and checks pregenerated output shape `(num_steps, *shape)`.
- CUDA test verifies that outputs stay on GPU when CUDA is available.
- Poisson spike source compares output shape against the legacy generator.

Coverage intent:

- Verifies that random current sources do not fall back to CPU and preserve the dtype/device of their Tensor inputs.
- Uses shape/statistical-interface checks for legacy random semantics rather than requiring sample-by-sample equality.

### Neuron, Monitor, Probe, and Conversion Tests

Cases:

- `test_lif_neurons_matches_legacy_single_step`
- `test_monitor_and_probe_behavior`
- `test_convert_moment_parameters_and_snn_forward`
- `test_spike_moment_to_snn_forward`

Design:

- Compares one-step LIF output with legacy `LIFNeurons`, and checks reset state alignment.
- Compares spike monitor count and duration.
- Compares probe record keys and basic export behavior; probe/export is treated as an explicit CPU collection boundary rather than a hidden computation fallback.
- Checks MNN-to-SNN conversion parameter keys and validates converted forward output shape.

Coverage intent:

- Covers simulation state, recorders, and new model conversion paths.
- Confirms that conversion APIs can identify new models/layers and produce runnable SNN forward paths.

### Benchmark

Case:

- `test_cpu_benchmark_snn_current`

Most recent CPU results:

| Case | Time |
| --- | ---: |
| legacy CPU `GaussianCurrentGenerator` | 0.000007s/call |
| Torch CPU `GaussianCurrentSource` | 0.000007s/call |

## `test_training_tools_api.py`

This file validates the new single-file training tools entrypoint `mnn.utils.training_tools_api` while ensuring the legacy `mnn.utils.training_tools` package still works.

### Export and Static Dependency Tests

Cases:

- `test_exports_and_legacy_package_still_work`
- `test_source_does_not_import_legacy_training_modules`

Design:

- Verifies that the new API exposes `TrainProcessCollections`.
- Verifies that legacy package-level exports still point to objects from legacy `general_prepare`, `functional`, and `general_train`.
- Statically checks that the new single-file API does not proxy old modules through `from . import general_prepare`, `from . import functional`, or `from . import general_train`.

Coverage intent:

- Ensures the new single-file API is an integrated implementation rather than a facade over the old modules.
- Ensures legacy entrypoints remain available.

### Config and Helper Tests

Cases:

- `test_config_read_and_reset_match_legacy`
- `test_transform_compose_matches_legacy_type`
- `test_input_predict_score_helpers`
- `test_average_and_progress_meter`

Design:

- Uses a temporary YAML file to compare `set_config2args` behavior against legacy behavior, including special-argument reset.
- Checks that transform composition returns the same object type as the legacy helper.
- Checks `InputPreprocess` dtype conversion and mean masking.
- Checks argmax prediction and equality-based scoring helpers.
- Checks `AverageMeter` accumulation and `ProgressMeter` display text.

Coverage intent:

- Covers the core training-flow utilities for configuration, input preprocessing, prediction, scoring, and logging display.

### Training Preparation and Pipeline Smoke Tests

Cases:

- `test_dataloader_optimizer_and_criterion_smoke`
- `test_train_and_validate_smoke`

Design:

- Uses a tiny `TensorDataset` to avoid external dataset downloads.
- Verifies minimal usable paths for `make_dataloader`, `prepare_optimizer_scheduler`, and `make_criterion`.
- Runs `TrainProcessCollections.train_one_epoch` and `validate` with `torch.nn.Linear`, `CrossEntropyLoss`, and `SGD`.

Coverage intent:

- Verifies that the new single-file API can run a minimal training loop.
- Keeps training-flow coverage as a smoke test rather than tying it to real dataset downloads or long training jobs.

### Docs Index Test

Case:

- `test_docs_migration_files_are_indexed`

Design:

- Checks that `docs/index.rst` includes `legacy_nn_migration`, `legacy_models_snn_migration`, and `legacy_training_tools_migration`.
- Checks that the corresponding Markdown files exist.

Coverage intent:

- Ensures newly added migration docs are not omitted from the documentation index.

## Coverage Boundaries

Currently covered:

- New-vs-legacy core mathematical outputs and gradients.
- Explicit exports for new NN layers, models, SNN APIs, and training tools.
- Static dependency constraints for new Torch-first modules.
- CPU benchmarks and GPU device-preservation checks when CUDA is available.
- Minimal training-loop smoke coverage.

Not covered, or only lightly covered:

- CUDA benchmarks were not run on the current machine and should be refreshed in a CUDA-enabled environment.
- Random SNN sampling mainly checks shape, dtype, device, and statistical helpers, not sample-by-sample equality.
- The ANN wrapper currently checks output shape only.
- Training tools do not run real dataset downloads, distributed training, multiple scheduler argument combinations, or full checkpoint resume workflows.
- Documentation build was not run in this command; documentation references are checked through unittest index assertions only.
