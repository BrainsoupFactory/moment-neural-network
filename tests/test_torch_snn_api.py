# -*- coding: utf-8 -*-
import inspect
import time
import unittest

import torch

from mnn import models, snn
from mnn.snn.base.currents import GaussianCurrentGenerator as LegacyGaussianCurrentGenerator
from mnn.snn.base.currents import PoissonSpikeGenerator as LegacyPoissonSpikeGenerator
from mnn.snn.base.monitors import SpikeMonitor
from mnn.snn.base.neurons import LIFNeurons
from mnn.snn.base.probes import NeuronProbe
from mnn.snn.base.currents_torch import GaussianCurrentSource, PoissonSpikeSource
from mnn.snn.base.functional_torch import pregenerate_gaussian_current, sample_shape
from mnn.snn.base.monitors_torch import SpikeMonitorTorch
from mnn.snn.base.neurons_torch import LifNeurons
from mnn.snn.base.probes_torch import NeuronProbeTorch
from mnn.snn.functional import sample_poisson_spike as legacy_sample_poisson_spike
from mnn.snn.functional import sparse_spike_train_statistics as legacy_sparse_spike_train_statistics
from mnn.snn.functional_torch import MomentSnnValidator, sample_poisson_spikes
from mnn.snn.functional_torch import sparse_spike_train_statistics as sparse_spike_train_statistics_torch
from mnn.snn.mnn_to_snn_torch import MomentMlpToSnn, SpikeMomentMlpToSnn, convert_moment_parameters


class TorchSnnApiTest(unittest.TestCase):
    def test_exports_new_snn_api(self):
        self.assertIs(snn.LifNeurons, LifNeurons)
        self.assertIs(snn.SpikeMonitorTorch, SpikeMonitorTorch)
        self.assertIs(snn.GaussianCurrentSource, GaussianCurrentSource)
        self.assertIs(snn.PoissonSpikeSource, PoissonSpikeSource)
        self.assertIs(snn.MomentMlpToSnn, MomentMlpToSnn)

    def test_new_snn_sources_do_not_use_numpy_or_legacy_core(self):
        forbidden = ("import numpy", "import scipy", ".numpy()", "from_numpy", ".data", "mnn_pytorch", "mnn_utils", "fast_dawson")
        modules = [
            snn.base.functional_torch,
            snn.base.currents_torch,
            snn.base.neurons_torch,
            snn.base.monitors_torch,
            snn.base.probes_torch,
            snn.mnn_to_snn_torch,
            snn.functional_torch,
        ]
        for module in modules:
            source = inspect.getsource(module)
            for token in forbidden:
                self.assertNotIn(token, source)

    def test_sample_shape_matches_legacy_sample_size(self):
        self.assertEqual(sample_shape(3, None), [1, 3])
        self.assertEqual(sample_shape(3, 4), (4, 3))
        self.assertEqual(sample_shape((2, 3), 4), [4, 2, 3])

    def test_top_level_functional_torch_exports(self):
        self.assertIs(snn.functional_torch.MomentSnnValidator, MomentSnnValidator)
        self.assertIs(snn.sample_poisson_spikes, sample_poisson_spikes)

    def test_top_level_poisson_sample_matches_legacy_shape(self):
        freqs = torch.full((3,), 0.2)
        torch.manual_seed(11)
        legacy = legacy_sample_poisson_spike(freqs, 0.1, 3, 5, dtype=torch.float64)
        torch.manual_seed(11)
        new = sample_poisson_spikes(freqs, 0.1, 3, 5, dtype=torch.float64)
        self.assertEqual(new.shape, legacy.shape)
        self.assertEqual(new.dtype, torch.float64)

    def test_top_level_sparse_statistics_matches_legacy(self):
        dense = torch.tensor(
            [
                [[1, 0, 1], [0, 1, 0]],
                [[0, 1, 0], [1, 0, 1]],
                [[1, 1, 0], [0, 0, 1]],
            ],
            dtype=torch.float32,
        )
        sparse = dense.to_sparse()
        expected = legacy_sparse_spike_train_statistics(sparse, 0.3)
        actual = sparse_spike_train_statistics_torch(sparse, 0.3)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_condition_modifier_uses_torch_correlation(self):
        mean = torch.ones(2, 3)
        covariance = torch.eye(3).expand(2, 3, 3).clone()
        output_mean, output_covariance = MomentSnnValidator.modify_value_by_condition((mean, covariance), "corr_only")
        torch.testing.assert_close(output_mean, torch.zeros_like(mean))
        torch.testing.assert_close(torch.diagonal(output_covariance, dim1=-1, dim2=-2), torch.ones(2, 3))

    def test_independent_gaussian_current_shape_dtype_device(self):
        torch.manual_seed(1)
        mean = torch.tensor([0.5, 1.0], dtype=torch.float64)
        std = torch.tensor([0.2, 0.3], dtype=torch.float64)
        current = GaussianCurrentSource((3, 2), mean, std, dt=0.1, pregenerate=True, num_steps=5)
        sample = current()
        self.assertEqual(sample.shape, (3, 2))
        self.assertEqual(sample.dtype, torch.float64)
        self.assertEqual(sample.device, mean.device)

    def test_correlated_gaussian_current_uses_torch_device(self):
        mean = torch.zeros(2, dtype=torch.float64)
        std = torch.ones(2, dtype=torch.float64)
        rho = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
        sample = pregenerate_gaussian_current((4, 2), 6, mean, std, rho)
        self.assertEqual(sample.shape, (6, 4, 2))
        self.assertEqual(sample.dtype, mean.dtype)
        self.assertEqual(sample.device, mean.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_correlated_gaussian_current_stays_on_cuda(self):
        mean = torch.zeros(2, dtype=torch.float64, device="cuda")
        std = torch.ones(2, dtype=torch.float64, device="cuda")
        rho = torch.eye(2, dtype=torch.float64, device="cuda")
        current = GaussianCurrentSource((3, 2), mean, std, rho, dt=0.1, pregenerate=True, num_steps=4)
        self.assertEqual(current().device.type, "cuda")

    def test_poisson_source_shape_matches_legacy(self):
        freqs = torch.full((2,), 0.4)
        torch.manual_seed(3)
        legacy = LegacyPoissonSpikeGenerator((4, 2), freqs, dt=0.1)
        torch.manual_seed(3)
        new = PoissonSpikeSource((4, 2), freqs, dt=0.1)
        self.assertEqual(new().shape, legacy().shape)

    def test_lif_neurons_matches_legacy_single_step(self):
        current = torch.tensor([30.0, 0.0])
        legacy = LIFNeurons(2)
        new = LifNeurons(2)
        torch.testing.assert_close(new(current), legacy(current))
        new.reset()
        legacy.reset()
        torch.testing.assert_close(new.voltage, legacy.V)

    def test_monitor_and_probe_behavior(self):
        spikes = torch.tensor([1, 0, 1], dtype=torch.float)
        legacy_monitor = SpikeMonitor(3)
        new_monitor = SpikeMonitorTorch(3)
        legacy_monitor(spikes)
        new_monitor(spikes)
        legacy_count, legacy_duration = legacy_monitor.spike_count()
        new_count, new_duration = new_monitor.spike_count()
        torch.testing.assert_close(new_count, legacy_count)
        self.assertEqual(new_duration, legacy_duration)

        legacy_probe = NeuronProbe("V", dt=1.0)
        new_probe = NeuronProbeTorch("voltage", dt=1.0)
        legacy_neuron = LIFNeurons(2)
        new_neuron = LifNeurons(2)
        legacy_probe(legacy_neuron)
        new_probe(new_neuron)
        self.assertIn("V", legacy_probe.get_data())
        self.assertIn("voltage", new_probe.get_data())

    def test_convert_moment_parameters_and_snn_forward(self):
        torch.manual_seed(5)
        model = MomentMlpToSnn([3, 4], num_classes=2).eval()
        params = convert_moment_parameters(model)
        self.assertIn("fc0.weight", params)
        self.assertIn("fc1.weight", params)
        model.mnn_to_snn(dt=0.1, batch_size=2)
        output = model(torch.randn(2, 3))
        self.assertEqual(output.shape, torch.Size([2, 4]))

    def test_spike_moment_to_snn_forward(self):
        model = SpikeMomentMlpToSnn([3], num_classes=2).eval()
        model.mnn_to_snn(dt=0.1, batch_size=2)
        output = model(torch.randn(2, 3))
        self.assertEqual(output.shape, torch.Size([2, 2]))

    def test_cpu_benchmark_snn_current(self):
        mean = torch.zeros(16)
        std = torch.ones(16)
        legacy = LegacyGaussianCurrentGenerator((8, 16), mean, std, dt=0.1)
        new = GaussianCurrentSource((8, 16), mean, std, dt=0.1)
        start = time.perf_counter()
        for _ in range(10):
            legacy()
        legacy_elapsed = (time.perf_counter() - start) / 10
        start = time.perf_counter()
        for _ in range(10):
            new()
        new_elapsed = (time.perf_counter() - start) / 10
        print(f"legacy cpu GaussianCurrentGenerator: {legacy_elapsed:.6f}s/call")
        print(f"torch cpu GaussianCurrentSource: {new_elapsed:.6f}s/call")


if __name__ == "__main__":
    unittest.main()
