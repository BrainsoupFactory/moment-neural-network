# -*- coding: utf-8 -*-
import time
import unittest
import warnings

import numpy as np
import torch

from mnn.mnn_core.fast_dawson import Dawson1 as VanillaDawsonFirstOrder
from mnn.mnn_core.fast_dawson import Dawson2 as VanillaDawsonSecondOrder
from mnn.mnn_core.mnn_pytorch import mnn_activate_no_rho as vanilla_activation_without_correlation
from mnn.mnn_core.mnn_pytorch import mnn_activate_trio as vanilla_activation_with_correlation
from mnn.mnn_core.mnn_utils import Mnn_Core_Func as VanillaCore
from mnn.mnn_core.torch_activation import mnn_activation_with_correlation
from mnn.mnn_core.torch_activation import mnn_activation_without_correlation
from mnn.mnn_core.torch_core import MNNCore
from mnn.mnn_core.torch_dawson import DawsonFirstOrder
from mnn.mnn_core.torch_dawson import DawsonSecondOrder
from mnn.mnn_core.nn.activation_torch import MomentActivation


def _as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _max_abs_and_rel_error(expected, actual):
    expected = np.asarray(_as_numpy(expected), dtype=np.float64)
    actual = np.asarray(_as_numpy(actual), dtype=np.float64)
    abs_error = np.max(np.abs(expected - actual))
    rel_error = np.max(np.abs(expected - actual) / np.maximum(np.abs(expected), 1.0))
    return abs_error, rel_error


def _assert_close(test_case, expected, actual, *, atol, rtol):
    abs_error, rel_error = _max_abs_and_rel_error(expected, actual)
    if abs_error <= atol or rel_error <= rtol:
        return
    test_case.fail(f"max abs error {abs_error} > {atol} and max relative error {rel_error} > {rtol}")


def _time_call(func, *args, repeats=20):
    start = time.perf_counter()
    for _ in range(repeats):
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / repeats, result


class TorchDawsonAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.vanilla_first = VanillaDawsonFirstOrder()
            cls.vanilla_second = VanillaDawsonSecondOrder()
        cls.torch_first = DawsonFirstOrder()
        cls.torch_second = DawsonSecondOrder()

    def test_first_order_dawson_matches_vanilla(self):
        x = torch.linspace(-8.0, 8.0, 129, dtype=torch.float64)
        x_np = x.numpy()

        _assert_close(self, self.vanilla_first.dawson1(x_np), self.torch_first.evaluate(x), atol=0.0, rtol=0.0)
        _assert_close(self, self.vanilla_first.int_fast(x_np), self.torch_first.integral(x), atol=3e9, rtol=1e-12)

    def test_second_order_dawson_matches_vanilla(self):
        x = torch.linspace(-8.0, 8.0, 129, dtype=torch.float64)
        x_np = x.numpy()

        _assert_close(self, self.vanilla_second.dawson2(x_np), self.torch_second.evaluate(x), atol=1e36, rtol=1e-11)
        _assert_close(self, self.vanilla_second.int_fast(x_np), self.torch_second.integral(x), atol=1e35, rtol=1e-11)


class TorchCoreAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.vanilla_core = VanillaCore()
        cls.torch_core = MNNCore()

    def setUp(self):
        self.mean = torch.tensor(
            [[0.2, 0.5, 0.8, 1.2, 2.0], [0.5, 1.1, 1.6, 1.8, 2.5]],
            dtype=torch.float64,
        )
        self.std = torch.tensor(
            [[0.0, 0.1, 0.05, 0.2, 0.3], [0.1, 0.2, 0.3, 0.0, 0.4]],
            dtype=torch.float64,
        )
        self.mean_np = self.mean.numpy()
        self.std_np = self.std.numpy()

    def test_compute_bounds_matches_vanilla(self):
        expected = self.vanilla_core.compute_bound(self.mean_np, self.std_np)
        actual = self.torch_core.compute_bounds(self.mean, self.std)

        _assert_close(self, expected[0], actual[0], atol=1e-12, rtol=1e-12)
        _assert_close(self, expected[1], actual[1], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(expected[2], _as_numpy(actual[2]))

    def test_forward_methods_match_vanilla(self):
        expected_mean = self.vanilla_core.forward_fast_mean(self.mean_np, self.std_np)
        actual_mean = self.torch_core.forward_mean(self.mean, self.std)
        expected_std = self.vanilla_core.forward_fast_std(self.mean_np, self.std_np, expected_mean)
        actual_std = self.torch_core.forward_std(self.mean, self.std, actual_mean)
        expected_chi = self.vanilla_core.forward_fast_chi(self.mean_np, self.std_np, expected_mean, expected_std)
        actual_chi = self.torch_core.forward_chi(self.mean, self.std, actual_mean, actual_std)
        actual_forward = self.torch_core.forward(self.mean, self.std)

        _assert_close(self, expected_mean, actual_mean, atol=1e-13, rtol=1e-12)
        _assert_close(self, expected_std, actual_std, atol=1e-12, rtol=1e-12)
        _assert_close(self, expected_chi, actual_chi, atol=1e-11, rtol=1e-11)
        _assert_close(self, expected_mean, actual_forward[0], atol=1e-13, rtol=1e-12)
        _assert_close(self, expected_std, actual_forward[1], atol=1e-12, rtol=1e-12)
        _assert_close(self, expected_chi, actual_forward[2], atol=1e-11, rtol=1e-11)

    def test_backward_methods_match_vanilla(self):
        expected_mean, expected_std, expected_chi = self.vanilla_core.fast_forward(self.mean_np, self.std_np)
        actual_mean, actual_std, actual_chi = self.torch_core.forward(self.mean, self.std)

        expected_mean_backward = self.vanilla_core.backward_fast_mean(self.mean_np, self.std_np, expected_mean)
        actual_mean_backward = self.torch_core.backward_mean(self.mean, self.std, actual_mean)
        expected_std_backward = self.vanilla_core.backward_fast_std(self.mean_np, self.std_np, expected_mean, expected_std)
        actual_std_backward = self.torch_core.backward_std(self.mean, self.std, actual_mean, actual_std)
        expected_chi_backward = self.vanilla_core.backward_fast_chi(self.mean_np, self.std_np, expected_mean, expected_chi)
        actual_chi_backward = self.torch_core.backward_chi(self.mean, self.std, actual_mean, actual_chi)
        expected_backward = self.vanilla_core.fast_backward(self.mean_np, self.std_np, expected_mean, expected_std, expected_chi)
        actual_backward = self.torch_core.backward(self.mean, self.std, actual_mean, actual_std, actual_chi)

        for expected, actual in zip(expected_mean_backward, actual_mean_backward):
            _assert_close(self, expected, actual, atol=1e-12, rtol=1e-11)
        for expected, actual in zip(expected_std_backward, actual_std_backward):
            _assert_close(self, expected, actual, atol=1e-11, rtol=1e-11)
        for expected, actual in zip(expected_chi_backward, actual_chi_backward):
            _assert_close(self, expected, actual, atol=1e-9, rtol=1e-10)
        for expected, actual in zip(expected_backward, actual_backward):
            _assert_close(self, expected, actual, atol=1e-9, rtol=1e-10)


class TorchActivationAccuracyTest(unittest.TestCase):
    def test_activation_without_correlation_matches_vanilla_forward_and_backward(self):
        vanilla_mean = torch.tensor([[0.5, 1.2, 2.0], [0.8, 1.6, 2.5]], dtype=torch.float64, requires_grad=True)
        vanilla_std = torch.tensor([[0.1, 0.2, 0.3], [0.05, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
        torch_mean = vanilla_mean.detach().clone().requires_grad_()
        torch_std = vanilla_std.detach().clone().requires_grad_()

        expected = vanilla_activation_without_correlation(vanilla_mean, vanilla_std)
        actual = mnn_activation_without_correlation(torch_mean, torch_std)
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-12, rtol=1e-12)

        sum(tensor.sum() for tensor in expected).backward()
        sum(tensor.sum() for tensor in actual).backward()

        _assert_close(self, vanilla_mean.grad, torch_mean.grad, atol=1e-11, rtol=1e-11)
        _assert_close(self, vanilla_std.grad, torch_std.grad, atol=1e-11, rtol=1e-11)

    def test_activation_with_correlation_matches_vanilla_forward_and_backward(self):
        vanilla_mean = torch.tensor([[0.5, 1.2, 2.0]], dtype=torch.float64, requires_grad=True)
        vanilla_std = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
        vanilla_correlation = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_()
        torch_mean = vanilla_mean.detach().clone().requires_grad_()
        torch_std = vanilla_std.detach().clone().requires_grad_()
        torch_correlation = vanilla_correlation.detach().clone().requires_grad_()

        expected = vanilla_activation_with_correlation(vanilla_mean, vanilla_std, vanilla_correlation)
        actual = mnn_activation_with_correlation(torch_mean, torch_std, torch_correlation)
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-12, rtol=1e-12)

        sum(tensor.sum() for tensor in expected).backward()
        sum(tensor.sum() for tensor in actual).backward()

        _assert_close(self, vanilla_mean.grad, torch_mean.grad, atol=1e-11, rtol=1e-11)
        _assert_close(self, vanilla_std.grad, torch_std.grad, atol=1e-11, rtol=1e-11)
        _assert_close(self, vanilla_correlation.grad, torch_correlation.grad, atol=1e-12, rtol=1e-12)

    def test_activation_float32_cutoff_region_uses_stable_internal_precision(self):
        sqrt_conductance = 0.05 ** 0.5
        mean = torch.tensor([[1.0 - 9.0 * sqrt_conductance * 0.1, 0.9]], dtype=torch.float32, requires_grad=True)
        std = torch.tensor([[0.1, 0.2]], dtype=torch.float32, requires_grad=True)
        correlation = torch.eye(2, dtype=torch.float32).unsqueeze(0).requires_grad_()

        outputs = mnn_activation_with_correlation(mean, std, correlation)
        for tensor in outputs:
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())

        sum(tensor.sum() for tensor in outputs).backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(std.grad).all())
        self.assertTrue(torch.isfinite(correlation.grad).all())


class TorchActivationBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.vanilla_first = VanillaDawsonFirstOrder()
            cls.vanilla_second = VanillaDawsonSecondOrder()
            cls.vanilla_core = VanillaCore()
        cls.torch_first = DawsonFirstOrder()
        cls.torch_second = DawsonSecondOrder()
        cls.torch_core = MNNCore()

    def _print_timing(self, label, vanilla_time, torch_time, torch_device):
        print(f"\nvanilla cpu {label}: {vanilla_time:.6f}s/call")
        print(f"torch {torch_device} {label}: {torch_time:.6f}s/call")

    def test_cpu_torch_dawson_timing_against_vanilla_cpu(self):
        x = torch.linspace(-8.0, 8.0, 2048, dtype=torch.float64)
        x_np = x.numpy()
        cases = (
            ("dawson_first.evaluate", self.vanilla_first.dawson1, self.torch_first.evaluate),
            ("dawson_first.integral", self.vanilla_first.int_fast, self.torch_first.integral),
            ("dawson_second.evaluate", self.vanilla_second.dawson2, self.torch_second.evaluate),
            ("dawson_second.integral", self.vanilla_second.int_fast, self.torch_second.integral),
        )

        for label, vanilla_func, torch_func in cases:
            vanilla_time, vanilla_output = _time_call(vanilla_func, x_np, repeats=10)
            torch_time, torch_output = _time_call(torch_func, x, repeats=10)
            _assert_close(self, vanilla_output, torch_output, atol=1e36, rtol=1e-10)
            self.assertGreater(vanilla_time, 0.0)
            self.assertGreater(torch_time, 0.0)
            self._print_timing(label, vanilla_time, torch_time, "cpu")

    def test_cpu_torch_core_timing_against_vanilla_cpu(self):
        mean = torch.linspace(0.2, 2.5, 2048, dtype=torch.float64)
        std = torch.linspace(0.05, 0.4, 2048, dtype=torch.float64)
        mean_np = mean.numpy()
        std_np = std.numpy()
        vanilla_forward_time, vanilla_forward = _time_call(self.vanilla_core.fast_forward, mean_np, std_np, repeats=10)
        torch_forward_time, torch_forward = _time_call(self.torch_core.forward, mean, std, repeats=10)
        for vanilla_output, torch_output in zip(vanilla_forward, torch_forward):
            _assert_close(self, vanilla_output, torch_output, atol=1e-9, rtol=1e-10)
        self._print_timing("core.forward", vanilla_forward_time, torch_forward_time, "cpu")

        vanilla_backward_time, vanilla_backward = _time_call(
            self.vanilla_core.fast_backward,
            mean_np,
            std_np,
            *vanilla_forward,
            repeats=10,
        )
        torch_backward_time, torch_backward = _time_call(
            self.torch_core.backward,
            mean,
            std,
            *torch_forward,
            repeats=10,
        )
        for vanilla_output, torch_output in zip(vanilla_backward, torch_backward):
            _assert_close(self, vanilla_output, torch_output, atol=1e-8, rtol=1e-9)
        self.assertGreater(vanilla_backward_time, 0.0)
        self.assertGreater(torch_backward_time, 0.0)
        self._print_timing("core.backward", vanilla_backward_time, torch_backward_time, "cpu")

    def test_cpu_torch_activation_timing_against_vanilla_cpu(self):
        mean = torch.linspace(0.2, 2.5, 2048, dtype=torch.float64)
        std = torch.linspace(0.05, 0.4, 2048, dtype=torch.float64)

        vanilla_time, vanilla_output = _time_call(vanilla_activation_without_correlation, mean, std, repeats=10)
        torch_time, torch_output = _time_call(mnn_activation_without_correlation, mean, std, repeats=10)

        for expected_tensor, actual_tensor in zip(vanilla_output, torch_output):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-10)
        self.assertGreater(vanilla_time, 0.0)
        self.assertGreater(torch_time, 0.0)
        self._print_timing("activation_without_correlation", vanilla_time, torch_time, "cpu")

        mean_trio = mean.reshape(1, -1)
        std_trio = std.reshape(1, -1)
        correlation = torch.eye(mean_trio.size(-1), dtype=torch.float64).unsqueeze(0)
        vanilla_time, vanilla_output = _time_call(
            vanilla_activation_with_correlation,
            mean_trio,
            std_trio,
            correlation,
            repeats=10,
        )
        torch_time, torch_output = _time_call(
            mnn_activation_with_correlation,
            mean_trio,
            std_trio,
            correlation,
            repeats=10,
        )
        for expected_tensor, actual_tensor in zip(vanilla_output, torch_output):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-10)
        self.assertGreater(vanilla_time, 0.0)
        self.assertGreater(torch_time, 0.0)
        self._print_timing("activation_with_correlation", vanilla_time, torch_time, "cpu")

        covariance = torch.eye(mean_trio.size(-1), dtype=torch.float64).unsqueeze(0) * std_trio.unsqueeze(-1) * std_trio.unsqueeze(-2)
        moment_activation = MomentActivation()
        torch_nn_time, torch_nn_output = _time_call(moment_activation, mean_trio, covariance, repeats=10)
        expected_covariance = torch_output[2] * torch_output[1].unsqueeze(-1) * torch_output[1].unsqueeze(-2)
        _assert_close(self, torch_output[0], torch_nn_output[0], atol=1e-6, rtol=1e-6)
        _assert_close(self, expected_covariance, torch_nn_output[1], atol=1e-6, rtol=1e-6)
        self.assertGreater(torch_nn_time, 0.0)
        print(f"torch cpu nn.MomentActivation: {torch_nn_time:.6f}s/call")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_torch_dawson_timing_against_vanilla_cpu(self):
        x_cpu = torch.linspace(-8.0, 8.0, 2048, dtype=torch.float64)
        x_gpu = x_cpu.to("cuda")
        x_np = x_cpu.numpy()
        cases = (
            ("dawson_first.evaluate", self.vanilla_first.dawson1, self.torch_first.evaluate),
            ("dawson_first.integral", self.vanilla_first.int_fast, self.torch_first.integral),
            ("dawson_second.evaluate", self.vanilla_second.dawson2, self.torch_second.evaluate),
            ("dawson_second.integral", self.vanilla_second.int_fast, self.torch_second.integral),
        )

        for label, vanilla_func, torch_func in cases:
            vanilla_time, vanilla_output = _time_call(vanilla_func, x_np, repeats=10)
            torch_time, torch_output = _time_call(torch_func, x_gpu, repeats=10)
            _assert_close(self, vanilla_output, torch_output, atol=1e36, rtol=1e-10)
            self.assertGreater(vanilla_time, 0.0)
            self.assertGreater(torch_time, 0.0)
            self._print_timing(label, vanilla_time, torch_time, "gpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_torch_core_timing_against_vanilla_cpu(self):
        mean_cpu = torch.linspace(0.2, 2.5, 2048, dtype=torch.float64)
        std_cpu = torch.linspace(0.05, 0.4, 2048, dtype=torch.float64)
        mean_gpu = mean_cpu.to("cuda")
        std_gpu = std_cpu.to("cuda")
        mean_np = mean_cpu.numpy()
        std_np = std_cpu.numpy()
        vanilla_forward_time, vanilla_forward = _time_call(self.vanilla_core.fast_forward, mean_np, std_np, repeats=10)
        torch_forward_time, torch_forward = _time_call(self.torch_core.forward, mean_gpu, std_gpu, repeats=10)
        for vanilla_output, torch_output in zip(vanilla_forward, torch_forward):
            _assert_close(self, vanilla_output, torch_output, atol=1e-9, rtol=1e-10)
        self._print_timing("core.forward", vanilla_forward_time, torch_forward_time, "gpu")

        vanilla_backward_time, vanilla_backward = _time_call(
            self.vanilla_core.fast_backward,
            mean_np,
            std_np,
            *vanilla_forward,
            repeats=10,
        )
        torch_backward_time, torch_backward = _time_call(
            self.torch_core.backward,
            mean_gpu,
            std_gpu,
            *torch_forward,
            repeats=10,
        )
        for vanilla_output, torch_output in zip(vanilla_backward, torch_backward):
            _assert_close(self, vanilla_output, torch_output, atol=1e-8, rtol=1e-9)
        self.assertGreater(vanilla_backward_time, 0.0)
        self.assertGreater(torch_backward_time, 0.0)
        self._print_timing("core.backward", vanilla_backward_time, torch_backward_time, "gpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_torch_activation_timing_against_vanilla_cpu(self):
        mean_cpu = torch.linspace(0.2, 2.5, 2048, dtype=torch.float64)
        std_cpu = torch.linspace(0.05, 0.4, 2048, dtype=torch.float64)
        mean_gpu = mean_cpu.to("cuda")
        std_gpu = std_cpu.to("cuda")

        vanilla_time, vanilla_output = _time_call(vanilla_activation_without_correlation, mean_cpu, std_cpu, repeats=10)
        torch_gpu_time, torch_gpu_output = _time_call(mnn_activation_without_correlation, mean_gpu, std_gpu, repeats=10)

        for expected_tensor, actual_tensor in zip(vanilla_output, torch_gpu_output):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-10)
        self.assertGreater(vanilla_time, 0.0)
        self.assertGreater(torch_gpu_time, 0.0)
        self._print_timing("activation_without_correlation", vanilla_time, torch_gpu_time, "gpu")

        mean_cpu = mean_cpu.reshape(1, -1)
        std_cpu = std_cpu.reshape(1, -1)
        mean_gpu = mean_cpu.to("cuda")
        std_gpu = std_cpu.to("cuda")
        correlation_cpu = torch.eye(mean_cpu.size(-1), dtype=torch.float64).unsqueeze(0)
        correlation_gpu = correlation_cpu.to("cuda")
        vanilla_time, vanilla_output = _time_call(
            vanilla_activation_with_correlation,
            mean_cpu,
            std_cpu,
            correlation_cpu,
            repeats=10,
        )
        torch_gpu_time, torch_gpu_output = _time_call(
            mnn_activation_with_correlation,
            mean_gpu,
            std_gpu,
            correlation_gpu,
            repeats=10,
        )
        for expected_tensor, actual_tensor in zip(vanilla_output, torch_gpu_output):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-10)
        self.assertGreater(vanilla_time, 0.0)
        self.assertGreater(torch_gpu_time, 0.0)
        self._print_timing("activation_with_correlation", vanilla_time, torch_gpu_time, "gpu")

        covariance_gpu = torch.eye(mean_gpu.size(-1), dtype=torch.float64, device="cuda").unsqueeze(0)
        covariance_gpu = covariance_gpu * std_gpu.unsqueeze(-1) * std_gpu.unsqueeze(-2)
        moment_activation = MomentActivation()
        torch_nn_time, torch_nn_output = _time_call(moment_activation, mean_gpu, covariance_gpu, repeats=10)
        expected_covariance = torch_gpu_output[2] * torch_gpu_output[1].unsqueeze(-1) * torch_gpu_output[1].unsqueeze(-2)
        _assert_close(self, torch_gpu_output[0], torch_nn_output[0], atol=1e-6, rtol=1e-6)
        _assert_close(self, expected_covariance, torch_nn_output[1], atol=1e-6, rtol=1e-6)
        self.assertGreater(torch_nn_time, 0.0)
        print(f"torch gpu nn.MomentActivation: {torch_nn_time:.6f}s/call")


if __name__ == "__main__":
    unittest.main(verbosity=2)
