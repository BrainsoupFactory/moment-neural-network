# -*- coding: utf-8 -*-
import inspect
import unittest

import torch

from mnn.mnn_core import nn
from mnn.mnn_core.nn.activation import ConstantCurrentActivation as LegacyConstantCurrentActivation
from mnn.mnn_core.nn.activation import OriginMnnActivation
from mnn.mnn_core.nn.batch_norm import BatchNorm1dDuo, BatchNorm1dNoRho
from mnn.mnn_core.nn.criterion import CrossEntropyOnMean, LabelSmoothing
from mnn.mnn_core.nn.custom_batch_norm import CustomBatchNorm1D
from mnn.mnn_core.nn.ensemble import EnsembleLinearDuo, EnsembleLinearNoRho
from mnn.mnn_core.nn.linear import LinearDuo, LinearNoRho
from mnn.mnn_core.nn.pooling import MnnPooling
from mnn.mnn_core.nn.activation_torch import ConstantCurrentActivationTorch, MomentActivation
from mnn.mnn_core.nn.batch_norm_torch import MomentBatchNorm1d, MomentBatchNorm1dNoCorrelation
from mnn.mnn_core.nn.conv_torch import MomentConv2d
from mnn.mnn_core.nn.criterion_torch import CrossEntropyOnMeanTorch, LabelSmoothingTorch
from mnn.mnn_core.nn.custom_batch_norm_torch import CustomMomentBatchNorm1d
from mnn.mnn_core.nn.ensemble_torch import MomentBlock, MomentBlockNoCorrelation
from mnn.mnn_core.nn.linear_torch import MomentLinear, MomentLinearNoCorrelation
from mnn.mnn_core.nn.pooling_torch import MomentPooling


def _assert_close(test_case, expected, actual, *, atol=1e-10, rtol=1e-10):
    torch.testing.assert_close(
        actual.detach().cpu() if isinstance(actual, torch.Tensor) else actual,
        expected.detach().cpu() if isinstance(expected, torch.Tensor) else expected,
        atol=atol,
        rtol=rtol,
    )


class TorchNNImportTest(unittest.TestCase):
    def test_new_api_exports_from_nn_package(self):
        self.assertIs(nn.MomentActivation, MomentActivation)
        self.assertIs(nn.MomentLinear, MomentLinear)
        self.assertIs(nn.MomentBatchNorm1d, MomentBatchNorm1d)
        self.assertIs(nn.CustomMomentBatchNorm1d, CustomMomentBatchNorm1d)
        self.assertIs(nn.MomentBlock, MomentBlock)
        self.assertIs(nn.MomentPooling, MomentPooling)

    def test_new_modules_do_not_import_legacy_core_modules(self):
        modules = [
            nn.activation_torch,
            nn.functional_torch,
            nn.linear_torch,
            nn.batch_norm_torch,
            nn.custom_batch_norm_torch,
            nn.ensemble_torch,
            nn.pooling_torch,
            nn.criterion_torch,
            nn.conv_torch,
        ]
        forbidden = ("mnn_pytorch", "mnn_utils", "fast_dawson")
        for module in modules:
            source = inspect.getsource(module)
            for token in forbidden:
                self.assertNotIn(token, source)


class TorchNNAccuracyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.mean = torch.tensor([[0.5, 1.2, 2.0], [0.8, 1.6, 2.5]], dtype=torch.float64, requires_grad=True)
        covariance = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1) * 0.04
        covariance[:, 0, 1] = 0.005
        covariance[:, 1, 0] = 0.005
        self.covariance = covariance.requires_grad_()
        self.std = torch.tensor([[0.1, 0.2, 0.3], [0.05, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)

    def test_activation_matches_legacy_forward_and_backward(self):
        legacy_mean = self.mean.detach().clone().requires_grad_()
        legacy_covariance = self.covariance.detach().clone().requires_grad_()
        torch_mean = self.mean.detach().clone().requires_grad_()
        torch_covariance = self.covariance.detach().clone().requires_grad_()

        legacy_output = OriginMnnActivation()(legacy_mean, legacy_covariance)
        torch_output = MomentActivation()(torch_mean, torch_covariance)

        for expected, actual in zip(legacy_output, torch_output):
            _assert_close(self, expected, actual, atol=1e-10, rtol=1e-10)

        sum(tensor.sum() for tensor in legacy_output).backward()
        sum(tensor.sum() for tensor in torch_output).backward()
        _assert_close(self, legacy_mean.grad, torch_mean.grad, atol=1e-10, rtol=1e-10)
        _assert_close(self, legacy_covariance.grad, torch_covariance.grad, atol=1e-10, rtol=1e-10)

    def test_constant_current_activation_matches_legacy(self):
        current = torch.tensor([0.5, 1.2, 2.0], dtype=torch.float64, requires_grad=True)
        current_torch = current.detach().clone().requires_grad_()
        legacy = LegacyConstantCurrentActivation()
        new = ConstantCurrentActivationTorch()

        expected = legacy(current)
        actual = new(current_torch)
        _assert_close(self, expected, actual)
        expected.sum().backward()
        actual.sum().backward()
        _assert_close(self, current.grad, current_torch.grad)

    def test_linear_matches_legacy(self):
        legacy = LinearDuo(3, 2, bias=True, bias_var=True).double()
        new = MomentLinear(3, 2, bias=True, bias_variance=True).double()
        with torch.no_grad():
            new.weight.copy_(legacy.weight)
            new.bias.copy_(legacy.bias)
            new.bias_variance.copy_(legacy.bias_var)

        expected = legacy(self.mean.detach(), self.covariance.detach())
        actual = new(self.mean.detach(), self.covariance.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_linear_no_correlation_matches_legacy(self):
        legacy = LinearNoRho(3, 2, bias=True, bias_var=True).double()
        new = MomentLinearNoCorrelation(3, 2, bias=True, bias_variance=True).double()
        with torch.no_grad():
            new.weight.copy_(legacy.weight)
            new.bias.copy_(legacy.bias)
            new.bias_variance.copy_(legacy.bias_var)

        expected = legacy(self.mean.detach(), self.std.detach())
        actual = new(self.mean.detach(), self.std.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_batch_norm_matches_legacy_eval(self):
        legacy = BatchNorm1dDuo(3, bias_var=True).double().eval()
        new = MomentBatchNorm1d(3, bias_variance=True).double().eval()
        with torch.no_grad():
            new.batch_norm_mean.load_state_dict(legacy.bn_mean.state_dict())
            new.bias_variance.copy_(legacy.bias_var)

        expected = legacy(self.mean.detach(), self.covariance.detach())
        actual = new(self.mean.detach(), self.covariance.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_batch_norm_no_correlation_matches_legacy_eval(self):
        legacy = BatchNorm1dNoRho(3, bias_std=True).double().eval()
        new = MomentBatchNorm1dNoCorrelation(3, bias_std=True).double().eval()
        with torch.no_grad():
            new.batch_norm_mean.load_state_dict(legacy.bn_mean.state_dict())
            new.bias_std.copy_(legacy.bias_std)

        expected = legacy(self.mean.detach(), self.std.detach())
        actual = new(self.mean.detach(), self.std.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_custom_batch_norm_matches_legacy_eval(self):
        legacy = CustomBatchNorm1D(3, bias_var=True).double().eval()
        new = CustomMomentBatchNorm1d(3, bias_variance=True).double().eval()
        with torch.no_grad():
            new.weight.copy_(legacy.weight)
            new.bias.copy_(legacy.bias)
            new.bias_variance.copy_(legacy.bias_var)
            new.running_mean.copy_(legacy.running_mean)
            new.running_variance.copy_(legacy.running_var)

        expected = legacy(self.mean.detach(), self.covariance.detach())
        actual = new(self.mean.detach(), self.covariance.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_pooling_matches_legacy(self):
        x = torch.randn(2, 3, 1, 5, dtype=torch.float64)
        legacy = MnnPooling(input_dim=3, mask_cov=True)
        new = MomentPooling(input_dim=3, mask_covariance=True)

        expected = legacy(x)
        actual = new(x)
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor)

    def test_criterion_matches_legacy(self):
        target = torch.tensor([1, 2])
        logits = torch.randn(2, 3, dtype=torch.float64)
        _assert_close(self, LabelSmoothing(3, 0.1)(target), LabelSmoothingTorch(3, 0.1)(target))
        _assert_close(self, CrossEntropyOnMean()(logits, target), CrossEntropyOnMeanTorch()(logits, target))

    def test_ensemble_matches_legacy_eval(self):
        legacy = EnsembleLinearDuo(3, 2, bn_bias_var=True).double().eval()
        new = MomentBlock(3, 2, bias_variance=True).double().eval()
        with torch.no_grad():
            new.linear.weight.copy_(legacy.linear.weight)
            new.norm.weight.copy_(legacy.bn.weight)
            new.norm.bias.copy_(legacy.bn.bias)
            new.norm.bias_variance.copy_(legacy.bn.bias_var)
            new.norm.running_mean.copy_(legacy.bn.running_mean)
            new.norm.running_variance.copy_(legacy.bn.running_var)

        expected = legacy(self.mean.detach(), self.covariance.detach())
        actual = new(self.mean.detach(), self.covariance.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-9)

    def test_ensemble_no_correlation_matches_legacy_eval(self):
        legacy = EnsembleLinearNoRho(3, 2, bn_bias_std=True).double().eval()
        new = MomentBlockNoCorrelation(3, 2, bias_std=True).double().eval()
        with torch.no_grad():
            new.linear.weight.copy_(legacy.linear.weight)
            new.norm.batch_norm_mean.load_state_dict(legacy.bn.bn_mean.state_dict())
            new.norm.bias_std.copy_(legacy.bn.bias_std)

        expected = legacy(self.mean.detach(), self.std.detach())
        actual = new(self.mean.detach(), self.std.detach())
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_close(self, expected_tensor, actual_tensor, atol=1e-9, rtol=1e-9)


class TorchNNBestPracticeTest(unittest.TestCase):
    def test_sparse_mask_is_buffer_and_uses_torch_shape(self):
        layer = MomentLinear(8, 4, sparse_degree=3)
        self.assertIn("sparse_mask", dict(layer.named_buffers()))
        self.assertEqual(layer.sparse_mask.shape, layer.weight.shape)
        self.assertEqual(layer.sparse_mask.sum(dim=-1).tolist(), [3.0, 3.0, 3.0, 3.0])

    def test_moment_conv_uses_module_list(self):
        layer = MomentConv2d(2, 3, 3, 1)
        self.assertIsInstance(layer.convolutions, torch.nn.ModuleList)
        self.assertEqual(len(list(layer.parameters())), 3)

    def test_moment_conv_forward_shapes(self):
        layer = MomentConv2d(2, 3, 3, 1).double()
        mean = torch.randn(4, 2, 5, 5, dtype=torch.float64)
        covariance = torch.randn(4, 2, 5, 5, 5, 5, dtype=torch.float64)
        output_mean, output_covariance = layer(mean, covariance)
        self.assertEqual(output_mean.shape, (4, 3, 3, 3))
        self.assertEqual(output_covariance.shape, (4, 3, 3, 3, 3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
