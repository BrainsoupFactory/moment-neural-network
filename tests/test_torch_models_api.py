# -*- coding: utf-8 -*-
import inspect
import time
import unittest

import torch

from mnn import models
from mnn.models.mlp import AnnMlp, MnnMlp, MnnMlpMeanOnly, SnnMlp
from mnn.models.mlp_torch import AnnMlpTorch, MomentMlp, MomentRateMlp, SpikeMomentMlp


def _copy_batch_norm(old, new):
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.copy_(old.bias)
        new.running_mean.copy_(old.running_mean)
        new.running_variance.copy_(old.running_var)


class TorchModelsApiTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.mean = torch.randn(5, 3, dtype=torch.float64)
        factor = torch.randn(5, 3, 3, dtype=torch.float64)
        self.covariance = torch.matmul(factor, factor.transpose(-1, -2)) + torch.eye(3, dtype=torch.float64) * 0.2

    def test_exports_new_model_api(self):
        self.assertIs(models.MomentMlp, MomentMlp)
        self.assertIs(models.SpikeMomentMlp, SpikeMomentMlp)
        self.assertIs(models.MomentRateMlp, MomentRateMlp)
        self.assertIs(models.AnnMlpTorch, AnnMlpTorch)

    def test_new_model_sources_do_not_use_legacy_nn_api(self):
        forbidden = ("LinearDuo", "OriginMnnActivation", "CustomBatchNorm1D", "mnn_core.nn.functional")
        for module in (models.mlp_torch, models.cnn_torch):
            source = inspect.getsource(module)
            for token in forbidden:
                self.assertNotIn(token, source)

    def test_moment_mlp_matches_legacy_forward(self):
        legacy = MnnMlp([3, 4], num_class=2, special_init=True).double().eval()
        new = MomentMlp([3, 4], num_classes=2, special_init=True).double().eval()
        with torch.no_grad():
            new.layers[0].weight.copy_(legacy.mlp[0].weight)
            _copy_batch_norm(legacy.mlp[1], new.layers[1])
            new.predict.weight.copy_(legacy.predict.weight)
        expected = legacy((self.mean.clone(), self.covariance.clone()))
        actual = new((self.mean.clone(), self.covariance.clone()))
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)

    def test_spike_moment_mlp_matches_legacy_forward(self):
        legacy = SnnMlp([3], num_class=2).double().eval()
        new = SpikeMomentMlp([3], num_classes=2).double().eval()
        with torch.no_grad():
            new.predict.linear.weight.copy_(legacy.predict.linear.weight)
            _copy_batch_norm(legacy.predict.bn, new.predict.norm)
        expected = legacy((self.mean.clone(), self.covariance.clone()))
        actual = new((self.mean.clone(), self.covariance.clone()))
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)

    def test_rate_mlp_matches_legacy_forward(self):
        x = torch.randn(6, 3, dtype=torch.float64)
        legacy = MnnMlpMeanOnly([3, 4], num_class=2).double().eval()
        new = MomentRateMlp([3, 4], num_classes=2).double().eval()
        with torch.no_grad():
            new.layers[0].weight.copy_(legacy.mlp[0].weight)
            if legacy.mlp[0].bias is not None:
                new.layers[0].bias.copy_(legacy.mlp[0].bias)
            new.layers[1].weight.copy_(legacy.mlp[1].weight)
            new.layers[1].bias.copy_(legacy.mlp[1].bias)
            new.layers[1].running_mean.copy_(legacy.mlp[1].running_mean)
            new.layers[1].running_var.copy_(legacy.mlp[1].running_var)
            new.predict.weight.copy_(legacy.predict.weight)
            new.predict.bias.copy_(legacy.predict.bias)
        torch.testing.assert_close(new(x), legacy(x), rtol=1e-6, atol=1e-6)

    def test_ann_mlp_torch_shape_matches_legacy(self):
        x = torch.randn(4, 3)
        legacy = AnnMlp([3, 5], num_class=2)
        new = AnnMlpTorch([3, 5], num_classes=2)
        self.assertEqual(new(x).shape, legacy(x).shape)

    def test_cpu_benchmark_model_api(self):
        legacy = MnnMlp([3, 8], num_class=2).double().eval()
        new = MomentMlp([3, 8], num_classes=2).double().eval()
        start = time.perf_counter()
        for _ in range(3):
            legacy((self.mean, self.covariance))
        legacy_elapsed = (time.perf_counter() - start) / 3
        start = time.perf_counter()
        for _ in range(3):
            new((self.mean, self.covariance))
        new_elapsed = (time.perf_counter() - start) / 3
        print(f"legacy cpu MnnMlp: {legacy_elapsed:.6f}s/call")
        print(f"torch cpu MomentMlp: {new_elapsed:.6f}s/call")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_model_forward_stays_on_gpu(self):
        mean = self.mean.cuda()
        covariance = self.covariance.cuda()
        model = MomentMlp([3, 4], num_classes=2).double().cuda().eval()
        output = model((mean, covariance))
        self.assertEqual(output[0].device.type, "cuda")
        self.assertEqual(output[1].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
