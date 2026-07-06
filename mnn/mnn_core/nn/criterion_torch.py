# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from . import functional_torch as functional


class LabelSmoothingTorch(torch.nn.Module):
    def __init__(self, num_classes: int = 10, alpha: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, target: Tensor) -> Tensor:
        target = F.one_hot(target.to(torch.long), num_classes=self.num_classes)
        return (1 - self.alpha) * target + self.alpha / self.num_classes

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, alpha={self.alpha}"


class CrossEntropyOnMeanTorch(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs, target: Tensor) -> Tensor:
        mean = outputs if isinstance(outputs, Tensor) else outputs[0]
        return self.loss(mean, target)


class MSEOnMeanTorch(torch.nn.Module):
    def __init__(self, num_classes: int = 10, alpha: float = 0.0, is_classification: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.target_smoothing = LabelSmoothingTorch(num_classes, alpha) if is_classification else torch.nn.Identity()
        self.loss = torch.nn.MSELoss(*args, **kwargs)

    def forward(self, outputs, target: Tensor) -> Tensor:
        mean = outputs if isinstance(outputs, Tensor) else outputs[0]
        return self.loss(mean, self.target_smoothing(target))


class LikelihoodMSETorch(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        alpha: float = 0.0,
        normalize: bool = False,
        add_epsilon_covariance: bool = False,
        regularize_covariance: bool = True,
        eps: float = 1.0,
        reduction: str = "mean",
        gamma: float = 1.0,
        is_classification: bool = False,
    ) -> None:
        super().__init__()
        self.target_smoothing = LabelSmoothingTorch(num_classes, alpha) if is_classification else torch.nn.Identity()
        self.num_classes = num_classes
        self.normalize = normalize
        self.reduction = reduction
        self.regularize_covariance = regularize_covariance
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        if add_epsilon_covariance:
            self.register_buffer("epsilon_covariance", torch.eye(num_classes) * eps)
        else:
            self.register_buffer("epsilon_covariance", None)

    def forward(self, output: Tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        mean, covariance = output
        target = self.target_smoothing(target)
        if self.normalize:
            mean, covariance = functional.normalize_mean_covariance(mean, covariance)
        if self.epsilon_covariance is not None:
            covariance = covariance + self.epsilon_covariance
        delta = mean - target.reshape_as(mean)
        loss = torch.matmul(torch.matmul(delta.unsqueeze(-2), torch.linalg.inv(covariance)), delta.unsqueeze(-1)).squeeze(-1)
        if self.regularize_covariance:
            loss = loss + torch.logdet(covariance) * self.gamma
        return torch.mean(loss) if self.reduction == "mean" else torch.sum(loss)


class GaussianSamplingCrossEntropyLossTorch(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_samples: int = 1000,
        eps: float = 1e-6,
        beta: float = 1.0,
        decoding_time: float = 1.0,
        do_predict: bool = False,
        return_prediction: bool = False,
        normalize: bool = True,
        reduction: str = "mean",
        loss_fn: str = "add",
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.register_buffer("eps", torch.eye(num_classes) * eps)
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction, **kwargs)
        self.return_prediction = return_prediction
        self.normalize = normalize
        self.beta = beta
        self.decoding_time = decoding_time
        self.do_predict = do_predict
        self.reduction = reduction
        self.loss_fn = loss_fn

    def compute_loss(self, transformed_samples: Tensor, target: Tensor) -> Tensor:
        output = F.softmax(transformed_samples * self.beta, dim=-1)
        output = torch.mean(output, dim=-2)
        return F.nll_loss(torch.log(output), target, reduction=self.reduction)

    def loss_sample_as_logits(self, transformed_samples: Tensor, target: Tensor) -> Tensor:
        target = target.unsqueeze(-1).expand(-1, self.num_samples).flatten()
        return self.loss(transformed_samples * self.beta, target)

    def make_prediction(self, output: Tuple[Tensor, Tensor], sample_points: Tensor, transformed_samples: Tensor) -> Tensor:
        mean, covariance = output
        return functional.gaussian_sampling_predict(
            sample_points,
            mean,
            covariance,
            self.eps,
            decoding_time=self.decoding_time,
            transformed_samples=transformed_samples,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            normalize=self.normalize,
            return_prediction=self.return_prediction,
        )

    def forward(self, output: Tuple[Tensor, Tensor], target: Tensor):
        mean, covariance = output
        sample_points = torch.randn(mean.size(0), self.num_samples, self.num_classes, device=mean.device)
        transformed_samples = functional.gaussian_sampling_transform(
            sample_points,
            mean,
            covariance,
            self.eps,
            decoding_time=self.decoding_time,
        )
        loss = self.compute_loss(transformed_samples, target) if self.loss_fn == "add" else self.loss_sample_as_logits(transformed_samples, target)
        if self.do_predict:
            return loss, self.make_prediction(output, sample_points, transformed_samples)
        return loss


class GaussianSamplingPredictTorch(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_samples: int = 1000,
        eps: float = 1e-6,
        decoding_time: float = 1.0,
        return_prediction: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("sample_points", torch.randn(num_samples, num_classes))
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.register_buffer("eps", torch.eye(num_classes) * eps)
        self.return_prediction = return_prediction
        self.normalize = normalize
        self.decoding_time = decoding_time

    @torch.no_grad()
    def forward(self, output: Tuple[Tensor, Tensor]) -> Tensor:
        mean, covariance = output
        return functional.gaussian_sampling_predict(
            self.sample_points,
            mean,
            covariance,
            self.eps,
            decoding_time=self.decoding_time,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            normalize=self.normalize,
            return_prediction=self.return_prediction,
            expand_sample=True,
        )

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, num_samples={self.num_samples}, "
            f"eps={self.eps[0, 0].item()}, decoding_time={self.decoding_time}, "
            f"return_prediction={self.return_prediction}, normalize={self.normalize}"
        )


class SampleBasedEarthMoverLossTorch(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_samples: int = 1000,
        eps: float = 1e-4,
        beta: float = 1.0,
        decoding_time: float = 1.0,
        loss_func: Optional[Callable] = None,
        reduction: str = "mean",
        regularize_covariance: bool = False,
        add_diagonal: Optional[float] = None,
        normalize: bool = False,
        is_classification: bool = False,
        alpha: float = 0.0,
        use_acos: bool = True,
        safe_scale: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.register_buffer("eps", torch.eye(num_classes) * eps)
        self.beta = beta
        self.decoding_time = decoding_time
        self.reduction = reduction
        self.default_distance = torch.nn.CosineSimilarity(dim=-1)
        self.regularize_covariance = regularize_covariance
        self.loss_func = loss_func
        self.normalize = normalize
        self.use_acos = use_acos
        self.safe_scale = safe_scale
        self.target_smoothing = LabelSmoothingTorch(num_classes, alpha) if is_classification else torch.nn.Identity()
        if add_diagonal is not None:
            self.register_buffer("add_diagonal", torch.eye(num_classes) * add_diagonal)
        else:
            self.register_buffer("add_diagonal", None)

    def _default_cosine_loss(self, output: Tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        mean, covariance = output
        if self.normalize:
            mean, covariance = functional.normalize_mean_covariance(mean, covariance)
        sample_points = torch.randn(mean.size(0), self.num_samples, self.num_classes, device=mean.device)
        transformed_samples = functional.gaussian_sampling_transform(
            sample_points,
            mean,
            covariance,
            self.eps,
            decoding_time=self.decoding_time,
        ) * self.beta
        target = target.reshape(-1, 1, self.num_classes).expand_as(transformed_samples)
        distance = self.default_distance(transformed_samples, target)
        loss = torch.mean(torch.arccos(distance * self.safe_scale), dim=-1) if self.use_acos else 1 - torch.mean(distance, dim=-1)
        if self.regularize_covariance:
            if self.add_diagonal is not None:
                loss = loss + torch.logdet(covariance + self.add_diagonal) / 2
            else:
                loss = loss + torch.log(torch.linalg.det(covariance) + 1) / 2
        return torch.sum(loss) if self.reduction == "sum" else torch.mean(loss)

    def forward(self, output: Tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        target = self.target_smoothing(target)
        if self.loss_func is None:
            return self._default_cosine_loss(output, target)
        return self.loss_func(self, output, target)


class FidelityLossTorch(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        decoding_time: float = 1.0,
        reduction: str = "mean",
        use_full: bool = True,
        fidelity_weight: Optional[Tensor] = None,
        trainable_weight_loss: bool = False,
    ) -> None:
        super().__init__()
        if decoding_time <= 0:
            raise ValueError("decoding_time must be positive")
        self.alpha = alpha
        self.reduction = reduction
        self.decoding_time = decoding_time
        self.use_full = use_full
        self.fidelity_weight = fidelity_weight
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.trainable_weight_loss = trainable_weight_loss

    def fidelity_entropy(self, mean: Tensor, covariance: Tensor, idx1: Tensor, idx2: Tensor, batch_idx: Tensor, eps: float = 1e-6) -> Tensor:
        pair_mean = (mean[batch_idx, idx2] - mean[batch_idx, idx1]) * math.sqrt(self.decoding_time)
        pair_variance = (
            covariance[batch_idx, idx1, idx1]
            + covariance[batch_idx, idx2, idx2]
            - covariance[batch_idx, idx1, idx2]
            - covariance[batch_idx, idx2, idx1]
        )
        p = 0.5 * torch.erfc(pair_mean / torch.sqrt(2 * pair_variance)) - eps
        q = 1 - p
        return -(p * torch.log(p) + q * torch.log(q))

    def forward(self, x: Tuple[Tensor, Tensor], target: Tensor, trainable_weight: Optional[Tensor] = None) -> Tensor:
        mean, covariance = x
        batch, num_classes = mean.size()
        if num_classes < 2:
            raise ValueError("FidelityLossTorch requires at least two classes")
        loss = self.ce(mean, target)
        second_loss = 0
        if self.alpha > 0:
            prediction = torch.max(mean, dim=-1)[1]
            sign = torch.where(prediction == target, 1, -1)
            batch_idx = torch.arange(batch, device=mean.device)
            if self.use_full and num_classes > 2:
                _, indices = torch.sort(mean, dim=-1, descending=True)
                best = indices[:, 0]
                if trainable_weight is not None:
                    weight = torch.nn.functional.softmax(
                        trainable_weight.clone().detach() if self.trainable_weight_loss else trainable_weight,
                        dim=-1,
                    )
                else:
                    weight = self.fidelity_weight
                trainable_losses = []
                for index in range(1, num_classes):
                    item_weight = 0.8 if weight is None and index == 1 else (0.2 / (num_classes - 2) if weight is None else weight[index - 1])
                    item_loss = self.fidelity_entropy(mean, covariance, best, indices[:, index], batch_idx)
                    second_loss = second_loss + item_loss * item_weight
                    if trainable_weight is not None and self.trainable_weight_loss:
                        trainable_losses.append(item_loss)
                if trainable_weight is not None and self.trainable_weight_loss:
                    stacked = torch.stack(trainable_losses, dim=-1)
                    trainable_idx = torch.max(stacked, dim=-1)[1]
                    expanded_weight = trainable_weight.unsqueeze(0).expand_as(stacked)
                    second_loss = second_loss - torch.gather(expanded_weight, dim=-1, index=trainable_idx.unsqueeze(-1)).reshape_as(second_loss)
            else:
                _, indices = torch.topk(mean, 2, dim=-1)
                second_loss = self.fidelity_entropy(mean, covariance, indices[:, 0], indices[:, 1], batch_idx)
            second_loss = sign * second_loss
            second_loss = torch.mean(second_loss) if self.reduction == "mean" else torch.sum(second_loss)
        return loss + self.alpha * second_loss
