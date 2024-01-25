import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from . import functional


class LabelSmoothing(torch.nn.Module):
    def __init__(self, num_class: int = 10, alpha: float = 0.1) -> None:
        super(LabelSmoothing, self).__init__()
        assert 0. <= alpha < 1.
        self.num_class = num_class
        self.alpha = alpha

    def forward(self, target: Tensor) -> Tensor:
        target = target.to(torch.long)
        target = F.one_hot(target, num_classes=self.num_class)
        return (1 - self.alpha) * target + self.alpha / self.num_class

    def extra_repr(self) -> str:
        return 'num_classes: {}, alpha: {}'.format(self.num_class, self.alpha)

class CrossEntropyOnMean(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyOnMean, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs, target):
        if not isinstance(outputs, Tensor):
            u, _ = outputs
        else:
            u = outputs
        return self.loss(u, target)

class MSEOnMean(torch.nn.Module):
    def __init__(self, num_class: int = 10, alpha: float = 0.,  is_classify=False, *args, **kwargs):
        super(MSEOnMean, self).__init__()
        if is_classify:
            self.target_smooth = LabelSmoothing(num_class, alpha)
        else:
            self.target_smooth = torch.nn.Identity()
        self.loss = torch.nn.MSELoss(*args, **kwargs)

    def forward(self, outputs, target):
        if not isinstance(outputs, Tensor):
            u, _ = outputs
        else:
            u = outputs
        target = self.target_smooth(target)
        return self.loss(u, target)

class LikelihoodMSE(torch.nn.Module):
    """
    Input shape:
    u: [batch, neuron]
    cov: [batch, neuron, neuron]
    target: [batch, 1] -> one hot encoding

    """
    def __init__(self, num_class: int = 10, alpha=0., normalize: bool = False, extra_add: bool = False,
                 regular_cov: bool = True, eps: float = 1., reduction: str = 'mean', gamma: float = 1.,
                 is_classify=False):
        super(LikelihoodMSE, self).__init__()
        if is_classify:
            self.target_smooth = LabelSmoothing(num_class, alpha)
        else:
            self.target_smooth = torch.nn.Identity()
        self.num_class = num_class
        self.normalize = normalize
        self.extra_add = extra_add
        self.reduction = reduction
        self.regular_cov = regular_cov
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        if extra_add:
            self.register_buffer('eps_cov', torch.eye(num_class) * eps)
        else:
            self.register_buffer('eps_cov', None)

    def forward(self, output: Tuple[Tensor, Tensor], target: Tensor) -> Tensor:
        u, cov = output
        target = self.target_smooth(target)
        if self.normalize:
            u, cov = functional.normalize_mean_cov(u, cov)
        if self.eps_cov is not None:
            cov = cov + self.eps_cov
        temp = u - target.reshape_as(u)
        loss = torch.matmul(torch.matmul(temp.unsqueeze(-2), torch.linalg.inv(cov)), temp.unsqueeze(-1)).squeeze(-1)
        if self.regular_cov:
            temp = torch.logdet(cov) * self.gamma
            loss = loss + temp
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss

    def extra_repr(self) -> str:
        return 'num_class: {}, normalize={}, extra_add={}, reduction: {}, regular_cov={}, alpha={}, gamma={}, eps={}'.format(
            self.num_class, self.normalize, self.extra_add, self.reduction, self.regular_cov, self.alpha, self.gamma, self.eps
        )


class GaussianSamplingCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_class=10, num_sample=1000, eps=1e-6, beta=1., decoding_time=1., do_predict=False,
                 return_pred=False, normalise=True, reduction='mean', loss_fn='add',**kwargs):
        super(GaussianSamplingCrossEntropyLoss, self).__init__()
        self.num_class = num_class
        self.num_sample = num_sample
        self.register_buffer('eps', torch.eye(num_class) * eps)
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction, **kwargs)
        self.return_pred = return_pred
        self.normalise = normalise
        self.beta = beta
        self.decoding_time = decoding_time
        self.do_predict = do_predict
        self.reduction = reduction
        self.loss_fn = loss_fn
    
    def compute_loss(self, transformed_samples, target):
        transformed_samples = transformed_samples * self.beta
        output = F.softmax(transformed_samples, dim=-1)
        output = torch.mean(output, dim=-2)
        log_prob = torch.log(output)
        loss = F.nll_loss(log_prob, target, reduction=self.reduction)
        return loss
    
    def make_predict(self, output, sample_points, transformed_samples):
        u, cov = output
        pred = functional.gaussian_sampling_pred(sample_points=sample_points, u=u, cov=cov, eps=self.eps, decoding_time=self.decoding_time,
                                                        transformed_samples=transformed_samples, num_class=self.num_class, num_sample=self.num_sample,
                                                        normalise=self.normalise, return_pred=self.return_pred, need_expand=False)
        return pred
        
    
    def loss_sample_as_logits(self, transformed_samples, target):
            target = target.unsqueeze(-1).expand(-1, self.num_sample).flatten()
            loss = self.loss(transformed_samples * self.beta, target)
            return loss
    
    def forward(self, output, target):
        u, cov = output
        sample_points = torch.randn(u.size(0), self.num_sample, self.num_class, device=u.device)
        transformed_samples = functional.gaussian_sampling_transform(sample=sample_points, u=u, cov=cov,
                                                        eps=self.eps, decoding_time=self.decoding_time, need_expand=False)
        if self.loss_fn == 'add':
            loss = self.compute_loss(transformed_samples, target)
        else:
            loss = self.loss_sample_as_logits(transformed_samples, target)
        
        if self.do_predict:
            pred = self.make_predict(output, sample_points, transformed_samples)
            return loss, pred
        else:
            return loss

    def extra_repr(self) -> str:
        return 'num_class: {}, num_sample: {}, eps={}, beta={}, decoding_time={}, do_predict={}, ' \
               'return_pred={}, normalise={}, reduction={}'.format(
            self.num_class, self.num_sample, self.eps[0, 0].item(), self.beta, self.decoding_time,
            self.do_predict, self.return_pred, self.normalise, self.reduction)


class GaussianSamplingPredict(torch.nn.Module):
    def __init__(self, num_class=10, num_sample=1000, eps=1e-6, decoding_time=1., return_pred=True, normalise=True, **kwargs):
        super(GaussianSamplingPredict, self).__init__()
        self.register_buffer('sample_point', torch.randn(num_sample, num_class))
        self.num_class = num_class
        self.num_sample = num_sample
        self.register_buffer('eps', torch.eye(num_class) * eps)
        self.return_pred = return_pred
        self.normalise = normalise
        self.decoding_time = decoding_time

    @torch.no_grad()
    def forward(self, output):
        u, cov = output
        pred = functional.gaussian_sampling_pred(sample_points=self.sample_point, u=u, cov=cov, eps=self.eps,
                                                 decoding_time=self.decoding_time, transformed_samples=None,
                                                 num_class=self.num_class, num_sample=self.num_sample,
                                                 normalise=self.normalise, return_pred=self.return_pred, need_expand=True)
        return pred

    def extra_repr(self) -> str:
        return 'num_class: {}, num_sample: {}, eps={}, decoding_time={}, return_pred={}, normalise={}'.format(
            self.num_class, self.num_sample, self.eps[0, 0].itme(), self.decoding_time, self.return_pred, self.normalise)


class SampleBasedEarthMoverLoss(torch.nn.Module):
    def __init__(self, num_class=2, num_sample=1000, eps=1e-4, beta=1., decoding_time=1. , loss_func=None, 
    reduction='mean', regular_cov=False, add_diag=None, normalize=False, is_classify=False, alpha=0., use_acos=True, safe_scale=0.99) -> None:
        super(SampleBasedEarthMoverLoss, self).__init__()
        self.num_class = num_class
        self.num_sample = num_sample
        self.register_buffer('eps', torch.eye(num_class) * eps)
        self.beta = beta
        self.decoding_time = decoding_time
        self.reduction = reduction
        self.default_distance =torch.nn.CosineSimilarity(dim=-1)
        self.regular_cov = regular_cov
        self.loss_func = loss_func
        self.normalize = normalize
        self.use_acos = use_acos
        self.safe_scale = safe_scale
        
        if add_diag is not None:
            self.register_buffer('add_diag', torch.eye(num_class) * add_diag)
        else:
            self.register_buffer('add_diag', None)
        
        if is_classify:
            self.target_smooth = LabelSmoothing(num_class=num_class, alpha=alpha)
        else:
            self.target_smooth = torch.nn.Identity()
    
    def extra_repr(self) -> str:
        return 'num_class: {}, num_sample: {}, eps={}, beta={}, decoding_time={}, reduction={}, loss_func: {}'.format(
            self.num_class, self.num_sample, self.eps[0, 0].item(), self.beta, self.decoding_time, self.reduction, self.loss_func)

    def _default_cosine_loss(self, output: Tuple, target: Tensor):
        u, cov = output
        if self.normalize:
            u, cov = functional.normalize_mean_cov(u, cov)
        sample_points = torch.randn(u.size(0), self.num_sample, self.num_class, device=u.device)
        transformed_samples = functional.gaussian_sampling_transform(sample=sample_points, u=u, cov=cov,
                                                        eps=self.eps, decoding_time=self.decoding_time, need_expand=False)
        transformed_samples = transformed_samples * self.beta
        target = target.reshape(-1, 1, self.num_class).expand_as(transformed_samples)
        distance = self.default_distance(transformed_samples, target)
        if self.use_acos:
            loss = torch.mean(torch.arccos(distance * self.safe_scale), dim=-1)
        else:
            loss = 1 - torch.mean(distance, dim=-1)
        if self.regular_cov:
            if self.add_diag is not None:
                loss = loss + torch.logdet(cov + self.add_diag) / 2
            else:
                loss = loss + torch.log(torch.linalg.det(cov) + 1) / 2
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss
    
    def forward(self, output: Tuple, target: Tensor):
        target = self.target_smooth(target)
        if self.loss_func is None:
            loss = self._default_cosine_loss(output, target)
        else:
            loss = self.loss_func(self, output, target)
        return loss


