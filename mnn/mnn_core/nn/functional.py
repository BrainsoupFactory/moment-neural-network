import math
import torch
from torch import Tensor
from typing import Tuple, Any, Optional
import torch.nn.functional as F


class ModifyDiagToOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, rho: Tensor) -> Tensor:
        torch.diagonal(rho, dim1=-1, dim2=-2).data.fill_(1.0)
        return rho

    @staticmethod
    def backward(ctx: Any, grad_rho: Tensor) -> Tensor:
        torch.diagonal(grad_rho, dim1=-1, dim2=-2).data.fill_(0.)
        return grad_rho

diag2one = ModifyDiagToOne.apply

def compute_covariance(std_in: Tensor, corr_in: Tensor, corr_check=False) -> Tensor:
    if corr_check:
        corr_in = diag2one(corr_in)
    cov = torch.mul(torch.matmul(std_in.unsqueeze(-1), std_in.unsqueeze(-2)), corr_in)
    return cov

def compute_correlation(cov: Tensor, eps: float = 1e-5, safe_check: bool = False) -> Tuple[Tensor, Tensor]:
    var_out = torch.diagonal(cov, dim1=-2, dim2=-1)
    if safe_check:
        cov, var_out = spd_insurance(cov, var_out)
    # add a eps to prevent divide zero error
    std_out = torch.sqrt(var_out) + eps
    corr_out = torch.div(cov, torch.matmul(std_out.unsqueeze(-1), std_out.unsqueeze(-2)))
    corr_out = diag2one(corr_out)
    return std_out, corr_out

def compute_signal_correlation(mean: Tensor, cov: Tensor):
    signal_feature = torch.matmul(mean.unsqueeze(-1), mean.unsqueeze(-2)) + cov
    return signal_feature

def spd_insurance(cov: Tensor, var_out: Optional[Tensor] = None):
    if var_out is None:
        var_out = torch.diagonal(cov, dim1=-2, dim2=-1)
    # make sure that cov is always symmetric pos-definite
    mask = torch.gt(var_out, 0).to(torch.float16)
    mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(-2))
    cov = cov * mask
    var_out = torch.relu(var_out)
    return cov, var_out

def modify_as_covariance(cov: Tensor, scale: Optional[float] = None) -> Tensor:
    cov = torch.matmul(cov, torch.transpose(cov, -2, -1))
    # inspired by Attention
    if scale is None:
        n = cov.size()[-1]
        cov = cov / math.sqrt(n)
    else:
        cov = cov * scale
    return cov

def var2cov(var: Tensor, scale=None):
    assert var.dim() == 1
    cov = torch.diag(F.softplus(var))
    if scale is not None:
        cov = cov / scale
    return cov

def mnn_linear(u: Tensor, cov: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               bias_var: Optional[Tensor] = None, dropout: Optional[torch.nn.Dropout] = None, scale=None) -> Tuple[Tensor, Tensor]:
    if dropout is not None:
        n = u.size()[0]
        weight = dropout(weight.unsqueeze(0).expand(n, -1, -1))

    u = torch.matmul(weight, u.unsqueeze(-1)).squeeze(-1)
    if bias is not None:
        u = u + bias
    cov = torch.matmul(weight, torch.matmul(cov, weight.transpose(-1, -2)))
    if bias_var is not None:
        cov = cov + var2cov(bias_var, scale)
    return u, cov

def mnn_bn1d_forward(bn: torch.nn.BatchNorm1d, u: Tensor, cov: Tensor, weight: Tensor,
                     bias_var: Optional[Tensor] = None, scale=None):
    u_hat = bn(u)
    if bn.training:
        mean_std = torch.std(u, dim=0, keepdim=True) + bn.eps
        cov = torch.mm(weight.unsqueeze(-1), weight.unsqueeze(-2)) * cov \
              / torch.matmul(mean_std.unsqueeze(-1), mean_std.unsqueeze(-2))
    else:
        if bn.track_running_stats is True:
            mean_std = torch.sqrt(bn.running_var) + bn.eps
            cov = torch.mm(weight.unsqueeze(-1), weight.unsqueeze(-2)) * cov \
                  / torch.matmul(mean_std.unsqueeze(-1), mean_std.unsqueeze(-2))
        else:
            mean_std = torch.std(u, dim=0, keepdim=True) + bn.eps
            cov = torch.mm(weight.unsqueeze(-1), weight.unsqueeze(-2)) * cov \
                  / torch.matmul(mean_std.unsqueeze(-1), mean_std.unsqueeze(-2))

    if bias_var is not None:
        cov = cov + var2cov(bias_var, scale)
    return u_hat, cov

def normalize_mean_cov(u: Tensor, cov: Tensor,  eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    weight = torch.sum(torch.pow(u, 2), dim=-1, keepdim=True) + eps
    cov = cov / weight.unsqueeze(-1)
    u = u / torch.sqrt(weight)
    return u, cov

def gaussian_sampling_transform(sample, u, cov, eps, decoding_time: float = 1., need_expand=False):
    assert eps.dim() >= 2
    b, n = u.size()
    if need_expand:
        sample = sample.unsqueeze(0).expand(b, -1, -1)

    l = torch.linalg.cholesky_ex(cov + eps)[0].unsqueeze(1)
    sample = torch.matmul(l, sample.unsqueeze(-1)).squeeze(-1) / math.sqrt(decoding_time) + u.unsqueeze(1)
    return sample

@torch.no_grad()
def gaussian_sampling_pred(sample_points, u, cov, eps, decoding_time=1., transformed_samples=None, num_class=10, num_sample=1000,
                           normalise=False, return_pred=True, need_expand=False):
    if transformed_samples is None:
        transformed_samples = gaussian_sampling_transform(sample_points, u, cov, eps, decoding_time, need_expand)
    pred = torch.max(transformed_samples, dim=-1)[-1]
    pred = F.one_hot(pred, num_class)
    pred = torch.sum(pred, dim=1)
    if normalise:
        pred = pred / num_sample
    if return_pred:
        pred = torch.max(pred, dim=-1)[-1]
    return pred

def triu_vec(cov, diagonal=0):
    n = cov.size()[-1]
    idx = torch.flatten(torch.triu(torch.ones(n, n, device=cov.device), diagonal=diagonal)).nonzero()
    vec: Tensor = torch.flatten(cov, start_dim=-2)[..., idx]
    return vec

def parse_input(args):
    if isinstance(args[0], Tensor):
        u, cov = args
    else:
        u, cov = args[0]
    return u, cov

def mean_cov_pooling(x: Tensor, unbias=True, biological_pooling=False):
    if x.dim() == 2:
        _, d = x.size()
    else:
        _, _, d = x.size()
    if unbias:
        d = d - 1
    mean = torch.mean(x, dim=-1, keepdim=True)
    cov = x - mean
    if x.dim() == 2:
        cov = torch.einsum('m i, n i -> m n', cov, cov) / d
    else:
        cov = torch.einsum('b m i, b n i -> b m n', cov, cov) / d
    mean = mean.squeeze(-1)
    if biological_pooling:
        cov = cov + torch.diag_embed(mean)
    return mean, cov

