# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import Tensor


class SpikeMonitorTorch(torch.nn.Module):
    def __init__(self, num_neurons, *, dt: float = 1e-1, dtype=torch.int, **kwargs) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.dtype = dtype
        self.step_count = 0
        self.register_buffer("monitor", torch.zeros(num_neurons, dtype=dtype).unsqueeze(0).to_sparse())

    def reset(self) -> None:
        device = self.monitor.device
        self.monitor = torch.zeros(self.num_neurons, dtype=self.dtype, device=device).unsqueeze(0).to_sparse()
        self.step_count = 0

    def forward(self, x: Tensor) -> Tensor:
        self.step_count += 1
        spikes = x.to(self.dtype).unsqueeze(0).to_sparse()
        self.monitor = torch.cat([self.monitor, spikes], dim=0)
        return x

    def spike_count(self, device="cpu") -> tuple[Tensor, float]:
        try:
            count = torch.sparse.sum(self.monitor, dim=0).to(device).to_dense()
        except RuntimeError:
            count = torch.zeros(self.num_neurons, device=device)
        return count, self.step_count * self.dt

    def dump_monitor_state(self, save_path) -> None:
        torch.save(self.state_dict(), save_path)

    def get_data(self) -> Tensor:
        return self.monitor.clone().detach().cpu()
