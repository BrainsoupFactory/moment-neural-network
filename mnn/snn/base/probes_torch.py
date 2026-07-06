# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Union

import torch


class NeuronProbeTorch(torch.nn.Module):
    def __init__(
        self,
        attr: Union[str, List[str]],
        *,
        dt: float = 1e-2,
        probe_interval: int = 1,
        indices=None,
        collect_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attr = attr
        self.dt = dt
        self.probe_interval = math.ceil(probe_interval / dt)
        self.indices = indices
        self.collect_on_cpu = collect_on_cpu
        self.step_count = 0
        self.records = defaultdict(list)

    def reset(self, *args, **kwargs) -> None:
        self.step_count = 0
        self.records = defaultdict(list)

    def collect_data(self, neurons: torch.nn.Module, key: str, neurons_alias=None) -> None:
        value = getattr(neurons, key)
        if self.indices is not None:
            value = value[self.indices]
        value = value.clone().detach()
        if self.collect_on_cpu:
            value = value.cpu()
        if neurons_alias is None:
            self.records[key].append(value.unsqueeze(0))
        else:
            if isinstance(self.records[neurons_alias], list):
                self.records[neurons_alias] = defaultdict(list)
            self.records[neurons_alias][key].append(value.unsqueeze(0))

    def forward(self, neurons: torch.nn.Module, neurons_alias=None) -> None:
        self.step_count += 1
        if self.step_count % self.probe_interval != 0:
            return
        if isinstance(self.attr, str):
            self.collect_data(neurons, self.attr, neurons_alias=neurons_alias)
        else:
            for key in self.attr:
                self.collect_data(neurons, key, neurons_alias=neurons_alias)

    def dump_collected_data(self, save_path) -> None:
        torch.save(self.records, save_path)

    def get_data(self):
        return self.records
