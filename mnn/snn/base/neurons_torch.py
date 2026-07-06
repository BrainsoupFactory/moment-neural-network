# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import Tensor


class LifNeurons(torch.nn.Module):
    __constants__ = ["num_neurons", "spike_dtype"]

    def __init__(
        self,
        num_neurons,
        *,
        membrane_conductance: float = 1 / 20,
        threshold_voltage: float = 20.0,
        resting_voltage: float = 0.0,
        spike_voltage: float = 50.0,
        dt: float = 1e-1,
        refractory_time: float = 5.0,
        init_voltage: str | None = None,
        spike_dtype=torch.float,
        **legacy_aliases,
    ) -> None:
        super().__init__()
        membrane_conductance = legacy_aliases.pop("L", membrane_conductance)
        threshold_voltage = legacy_aliases.pop("V_th", threshold_voltage)
        resting_voltage = legacy_aliases.pop("V_res", resting_voltage)
        spike_voltage = legacy_aliases.pop("V_spk", spike_voltage)
        refractory_time = legacy_aliases.pop("T_ref", refractory_time)
        init_voltage = legacy_aliases.pop("init_vol", init_voltage)
        if legacy_aliases:
            unknown = ", ".join(sorted(legacy_aliases))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        self.num_neurons = num_neurons
        self.init_voltage = init_voltage
        self.spike_dtype = spike_dtype
        if init_voltage == "uniform":
            voltage = torch.rand(num_neurons) * threshold_voltage
        else:
            voltage = torch.ones(num_neurons) * resting_voltage
        self.register_buffer("voltage", voltage)
        self.register_buffer("is_active", torch.ones(num_neurons, dtype=torch.bool))
        self.register_buffer("refractory_state", torch.zeros(num_neurons))
        self.register_buffer("membrane_conductance", torch.ones(1) * membrane_conductance)
        self.register_buffer("threshold_voltage", torch.ones(1) * threshold_voltage)
        self.register_buffer("resting_voltage", torch.ones(1) * resting_voltage)
        self.register_buffer("spike_voltage", torch.ones(1) * spike_voltage)
        self.register_buffer("dt", torch.ones(1) * dt)
        self.register_buffer("refractory_time", torch.ones(1) * refractory_time)

        self.V = self.voltage
        self.is_active = self.is_active
        self.ref_state = self.refractory_state
        self.L = self.membrane_conductance
        self.V_th = self.threshold_voltage
        self.V_res = self.resting_voltage
        self.V_spk = self.spike_voltage
        self.T_ref = self.refractory_time

    @torch.no_grad()
    def reset(self) -> None:
        if self.init_voltage == "uniform":
            self.voltage.copy_(torch.rand(self.num_neurons, dtype=self.voltage.dtype, device=self.voltage.device) * self.threshold_voltage)
        else:
            self.voltage.fill_(self.resting_voltage.item())
        self.is_active.fill_(True)
        self.refractory_state.zero_()

    def update_current(self, input_current: Tensor) -> Tensor:
        return input_current

    def update_voltage(self, input_current: Tensor) -> None:
        refractory = torch.logical_not(self.is_active)
        active = self.is_active
        self.voltage[active] += -self.voltage[active] * self.membrane_conductance * self.dt + input_current[active]
        self.refractory_state[refractory] += self.dt

    def update_state(self) -> Tensor:
        spikes = self.voltage >= self.threshold_voltage
        recovered = self.refractory_state >= self.refractory_time
        self.refractory_state.masked_fill_(recovered, 0.0)
        self.is_active.masked_fill_(recovered, True)
        self.is_active.masked_fill_(spikes, False)
        self.voltage.masked_fill_(spikes, self.resting_voltage.item())
        return spikes

    def forward(self, input_current: Tensor) -> Tensor:
        input_current = self.update_current(input_current)
        self.update_voltage(input_current)
        return self.update_state().to(self.spike_dtype)

    def extra_repr(self) -> str:
        return (
            f"membrane_conductance={self.membrane_conductance.item()}, "
            f"threshold_voltage={self.threshold_voltage.item()}, "
            f"resting_voltage={self.resting_voltage.item()}, dt={self.dt.item()}, "
            f"refractory_time={self.refractory_time.item()}"
        )
