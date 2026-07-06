# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .. import utils
from ..mnn_core.nn import functional_torch as moment_functional
from . import base
from .base.currents_torch import GaussianCurrentSource, PoissonSpikeSource
from .base.functional_torch import sample_shape
from .base.monitors_torch import SpikeMonitorTorch
from .mnn_to_snn_torch import MomentMlpToSnn, SpikeMomentMlpToSnn


class MomentSnnValidator:
    def __init__(
        self,
        args,
        *,
        running_time: float = 20,
        dt: float = 1e-2,
        num_trials=100,
        monitor_size=None,
        pregenerate: bool = False,
        resume_best: bool = True,
        train: bool = False,
        init_voltage=None,
        alias: str = "",
        input_type: str = "gaussian",
        train_funcs=None,
        unsqueeze_input=None,
        align_batch_size: bool = False,
        config_save_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.running_time = running_time
        self.train = train
        self.pregenerate = pregenerate
        self.dt = dt
        self.num_steps = int(running_time / dt)
        self.num_trials = num_trials
        self.monitor_size = monitor_size
        self.init_voltage = init_voltage
        self.alias = alias
        self.prefix = f"run{running_time}_dt{dt}"
        self.input_type = input_type
        self.unsqueeze_input = unsqueeze_input
        self.config_save_path = config_save_path
        self.train_funcs = train_funcs or utils.training_tools.general_train.TrainProcessCollections()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.args = self.resume_config(args, resume_best)
        if align_batch_size:
            self.args.bs = self.num_trials
        self.prepare_dump_dir()
        self.generate_dataset()
        self.generate_model()
        self.extra_works(**kwargs)

    def extra_works(self, *args, **kwargs) -> None:
        pass

    def resume_config(self, args, resume_best: bool):
        save_path = self.config_save_path or getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
        args.save_path = save_path
        args.config = save_path + args.save_name + "_config.yaml"
        args = utils.training_tools.set_config2args(args)
        args.save_path = save_path
        args.config = save_path + args.save_name + "_config.yaml"
        args.resume_best = resume_best
        if getattr(self, "local_rank", None) is not None:
            args.local_rank = self.local_rank
        args.use_cuda = getattr(args, "use_cuda", True)
        args.unsqueeze_input = self.unsqueeze_input
        return args

    def reset(self) -> None:
        self.snn.reset()
        self.snn.eval()
        self.mnn.eval()

    def custom_reset(self) -> None:
        self.snn.reset_spike_count_list()
        self.snn.reset_generator()
        self.snn.reset_monitor()
        self.snn.reset_probe()

    def __len__(self) -> int:
        return len(getattr(self, "dataset"))

    def prepare_dump_dir(self) -> None:
        save_path = getattr(self.args, "dump_path", "./checkpoint/") + self.args.dir + f"/{self.args.save_name}_snn_validate_result/"
        self.spike_dump_path = save_path
        utils.training_tools.RecordMethods.make_dir(save_path)

    def generate_dataset(self) -> None:
        train_loader, test_loader = self.train_funcs.prepare_dataloader(self.args)
        setattr(self, "dataset", getattr(train_loader, "dataset") if self.train else getattr(test_loader, "dataset"))

    def generate_model(self) -> None:
        mnn_model = self.generate_mnn_model()
        snn_model = self.generate_snn_model()
        snn_model.load_state_dict(mnn_model.state_dict())
        snn_model.mnn_to_snn(
            dt=self.dt,
            batch_size=self.num_trials,
            monitor_size=self.monitor_size,
            pregenerate=self.pregenerate,
            num_steps=self.num_steps,
            init_voltage=self.init_voltage,
            **getattr(self.args, "NEURONS", {}),
        )
        self.mnn = mnn_model
        self.snn = snn_model
        if getattr(self.args, "use_cuda", True) and torch.cuda.is_available():
            self.mnn.cuda(self.args.local_rank)
            self.snn.cuda(self.args.local_rank)
        self.reset()

    def generate_mnn_model(self):
        return utils.training_tools.model_generator(
            self.args.save_path,
            self.args.save_name,
            to_cuda=True,
            resume_model=True,
            resume_best=self.args.resume_best,
            local_rank=self.args.local_rank,
            make_func=self.train_funcs,
        )

    def generate_snn_model(self):
        model_type = self.args.MODEL["meta"]["mlp_type"]
        model_args = self.args.MODEL[model_type]
        if model_type in {"snn_mlp", "spike_moment_mlp"}:
            model_cls = getattr(self, "SNN_MLP", SpikeMomentMlpToSnn)
        else:
            model_cls = getattr(self, "MNN_MLP", MomentMlpToSnn)
        return model_cls(**model_args)

    def save_result(self, idx, overwrite: bool = True, probe_alias=None, reset_probe: bool = True, **result) -> None:
        data_source = "train" if self.train else "test"
        result.update(
            {
                "input_type": self.input_type,
                "idx": idx,
                "data_source": data_source,
                "spike_count": self.snn.spike_count,
                "record_duration": self.snn.record_duration,
                "probe_data": self.collect_probe_data(probe_alias=probe_alias, reset_probe=reset_probe),
            }
        )
        save_name = f"{self.alias}{self.prefix}_{self.input_type}_{data_source}_idx_{idx}.snnval"
        if not overwrite:
            save_name = utils.training_tools.RecordMethods.rename_duplicate_file(self.spike_dump_path, save_name)
        torch.save(result, self.spike_dump_path + save_name)

    def collect_probe_data(self, probe_alias=None, reset_probe: bool = True):
        return self.snn.collect_probe_data(probe_alias=probe_alias, reset_probe=reset_probe)

    def dump_spike_train(self, idx, overwrite: bool = True, monitor_alias=None) -> None:
        data_source = "train" if self.train else "test"
        save_name = f"{self.alias}{self.prefix}_{self.input_type}_{data_source}_idx_{idx}.spt"
        aliases = [monitor_alias] if isinstance(monitor_alias, str) else monitor_alias
        if aliases is None:
            aliases = self.snn.monitor_alias
        result = {}
        for name in aliases:
            monitor = getattr(self.snn, name)
            if not isinstance(monitor, SpikeMonitorTorch):
                raise TypeError(f"{name} is not a SpikeMonitorTorch")
            result[name] = monitor.get_data()
        if not overwrite:
            save_name = utils.training_tools.RecordMethods.rename_duplicate_file(self.spike_dump_path, save_name)
        torch.save(result, self.spike_dump_path + save_name)

    @staticmethod
    def predict_policy(data):
        mean = data if isinstance(data, Tensor) else data[0]
        return torch.max(mean.reshape(1, -1), dim=-1)[-1].cpu()

    @torch.inference_mode()
    def mnn_validate_one_sample(self, idx):
        data, target = getattr(self, "dataset")[idx]
        data, target = getattr(self.train_funcs, "data2device")(data, target, self.args)
        data = self.mnn(data)
        return data, self.predict_policy(data), target

    @staticmethod
    def modify_value_by_condition(data: tuple[Tensor, Tensor], condition: str) -> tuple[Tensor, Tensor]:
        mean, covariance = data
        if "mask_mean" in condition:
            mean = torch.zeros_like(mean)
        elif "shuffle_cov" in condition:
            covariance = covariance * torch.eye(covariance.size(-1), dtype=covariance.dtype, device=covariance.device)
        elif "corr_only" in condition:
            mean = torch.zeros_like(mean)
            _, covariance = moment_functional.compute_correlation(covariance)
        elif "mean_only" in condition:
            covariance = torch.zeros_like(covariance)
        return mean, covariance

    def prepare_inputs(self, idx):
        data, _ = getattr(self, "dataset")[idx]
        (mean, covariance), _ = getattr(self.train_funcs, "data2device")(data, None, self.args)
        condition = getattr(self.args, "cov_condition", "full")
        mean, covariance = self.modify_value_by_condition((mean, covariance), condition)
        input_neurons = mean.size(-1)
        if self.input_type == "gaussian":
            std, rho = moment_functional.compute_correlation(covariance)
            if "shuffle_cov" in condition:
                rho = None
            current = GaussianCurrentSource(
                (self.num_trials, input_neurons),
                mean,
                std,
                rho,
                dt=self.dt,
                pregenerate=self.pregenerate,
                num_steps=self.num_steps,
            )
        else:
            current = PoissonSpikeSource(
                (self.num_trials, input_neurons),
                mean,
                dt=self.dt,
                pregenerate=self.pregenerate,
                num_steps=self.num_steps,
            )
        if getattr(self.args, "use_cuda", True) and torch.cuda.is_available():
            current = current.cuda(self.args.local_rank)
        return current

    @torch.inference_mode()
    def run_one_simulation(self, idx, record: bool = True, dump_spike_train: bool = False, overwrite: bool = True, **kwargs) -> None:
        inputs = self.prepare_inputs(idx)
        for _ in range(self.num_steps):
            self.snn(inputs())
        if record:
            if dump_spike_train:
                self.dump_spike_train(idx, overwrite=overwrite)
            self.snn.spike_statistic()

    @torch.inference_mode()
    def validate_one_sample(self, idx, do_reset: bool = True, dump_spike_train: bool = True, record: bool = True, print_log: bool = True, **kwargs) -> None:
        self.reset() if do_reset else self.custom_reset()
        mnn_outputs, mnn_pred, target = self.mnn_validate_one_sample(idx)
        self.run_one_simulation(idx, record=record, dump_spike_train=dump_spike_train, **kwargs)
        snn_outputs = self.snn.make_predict()
        pred = self.predict_policy(snn_outputs)
        if print_log:
            data_source = "train set" if self.train else "test set"
            print(f"{data_source}, Img idx: {idx}, target: {target}, pred: {pred}")
        self.save_result(
            idx=idx,
            mnn_output=(mnn_outputs, mnn_pred),
            target=target,
            snn_output=(snn_outputs, pred),
            running_time=self.running_time,
            dt=self.dt,
            **kwargs,
        )


def sample_poisson_spikes(freqs: Tensor, dt: float, num_neurons, num_steps, *, device=None, dtype=torch.float) -> Tensor:
    device = freqs.device if device is None else device
    shape = sample_shape(num_neurons, num_steps)
    return (torch.rand(shape, dtype=freqs.dtype, device=device) < freqs.to(device) * dt).to(dtype)


def sparse_spike_train_statistics(spike_train: Tensor, time_window: float, start_time_step: Optional[int] = None) -> tuple[Tensor, Tensor]:
    if start_time_step is not None:
        index = torch.arange(start_time_step, spike_train.size(0), device=spike_train.device)
        spike_train = torch.index_select(spike_train, dim=0, index=index)
    spike_count = torch.sparse.sum(spike_train, dim=0).to_dense().to(torch.float).T
    mean = torch.mean(spike_count, dim=-1) / time_window
    covariance = torch.cov(spike_count) / time_window
    return mean, covariance


class DirectReadoutMomentMlpToSnn(MomentMlpToSnn):
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        weight = self.predict.weight
        x = torch.matmul(weight, x.unsqueeze(-1)).squeeze(-1)
        if self.predict.bias is not None:
            x = x + self.predict.bias * self.dt
        return x


@torch.no_grad()
def snn_validation(
    checkpoint,
    save_name,
    train_funcs,
    *,
    dt: float = 1,
    num_trials=None,
    running_time: float = 100,
    resume_best: bool = False,
    pre_run: float = 50,
    do_reset: bool = True,
    init_voltage=None,
    input_type: str = "poisson",
):
    args = utils.training_tools.TempArgs()
    args.config = checkpoint + f"{save_name}_config.yaml"
    args = utils.training_tools.set_config2args(args)
    args.config = checkpoint + f"{save_name}_config.yaml"
    args.dir = checkpoint.split("/")[-2]
    total_trials = args.bs if num_trials is None else (num_trials, args.bs)
    _, test_loader = train_funcs.prepare_dataloader(args)
    simulator = MomentSnnValidator(
        args,
        train_funcs=train_funcs,
        dt=dt,
        num_trials=total_trials,
        running_time=running_time,
        MNN_MLP=DirectReadoutMomentMlpToSnn,
        resume_best=resume_best,
        init_voltage=init_voltage,
        input_type=input_type,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simulator.snn.to(device).eval()
    results = []
    targets = []
    for data, labels in test_loader:
        if do_reset:
            model.reset()
        (data, _), _ = getattr(train_funcs, "data2device")(data, None, args)
        if num_trials is None:
            num_neurons = data.size()
        else:
            num_neurons = [num_trials] + list(data.size())
            data = data.unsqueeze(0)
        if input_type == "poisson":
            current = PoissonSpikeSource(num_neurons, data, dt=dt).to(device)
        else:
            current = GaussianCurrentSource(num_neurons, data, torch.sqrt(data), dt=dt).to(device)
        case = []
        for _ in range(int(pre_run / dt)):
            model(current())
        for _ in range(int(running_time / dt)):
            case.append(model(current()).unsqueeze(0))
        results.append(torch.cat(case, dim=0).cpu())
        targets.append(labels)
    return torch.cat(results, dim=-2), torch.cat(targets, dim=0)
