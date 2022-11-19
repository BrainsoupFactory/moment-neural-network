import torch
from mnn import snn, utils

class MnistTrainFuncs(utils.training_tools.TrainProcessCollections):
    # a batch of functions to support MNN training in a variety of tasks
    def set_random_seed(self, seed):
        # this function can fix all random seed (numpy, pytorch) for better reproducibility
        return super().set_random_seed(seed)
    
    def make_model(self, model_args: dict):
        # this func will generate model, rewrite this func for your own model.
        return super().make_model(model_args)
    
    def prepare_dataloader(self, args, data_dir='./data/'):
        # rewrite this func for your dataloader, by default it will return MNIST loader
        return super().prepare_dataloader(args, data_dir)
    
    def prepare_criterion(self, args):
        # rewrite this func for your criterion, alternatively to specify config for provided criterion. 
        return super().prepare_criterion(args)
    
    def prepare_optimizer_scheduler(self, params_group, args):
        # rewrite this func for your optimizer and LR scheduler, by default it return AdamW (base on config) and None for sheduler
        return super().prepare_optimizer_scheduler(params_group, args)
    
    def specify_params_group(self, model):
        # specify which part of model to be trained, by default the whole model is trainable
        return super().specify_params_group(model)
    
    def data2device(self, data, target, args):
        # Send data to specify device (cpu or gpu), For MNN we return mean and cov by assuming data is Poisson rate.
        return super().data2device(data, target, args)
    
    def clip_model_params(self, model, args):
        # For special purpose, we want to limit model params in a specific range, by default this func does nothing.
        return super().clip_model_params(model, args)
    
    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, args):
        # A general pipeline for training
        return super().train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
    
    def validate(self, val_loader, model, criterion, args, epoch=0):
        # A general pipeline for testing
        return super().validate(val_loader, model, criterion, args, epoch)
    
    def score_function(self, output, target, *args, **kwargs):
        # Decide the score of model performance in classify task. In regression we use loss.
        return super().score_function(output, target, *args, **kwargs)
    
def train_mnist(args):
    # I has write a pipeline for training MNN, see the source code for implementing details
   utils.training_tools.general_train_pipeline(args, train_func=MnistTrainFuncs) 


class MnistSnnValidate(snn.functional.MnnSnnValidate):
    # A batch of functions to simulate snn base on corresponding MNN model
    def __init__(self, args, running_time=20, dt=0.01, num_trials=100, monitor_size=None, pregenerate=False, resume_best=True, train=False, init_vol=None, alias='', input_type='gaussian', **kwargs) -> None:
        """_summary_

        Args:
            args (argparser): the args that used to training MNN 
            running_time (int, optional): The total simulation time (ms). Defaults to 20.
            dt (float, optional): The time step interval. Defaults to 0.01.
            num_trials (int, optional): models that run in parallel. Defaults to 100.
            monitor_size (_type_, optional): deprecated. Defaults to None.
            pregenerate (bool, optional): general all inputs at begining or on the fly. Defaults to False.
            resume_best (bool, optional): use the best model in MNN training or the last epoch. Defaults to False.
            train (bool, optional): use the train or test set for validate. Defaults to False.
            init_vol (_type_, optional): initial policy for LIF neuron membrane potential. Defaults to None.
            alias (str, optional): alias for file name to store. Defaults to ''.
            input_type (str, optional): the way to generate input. Defaults to 'gaussian'.
        """
        super().__init__(args, running_time, dt, num_trials, monitor_size, pregenerate, resume_best, train, init_vol, alias, input_type, **kwargs)
    
    def extra_works(self, *args, **kwargs):
        # This func will call at the end of __init__(), by default it does nothing but you can rewrite it to support different needs 
        return super().extra_works(*args, **kwargs)
    
    def resume_config(self, args, resume_best):
        # resume config base on trained mnn config (should be found in the directory where store trained model)
        return super().resume_config(args, resume_best)
    
    def prepare_dump_dir(self):
        # set path(dump_path, spike_dump_path) that store simulated data (statistic data and raw spike train)
        return super().prepare_dump_dir()
    
    def reset(self):
        # reset whole model to initial state
        return super().reset()
    
    def custom_reset(self):
        # only reset monitors and probes, neurons' membrane potential do not change.
        return super().custom_reset()
    
    def generate_model(self):
        # define how to restore and recontrust MNN and SNN
        return super().generate_model()
    
    def save_result(self, idx, overwrite=True, probe_alias=None, reset_probe=True, **result):
        # save data
        return super().save_result(idx, overwrite, probe_alias, reset_probe, **result)

    def predict_policy(self, data):
        # decide the result of output
        return super().predict_policy(data)
    
    def generate_dataset(self):
        # generate dataset for simulation
        train_loader, test_loader = utils.training_tools.prepare_dataloader(self.args, self.args.data_dir)
        if self.train:
            setattr(self, 'dataset', train_loader.dataset)
        else:
            setattr(self, 'dataset', test_loader.dataset)
    
    def prepare_inputs(self, idx):
        # prepare input current for snn
        img, _ = self.dataset[idx]
        freqs = self.data2cuda(torch.flatten(img)).unsqueeze(0) * getattr(self.args, 'scale_factor', 1.)
        num_neurons = (self.num_trials, freqs.size(-1))
        if self.input_type == 'poisson':
            input_current = snn.base.PoissonSpikeGenerator(num_neurons=num_neurons,
            freqs=freqs, dt=self.dt, pregenerate=self.pregenerate, num_steps=self.num_steps)
        else:
            std = torch.sqrt(torch.abs(freqs))
            input_current = snn.base.GaussianCurrentGenerator(num_neurons=num_neurons,
            mean=freqs, std=std, dt=self.dt, pregenerate=self.pregenerate, num_steps=self.num_steps)
        
        input_current = input_current.cuda(self.args.local_rank)
        input_current.reset()
        return input_current
    
    @torch.inference_mode()
    # result of MNN predict, assume input is independent poisson
    def mnn_validate_one_sample(self, idx):
        img, target = self.dataset[idx]
        img = self.data2cuda(torch.flatten(img).unsqueeze(0)) * getattr(self.args, 'scale_factor', 1.)
        assert img.size() == (1, 28*28)
        mean = img
        cov = torch.diag_embed(torch.abs(mean))
        mean, cov  = self.mnn((mean, cov))
        pred = self.predict_policy((mean, cov))
        return mean, cov, pred, target
    
    @torch.inference_mode()
    def run_one_simulation(self, idx, record=True, dump_spike_train=False, overwrite=True, **kwargs):
        # pipeline for snn simulation
        return super().run_one_simulation(idx, record, dump_spike_train, overwrite, **kwargs)
    
    @torch.inference_mode()
    def validate_one_sample(self, idx, do_reset=False, print_log=False, **kwargs):
        # func to call mnn-snn simulation
        return super().validate_one_sample(idx, do_reset, print_log, **kwargs)


def mnn2snn_simulation(args):
    dt = 0.01
    input_type = 'poisson'
    num_trial = 100
    running_time = 100
    pregenerate = False
    m = MnistSnnValidate(args, running_time=running_time, dt=dt, num_trials=num_trial, 
    pregenerate=pregenerate, resume_best=False, input_type=input_type)
    for index in range(5):
        m.validate_one_sample(index, do_reset=True, dump_spike_train=True, record=True)

def main():
    config = utils.training_tools.deploy_config()
    train_mnist(config)
    mnn2snn_simulation(config)

if __name__ == '__main__':
    main()