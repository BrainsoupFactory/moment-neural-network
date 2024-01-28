import torch
from mnn import snn, utils

class MnistTrainFuncs(utils.training_tools.TrainProcessCollections):
    """
    The general pipeline to construct MNN for training.
    """
    def set_random_seed(self, seed):
        """
        This function can fix all random seed (numpy, pytorch) for better reproducibility.
        By default, the seed is None. You can configure the random seed in config.yaml by adding a new item "seed: YOU SEED".
        Alternatively, you can just set args.seed = YOU SEED in you code before calling "utils.training_tools.general_train_pipeline()"
        """
        return super().set_random_seed(seed)
    
    def make_model(self, model_args: dict):
        """
        This function will make the MNN model based on you model configuration, which is defined in config.yaml (MODEL).
        Be free to rewrite the function for building you model.
        """
        model = super().make_model(model_args)
        return model
    
    def prepare_dataloader(self, args):
        """
        This function will create the dataloaders for training and testing.
        You can rewrite the function to return the dataloads for your own purpose.
        Note that by default the directory of data should be found in the './data/' but you can modify it setting the value of args.data_dir or specify in config.yaml before training.
        Make sure when calling 'enumerate(dataloader)', the outputs  the form of (idx, data, target).
        """
        train_loader, val_loader = super().prepare_dataloader(args)
        return train_loader, val_loader
    
    def prepare_criterion(self, args):
        """
        This function will create the criterion for computing loss.
        We prepared a family of criterions for MNN (see mnn_core.nn.criterion).
        By default, we use CrossEntropyOnMean for classification.
        You can specify criterion by modifying the 'CRITERION' in config.yaml.
        You can alway create you criterion by rewriting this function, as long as it can be used for computing the loss between the model outputs and the target.
        """
        criterion = super().prepare_criterion(args)
        return criterion
    
    def prepare_optimizer_scheduler(self, params_group, args):
        """
        This function will create optimizer and scheduler for training.
        By default, we use AdamW as optimizer and None for scheduler.
        If you use a learning scheduler, it will call scheduler.step() when end the training of one epoch.
        You can specify the optimizer in config.yaml (OPTIMIZER) or rewrite the function if you like.
        The params_group contains the model parameters for training (see follow)
        """
        optimizer, scheduler = super().prepare_optimizer_scheduler(params_group, args)
        return optimizer, scheduler
    
    def specify_params_group(self, model):
        # specify which part of model to be trained, by default the whole model is trainable
        return super().specify_params_group(model)
    
    def data2device(self, data, target, args):
        """
        Send data to specify device (cpu or gpu), For MNN we return mean and cov by assuming data is Poisson rate.
        Namely, the input data is a batch of images, we flatten image and create the covariance matrix base on it.
        The return data is a tuple (mean, cov).
        You can rewrite the function to define the procedure of data preprocessing.
        """
        data, target =  super().data2device(data, target, args)
        return data, target
    
    def clip_model_params(self, model, args):
        """
        For special purpose, we want to limit model params in a specific range, by default this func does nothing.
        This function will be called at each iteration.
        """
        return super().clip_model_params(model, args)
    
    def compute_model_output(self, model, inputs, args=None):
        """
        This function will be called during training and validation to compute the model output.
        By default it simple be 'output = model(innputs)', where inputs are the returned data of 'data2device()' and the output is a tuple(mean, cov).
        You can rewrite the function to define you computation pipeline. For example, for RNN you may want to gather a list of output by for loop or the internal representation of the model is necessary for task or loss function.
        """
        output = super().compute_model_output(model, inputs, args)
        return output
    
    def compute_loss(self, output, target, criterion, model=None, args=None, inputs=None):
        """
        This function compute the loss.
        By default it simply be 'loss = criterion(output, target)'.
        The model you use, args and inputs will pass into this function in this pipeline.
        You can modify the function for your own need.
        For example, auto encoder requires input data for computing loss and you may want to limit model parameters by adding them into loss.
        The returned loss will call the backward() to compute gradients.
        """
        loss = super().compute_loss(output, target, criterion, model, args, inputs)
        return loss
    
    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, args):
        # A general pipeline for training
        return super().train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
    
    def validate(self, val_loader, model, criterion, args, epoch=0):
        # A general pipeline for testing
        return super().validate(val_loader, model, criterion, args, epoch)
    
    def score_function(self, output, target, *args, **kwargs):
        """
        Decide the score of model performance in classify task. 
        In regression this function will ignore.
        The task type is determined by args.task_type.
        You can specify it in config.yaml or change it before pass args into pipeline.
        By default, args.task_type = 'classify'
        """
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