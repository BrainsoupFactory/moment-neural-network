import torch
from mnn import snn, utils


class MnistSnnValidate(snn.functional.MnnSnnValidate):
    def generate_dataset(self):
        train_loader, test_loader = utils.training_tools.prepare_dataloader(self.args, self.args.data_dir)
        if self.train:
            setattr(self, 'dataset', train_loader.dataset)
        else:
            setattr(self, 'dataset', test_loader.dataset)
    
    @torch.inference_mode()
    def mnn_validate_one_sample(self, idx):
        img, target = self.dataset[idx]
        img = self.data2cuda(torch.flatten(img).unsqueeze(0)) * getattr(self.args, 'scale_factor', 1.)
        assert img.size() == (1, 28*28)
        mean = img
        cov = torch.diag_embed(torch.abs(mean))
        mean, cov  = self.mnn((mean, cov))
        pred = self.predict_policy((mean, cov))
        return mean, cov, pred, target

    def prepare_inputs(self, idx):
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


def train_mnist(args):
   utils.training_tools.general_train_pipeline(args)

def mnn2snn_simulation(args, index=0):
    dt = 0.01
    input_type = 'poisson'
    num_trial = 100
    running_time = 100
    pregenerate = False
    m = MnistSnnValidate(args, running_time=running_time, dt=dt, num_trials=num_trial, 
    pregenerate=pregenerate, resume_best=False, input_type=input_type)
    m.validate_one_sample(index, do_reset=True, dump_spike_train=True, record=True)

def main():
    config = utils.training_tools.deploy_config()
    train_mnist(config)
    mnn2snn_simulation(config)

if __name__ == '__main__':
    main()