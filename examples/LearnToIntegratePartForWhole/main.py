import torch
import numpy as np
from tqdm import tqdm
import torchvision
from mnn import mnn_core, utils, models, snn

class MotionDirectionTrain(utils.training_tools.TrainProcessCollections):
    def prepare_dataloader(self, args):
        train_loader, test_loader = utils.dataloaders.motion_direction_dataloader(args)
        return train_loader, test_loader
    
    def make_model(self, args):
        return super().make_model(args)
    
    def data2device(self, data, target, args):
        mean, cov = data
        if args.use_cuda:
            mean = mean.cuda(args.local_rank, non_blocking=True)
            cov = cov.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
        return (mean, cov), target
    

class FineGrainTrain(utils.training_tools.TrainProcessCollections):
    def prepare_dataloader(self, args):
        train_loader, test_loader = utils.dataloaders.cub_bird_dataloader(args.data_dir)
        return train_loader, test_loader
    
    def data2device(self, data, target, args):
        if args.use_cuda:
            data = data.cuda(args.local_rank, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.cuda(args.local_rank, non_blocking=True)
        return data, target
    
    def make_model(self, args):
        meta = args.MODEL['meta']
        feature_extractor: torch.nn.Module = torchvision.models.vgg16(weighs=torchvision.models.VGG16_Weights.DEFAULT)[:-1]
        for p in feature_extractor.parameters():
            p.requires_grad = False
        if meta['mlp_type'] == 'mnn_mlp':
            pool = mnn_core.nn.MnnPooling(**args.MODEL['pooling'])
            decoder = models.MnnMlp(**args.MODEL['mnn_mlp'])
        else:
            pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)), 
                                       torch.nn.Flatten())
            decoder = models.AnnMlp(**args.MODEL['mnn_mlp'])
        model = models.GeneralCnnPool(feature_extractor, pool, decoder)
        return model        

class MotionDirectionSnnValidate(snn.functional.MnnSnnValidate):
    def generate_dataset(self):
        dataset_args = self.args.DATASET
        dataset_args['rate_policy'] = self.rate_policy
        dataset_args['dt'] = self.dt
        dataset_args['duration'] = self.running_time
        dataset = utils.datasets.custom.VisualOrientationDataset(train=False, **dataset_args)
        self.train = False
        dataset.c = torch.tensor([dataset.cmax])
        dataset.theta = torch.linspace(-np.pi, np.pi, 50+2)[1:-1]
        setattr(self, 'dataset', dataset)
        self.alias = self.alias + self.rate_policy + '_'
    
    def predict_policy(self, data):
        return None
    
    def prepare_inputs(self, idx):
        i, j = np.unravel_index(idx, (len(self.dataset.c),len(self.dataset.theta)))
        theta = self.dataset.theta[j].reshape(1)
        c = self.dataset.c[i].reshape(1)
        rate = self.dataset.gen_snn_input_rate(theta, c)
        rate = self.data2cuda(rate)
        input_neuron = rate.size(-1)
        current = snn.base.InhomogeneousPoissonSpikeGenerator((self.num_trials, input_neuron), freqs=rate, dt=self.dt, pregenerate=self.pregenerate).cuda()
        current.reset()
        return current

class FineGrainSnnValidate(snn.functional.MnnSnnValidate):
    def generate_model(self):
        mnn_model = self.generate_mnn_model()
        snn_model = self.generate_snn_model()
        snn_model.load_state_dict(mnn_model.classifier.state_dict())
        snn_model.mnn2snn(dt=self.dt, batch_size=self.num_trials, monitor_size=self.monitor_size, 
        pregenerate=self.pregenerate, num_steps=self.num_steps, init_vol=self.init_vol, **getattr(self.args, 'NEURONS', {}))
        setattr(self, 'mnn', mnn_model)
        setattr(self, 'snn', snn_model)
        self.mnn.cuda(self.args.local_rank)
        self.snn.cuda(self.args.local_rank)
        self.reset()
        
    def generate_dataset(self):
        return super().generate_dataset()

    def mnn_validate_one_sample(self, idx):
        _, target = self.dataset[idx]
        return None, None, target
    
    @torch.no_grad()
    def prepare_inputs(self, idx):
        data, target = self.dataset[idx]
        data, target = self.train_funcs.data2device(data, target, self.args)
        data = data.unsqueeze(0)
        condition = self.args.save_name
        feature_map = self.mnn.feature_extractor(data)
        mean, cov = self.mnn.cov_pool(feature_map)
        std, rho = mnn_core.nn.functional.compute_correlation(cov)
        input_neuron = mean.size()[-1]
        if 'with_corr' in condition:
            rho = rho.squeeze()
        else:
            rho = None
        input_current = snn.base.GaussianCurrentGenerator(num_neurons=(self.num_trials, input_neuron), mean=mean.squeeze(),std=std.squeeze(),rho=rho, dt=self.dt, pregenerate=self.pregenerate,num_steps=self.num_steps).cuda(self.args.local_rank)
        return input_current

def train_motion_direction(args, suffix=0):
    args.save_name = 'motion_direction' + '_' + str(suffix)
    args.task_type = 'regression'
    args.config = './motion_direction.yaml'
    args = utils.training_tools.set_config2args(args)
    utils.training_tools.general_train_pipeline(args, MotionDirectionTrain)

def train_fine_grain(args, suffix=0):
    args.save_name = 'with_corr' + '_' + str(suffix)
    args.task_type = 'classification'
    args.config = './fine_grain.yaml'
    args = utils.training_tools.set_config2args(args)
    utils.training_tools.general_train_pipeline(args, FineGrainTrain)
    
    args.save_name = 'without_corr' + '_' + str(suffix)
    args.MODEL['pooling']['mask_cov'] = True
    utils.training_tools.general_train_pipeline(args, FineGrainTrain)
    
    args.save_name = 'ann' +  '_' + str(suffix)
    args.MODEL['meta']['mlp_type'] = 'ann_mlp'
    utils.training_tools.general_train_pipeline(args, FineGrainTrain)

def motion_direction_snn_simulation(args, suffix=0, **kwargs):
    args.save_name = 'motion_direction' + '_' + str(suffix)
    input_type = 'poisson'
    running_time = int(2 * np.pi * 200)
    pregenerate = False
    alias = ''
    dt = 0.1
    init_vol = 'uniform'
    num_trial = 500
    policy = 'continous'
    train_funcs = MotionDirectionTrain()
    m = MotionDirectionSnnValidate(args, running_time=running_time, num_trials=num_trial, dt=dt, alias=alias,
        pregenerate=pregenerate, init_vol=init_vol, rate_policy=policy, input_type=input_type, train_funcs=train_funcs, **kwargs)
    for j in tqdm(range(len(m))):
        m.validate_one_sample(j, do_reset=False, dump_spike_train=True, record=True)

def fine_grain_snn_simulation(args, suffix=0,  **kwargs):
    names = ['with_corr', 'without_corr']
    input_type = 'gaussian'
    running_time = 150
    pregenerate = True
    dt = 0.1
    init_vol = 'uniform'
    num_trial = 100
    train = False
    alias = ''
    for name in names:
        args.save_name = name + '_' + str(suffix)
        m = FineGrainSnnValidate(args, running_time=running_time, num_trials=num_trial, dt=dt, alias=alias, pregenerate=pregenerate, init_vol=init_vol, input_type=input_type, train=train, train_funcs=FineGrainTrain(), **kwargs)
        for j in tqdm(range(len(m))):
            m.validate_one_sample(j, do_reset=True, dump_spike_train=True, record=True)


if __name__ == '__main__':
    config = utils.training_tools.deploy_config()
    trial = 1
    for i in range(trial):
        train_motion_direction(config, suffix=i)
        train_fine_grain(config, suffix=i)
        motion_direction_snn_simulation(config, suffix=i)
        fine_grain_snn_simulation(config, suffix=i)
    