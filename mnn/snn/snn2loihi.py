import os
import torch
import nxsdk.api.n2a as nx
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from n2_apps.modules.slayer import src as nxSlayer
from n2_apps.modules.slayer.src.slayer2loihi import Slayer2Loihi
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
from typing import Optional
from attrdict import AttrDict as Event
import argparse
import time
from collections import defaultdict
import random
import pandas as pd
import torchvision


_SPECIAL_ARGS = ['DATASET', 'DATALOADER', 'LOIHI']

def prepare_args(parser):
    args = parser.parse_args()
    args.save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
    if args.config is not None:
        args = set_config2args(args)
    os.environ['SLURM'] = args.SLURM
    os.environ['PARTITION'] = args.PARTITION
    os.environ['BOARD'] = args.BOARD
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']= args.PROTOCOL
    if args.seed is not None:
        seed_everything(args.seed)
    return args

def reset_special_args(args):
    for key in _SPECIAL_ARGS:
        setattr(args, key, None)
    return args

def set_config2args(args):
    cfg = read_yaml_config(args.config)
    args = reset_special_args(args)
    for key, value in cfg.items():
        setattr(args, key, value)
    return args

def read_yaml_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_transforms_compose(params):
    order = params['aug_order']
    transform_list = []
    for key in order:
        if key == 'ToTensor':
            transform_list.append(torchvision.transforms.ToTensor())
        else:
            transform_list.append(getattr(torchvision.transforms, key)(**params[key]))
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def deploy_config(is_parse: bool = True):
    parser = argparse.ArgumentParser(description='SNN to Loihi converting template')
    # my custom parameters
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='YAML config file')
    parser.add_argument('--bs', default=50, type=int, help='batch size')
    parser.add_argument('--dir', default='mnist', type=str, help='dir path that used to save checkpoint')
    parser.add_argument('--trials', default=1, type=int, help='# to call the functions')
    parser.add_argument('--dataset_type', default='poisson', type=str, help='methods used in data loaders')
    parser.add_argument('--dataset', default='mnist', type=str, help='type of dataset')
    parser.add_argument('--data_dir', default='./data/', type=str, help='directory path of dataset')
    parser.add_argument('--eps', default=0.1, type=float, help='eps')
    parser.add_argument('--save_name', default='mnn_net', type=str, help='alias to save net')

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for random generator. ')
    parser.add_argument('--workers', default=1, type=int,
                        help='num workers for dataloader. ')

    # environment varibles
    parser.add_argument('--PARTITION', default='nahuku32', type=str, help='specify PARTITION')
    parser.add_argument('--BOARD', default='ncl-ext-ghrd-02', type=str, help='specify BORARD')
    parser.add_argument('--SLURM', default='1', type=str, help='specify SLURM')
    parser.add_argument('--PROTOCOL', default='python', type=str, help='specify PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')
    if is_parse:
        args = prepare_args(parser)
        return args
    else:
        return parser

def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class FuncForLoihiCollections:
    def __init__(self, args, num_samples, dt=1., sample_length=100, train=False, alias='', 
                 loihi_max_weight=254, mnn_threshold=20, L=1/20, current_decay=1., refrac=5., use_membrane=False, first_spike_threshold=None, **kwargs):
        self.args = args
        self.num_samples = num_samples
        self.dt = dt
        self.sample_length = sample_length
        self.train = train
        self.alias = alias
        self.loihi_max_weight = loihi_max_weight
        self.mnn_threshold = mnn_threshold
        self.use_membrane = use_membrane
        self.L = L
        self.voltage_decay = 1 - np.exp(- L* dt)
        self.current_decay = current_decay
        self.model_dir = getattr(self.args, 'dump_path', './checkpoint/') + self.args.dir + '/'
        self.refrac = refrac
        self.simulation_results = defaultdict(list)
        self.network_probes = dict()
        self.first_spike_threshold = first_spike_threshold

        self.energy_probe_properties = ['allTimeStamps', 'binSize', 'bufferSize', 'endTimeStamps', 'hostTimePerTimeStep',
        'learningTimePerTimeStep', 'managementTimePerTimeStep', 'numStepsMax', 'numStepsRan', 'spikingTimePerTimeStep',
        'startTimeStamps', 'tEnd', 'tStart', 'timeSteps', 'timeUnits', 'totalExecutionTime', 'totalHostTime', 'totalLearningTime',
        'totalManagementTime', 'totalPreLearningMgmtTime', 'totalSpikingTime', 'totalTimePerTimeStep', 'EnergyPhase', 'EnergyType',
        'energyUnits', 'hostPhaseEnergyPerTimeStep', 'learningPhaseEnergyPerTimeStep', 'managementPhaseEnergyPerTimeStep',
        'power', 'powerUnits', 'powerVdd', 'powerVdda', 'powerVddio', 'powerVddm', 'preLearnManagementPhaseEnergyPerTimeStep', 
        'rawPowerTimeStamps', 'rawPowerTotal', 'rawPowerVdd', 'rawPowerVdda', 'rawPowerVddm', 'spikingPhaseEnergyPerTimeStep', 'time',
        'totalEnergy', 'totalEnergyPerTimeStep', 'totalHostPhaseEnergy', 'totalLearningPhaseEnergy', 'totalManagementPhaseEnergy', 
        'totalPreLearnManagementPhaseEnergy', 'totalSpikingPhaseEnergy']

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.prepare_dump_dir()
        self.default_config(getattr(args, 'LOIHI', None))
        info = '\n#################{}#################'.format(time.ctime())
        self.record_info(info)
        self.extra_works()

    def prepare_dump_dir(self):
        save_path = getattr(self.args, 'dump_path', './checkpoint/') + self.args.dir + '/{}_loihi_result/'.format(self.args.save_name)
        setattr(self, 'loihi_dump_path', save_path)
        self.make_dir(save_path)
        log_path = save_path + 'log.txt'
        setattr(self, 'loihi_log_path', log_path)
    
    def extra_works(self, *args, **kwargs):
        pass

    def renew_log(self):
        info = '\n#################{}#################'.format(time.ctime())
        self.record_info(info, mode='w+')

    @staticmethod
    def make_dir(dir_path: str):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def mean_estimation(error_array, alpha=0.95):
        x, y = error_array.shape
        mean_error,lower_error,upper_error=np.zeros(x),np.zeros(x),np.zeros(x)
        for i in range(x):
            res_mean,_,_=stats.bayes_mvs(error_array[i,:],alpha=alpha)
            mean_error[i]=res_mean[0]
            lower_error[i]=res_mean[1][0]
            upper_error[i]=res_mean[1][1]
        return mean_error, lower_error, upper_error

    @staticmethod
    def record_hyper_parameter(path: str, name: str, **kwargs):
        with open(path + '{:}_config.yaml'.format(name), 'w') as f:
            yaml.dump(kwargs, f, default_flow_style=False)
    
    def record_info(self, info, print_info=True, suffix='\n', mode='a+'):
        with open(self.loihi_log_path, encoding='utf-8', mode=mode) as f:
            f.write(info + suffix)
            if print_info:
                print(info)

    @staticmethod
    def rename_duplicate_file(file_path, file_name, suffix_pos=None):
        temp = file_name
        i = 1
        while os.path.exists(file_path + temp):
            if suffix_pos is None:
                name, suffix = file_name.split('.')
            else:
                name = file_name[:suffix_pos-1]
                suffix = file_name[suffix_pos:]
            name += '({})'.format(i)
            i += 1
            temp = name + '.{}'.format(suffix)
        return temp
    
    def save_objects(self, obj, save_path, save_name, suffix_pos=None, overwrite=True):
        if not overwrite:
            save_name = self.rename_duplicate_file(save_path, save_name, suffix_pos=suffix_pos)
        torch.save(obj, save_path + save_name)
    
    def params2loihi(self, snn_params: dict, debug=False, **kwargs):
        loihi_params = {}
        loihi_thresh = []
        for i in range(int(len(snn_params)/ 2)):
            key = 'fc{}.weight'.format(i)
            value = snn_params[key]
            scale_factor = torch.floor(self.loihi_max_weight / torch.max(torch.abs(value)) * self.mnn_threshold)
            loihi_thresh.append(scale_factor.item())
            scale_factor = scale_factor / self.mnn_threshold
            weight = (value * scale_factor).int()
            weight  = (weight >> 1) << 1
            assert torch.max(torch.abs(weight)) <= self.loihi_max_weight
            loihi_params[key] = weight.detach()
            key = 'fc{}.bias'.format(i)
            bias = snn_params[key]
            if bias is not None:
                #self.voltage_decay = 1 - np.exp(- L* dt)
                bias = (bias * scale_factor * self.voltage_decay / self.L).int().detach()
            loihi_params[key] = bias
        if debug:
            info = 'After conversion to Loihi, parameters are in the following range: '
            self.record_info(info)
            for key, value in loihi_params.items():
                if value is not None:
                    info = '{}: min={}, max={}'.format(key, torch.min(value), torch.max(value))
                else:
                    info = '{} is None!'.format(key)
                self.record_info(info)
        setattr(self, 'loihi_params', loihi_params)
        setattr(self, 'loihi_thresh', loihi_thresh)
        return loihi_params, loihi_thresh
    
    def loihi2net(self, **kwargs):
        config = self.loihi_config
        loihi_params = self.loihi_params
        structure = config['net_structure']
        num_class = config['num_class']
        with h5py.File(config['loihi_net'], 'w') as f:
            layer = f.create_group('layer')
            for i, dims in enumerate(structure):
                sub_layer = layer.create_group(str(i))
                if i == 0:
                    sub_layer.create_dataset('shape', data=np.array([1, 1, dims], dtype='int64'))
                    sub_layer.create_dataset('type', data=np.array(['input'.encode('ascii')]))
                else:
                    sub_layer.create_dataset('inFeatures', data=structure[i-1])
                    sub_layer.create_dataset('outFeatures', data=dims)
                    sub_layer.create_dataset('shape', data=np.array([dims, 1, 1], dtype='int64'))
                    sub_layer.create_dataset('type', data=np.array(['dense'.encode('ascii')]))
                    sub_layer.create_dataset('weight', data=loihi_params['fc{}.weight'.format(i-1)].numpy(), dtype='<f4')
                    bias = loihi_params['fc{}.bias'.format(i-1)]
                    if isinstance(bias, torch.Tensor):
                        bias = bias.numpy()
                    if bias is not None:
                        sub_layer.create_dataset('bias', data=bias)
                    
                    neuron = sub_layer.create_group('neuron')
                    neuron.create_dataset('iDecay', data=int(self.current_decay * 2 ** 12), dtype='int64')
                    neuron.create_dataset('refDelay', data=config['refrac'], dtype='int64')
                    neuron.create_dataset('vDecay', data=int(self.voltage_decay * 2 ** 12), dtype='int64')
                    neuron.create_dataset('vThMant', data=int(self.loihi_thresh[i-1]), dtype='int64')
                    neuron.create_dataset('wgtExp', data=6, dtype='int64')

            sub_layer = layer.create_group(str(i+1))
            sub_layer.create_dataset('inFeatures', data=dims)
            sub_layer.create_dataset('outFeatures', data=num_class)
            sub_layer.create_dataset('shape', data=np.array([num_class, 1, 1]), dtype='int64')
            sub_layer.create_dataset('type', data=np.array(['dense'.encode('ascii')]))
            sub_layer.create_dataset('weight', data=loihi_params['fc{}.weight'.format(i)].numpy(), dtype='<f4')
            bias = loihi_params['fc{}.bias'.format(i)]
            if isinstance(bias, torch.Tensor):
                bias = bias.numpy()
            if bias is not None:
                sub_layer.create_dataset('bias', data=bias)
            
            neuron = sub_layer.create_group('neuron')
            neuron.create_dataset('iDecay', data=int(self.current_decay * 2 ** 12), dtype='int64')
            if self.first_spike_threshold is None:
                neuron.create_dataset('refDelay', data=config['refrac'], dtype='int64')
            else:
                neuron.create_dataset('refDelay', data=1, dtype='int64')
            if self.use_membrane:
                neuron.create_dataset('vDecay', data=int(2 ** 12), dtype='int64')
                neuron.create_dataset('vThMant', data=int(2 ** 12), dtype='int64')
            else:
                if self.first_spike_threshold is None:
                    neuron.create_dataset('vDecay', data=int(self.voltage_decay * 2 ** 12), dtype='int64')
                    neuron.create_dataset('vThMant', data=int(self.loihi_thresh[i]), dtype='int64')
                else:
                    neuron.create_dataset('vDecay', data=int(0), dtype='int64')
                    neuron.create_dataset('vThMant', data=int(self.first_spike_threshold), dtype='int64')

            simulation = f.create_group('simulation')
            simulation.create_dataset('Ts', data=1, dtype='int64')
            simulation.create_dataset('tSample', data=config['simulation_time'], dtype='int64')

    def make_loihi_net(self, corenum=1):
        config = self.loihi_config
        compartmentPerCore = config['numCompartmentPerCore']
        structure = config['net_structure']
        num_class = config['num_class']
        
        net = nx.NxNet()
        input_spec = dict()
        input_spec["sizeX"] = config['input_dims']
        input_spec["sizeY"] = 1
        input_spec["sizeC"] = 1
        
        input_layer, input_conn_group, corenum = Slayer2Loihi.inputLayer(net, input_spec, corenum, compartmentPerCore)
        
        loihi_params: dict = self.loihi_params
        pre_layer = input_layer
        for i in range(len(structure) - 1):
            full_spec = dict()
            full_spec['weight'] = loihi_params['fc{}.weight'.format(i)].numpy()
            bias = loihi_params['fc{}.bias'.format(i)]
            if isinstance(bias, torch.Tensor):
                bias = bias.numpy()
            full_spec['bias'] = bias
            full_spec['dim'] = full_spec['weight'].shape[0]
            full_spec['compProto'] = nx.CompartmentPrototype(vThMant=int(self.loihi_thresh[i]),
                                                            refractoryDelay=int(config['refrac']),
                                                            compartmentVoltageDecay=int(self.voltage_decay * 2 ** 12),
                                                            compartmentCurrentDecay=int(self.current_decay * 2 ** 12))
            post_layer, corenum = Slayer2Loihi.fullLayer(pre_layer, full_spec, corenum, compartmentPerCore)
            if getattr(self, 'set_fc{}_probe'.format(i), False):
                self.network_probes['fc{}_probe'.format(i)] = self.set_probes(post_layer)
                
            pre_layer = post_layer
        
        i += 1
        full_spec = dict()
        full_spec['weight'] = loihi_params['fc{}.weight'.format(i)].numpy()
        bias = loihi_params['fc{}.bias'.format(i)]
        if isinstance(bias, torch.Tensor):
            bias = bias.numpy()
        full_spec['bias'] = bias
        full_spec['dim'] = num_class
        voltage_decay = 1 if self.use_membrane else self.voltage_decay
        vThMant = int(2 ** 12) if self.use_membrane else int(self.loihi_thresh[i])
        t_ref = int(config['refrac'])
        if not self.use_membrane and self.first_spike_threshold is not None:
            #voltage_decay = 0
            vThMant = self.first_spike_threshold
            t_ref = 1
        full_spec['compProto'] = nx.CompartmentPrototype(vThMant=vThMant,
                                                        refractoryDelay=t_ref,
                                                        compartmentVoltageDecay=int(voltage_decay * 2 ** 12),
                                                        compartmentCurrentDecay=int(self.current_decay * 2 ** 12))
        post_layer, corenum = Slayer2Loihi.fullLayer(pre_layer, full_spec, corenum, compartmentPerCore)  
        pre_layer = post_layer

        output_layer = post_layer
        return net, (input_layer, input_conn_group), output_layer, corenum

    def model_compiler(self, corenum=1):
        start = time.time()
        net, (input_layer, input_conn_group), output_layer, corenum = self.make_loihi_net(corenum=corenum)
        output_probes = self.set_output_probes(output_layer)
        compiler = nx.N2Compiler()
        board = compiler.compile(net)
        info = "BOARD DONE, cost: {}".format(time.time() - start)
        self.record_info(info)
        return (input_layer, input_conn_group), (output_layer, output_probes, corenum), board
    
    def set_probes(self, output_layer):
        u_probe, v_probe, s_probe = output_layer.probe([
            nx.ProbeParameter.COMPARTMENT_CURRENT,
            nx.ProbeParameter.COMPARTMENT_VOLTAGE,
            nx.ProbeParameter.SPIKE])
        return {'current': u_probe, 'voltage': v_probe, 'spike': s_probe}
    
    def set_output_probes(self, output_layer):
        if self.use_membrane:
            v_probe, = output_layer.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE])
            probe = {'voltage': v_probe}
        else:
            s_probe, = output_layer.probe([nx.ProbeParameter.SPIKE])
            probe = {'spike': s_probe}
        return probe
    
    def dump_probe_data(self, probe, end_steps, alias='output', overwrite=False, store_sparse=False):
        dump_path = self.loihi_dump_path
        output = probe.data[:, :end_steps].T
        sample_length = int(self.sample_length / self.dt)
        num_samples = self.num_samples
        output = np.reshape(output, (num_samples, sample_length, -1))
        
        save_name = '{}.tensor'.format(alias)
        if not overwrite:
            save_name = self.rename_duplicate_file(dump_path, save_name)
        output = torch.from_numpy(output)
        if store_sparse:
            output = output.to_sparse()
        torch.save(output, dump_path + save_name)
        return output

    def batch_reshape(self, new_shape, *args):
        temp = []
        for d in args:
            temp.append(d.reshape(new_shape))
        return temp
    
    def predict_accuracy(self, probe,end_steps, labels):
        S_output = probe.data[:, :end_steps].T
        sample_length = int(self.sample_length / self.dt)
        num_samples = self.num_samples
        S_output = np.reshape(S_output, (num_samples, sample_length, -1))
        S_output = np.sum(S_output, axis=1)
        num_samples = S_output.shape[0]
        results = np.argmax(S_output, axis=1)
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        accuracy = np.sum(results == labels) / num_samples
        self.simulation_results['accuracy'].append(accuracy)
        return accuracy

    def run_simulation(self, dataset, input_componets, output_compoents, 
                    board, regenerateCoreAxon=True, numSnips=1, spikesPerPacket=2048, save_probe_data=False, overwrite=True):
        start = time.time()
        dt = self.dt
        config = self.loihi_config
        num_samples = len(dataset)
        sample_length = int(config['simulation_time'] * config['batch_size'] / dt)
        input_layer, input_conn_group = input_componets
        output_layer, output_probes, corenum = output_compoents
        
        Slayer2Loihi.writeHeader(output_layer, spikesPerPacket, sample_length)
        spikeChannels, core, axon = Slayer2Loihi.prepSpikeInjection(input_conn_group, board, spikesPerPacket, 
                                                                    sample_length, numSnips, regenerateCoreAxon)
        end = time.time()
        info = "prep injection done, cost: {:.3f}".format(end - start)
        self.record_info(info)

        
        spikeData, numSteps = Slayer2Loihi.prepSpikeData(core, axon, spikesPerPacket, input_layer,
                                                        dataset, num_samples, sample_length, numSnips)
        #numClasses = config['num_class']
        #spikeCntrChannel = Slayer2Loihi.prepSpikeCounter(board, num_samples, numClasses, int(corenum))
       
        info = "spike injection done, cost: {:.3f}".format(time.time() - end)
        self.record_info(info)
        
        board.start()
        board.run(numSteps, aSync=True)
        Slayer2Loihi.sendSpikeData(spikeData, spikeChannels, spikesPerPacket)
        board.finishRun()
        board.disconnect()
        
        # end_steps is necessary because there is a 1000 delay for a full run  
        end_steps = num_samples * sample_length
        if save_probe_data:
            self.save_probes(output_probes, end_steps, overwrite=overwrite)
        predict_probe = output_probes['voltage'] if self.use_membrane else output_probes['spike']
        accuracy = self.predict_accuracy(predict_probe, end_steps, dataset.labels)
        self.summary_simulation_result(accuracy, start)
        return config
    
    def summary_simulation_result(self, accuracy, start_time):
        info = 'Time cost for running simulation: {:.3f}'.format(time.time() - start_time)
        self.record_info(info)
        info = "Accuracy is {:.2f}% ".format(accuracy * 100)
        self.record_info(info)
        info = '###########################################'
        self.record_info(info)

    def save_probes(self, probe, end_steps, overwrite=True):
        for key, value in probe.items():
            alias = '{}output_{}'.format(self.alias, key)
            store_sparse = True if key == 'spike' else False
            self.dump_probe_data(value, end_steps, overwrite=overwrite, alias=alias, store_sparse=store_sparse)
        for layer_name, probes in self.network_probes.items():
            for key, value in probes.items():
                alias = '{}{}_{}'.format(self.alias, layer_name,key)
                store_sparse = True if key == 'spike' else False
                self.dump_probe_data(value, end_steps, overwrite=overwrite, alias=alias, store_sparse=store_sparse)
        
    def loihi_run_benchmarking(self, dataset, streamOnce=False, probeEnergy=False):
        config = self.loihi_config
        numChips = config['numChips']
        numLmts = config['numLmts']
        bufferSize = config['bufferSize']
        binSize = config['binSize']
        slack = config['slack']
        tStart = bufferSize * binSize + slack
        tEnd = tStart + bufferSize * binSize
        # just make sure the tEnd > run time and the recoding time (run time - tStart) < bufferSize * binSize
        
        net = nxSlayer.benchmark.MultiChipNetwork(config['loihi_net'])
        net.create(numChips=numChips, streamOnce=streamOnce)
        net.compile(config['boardName'], regenerateBoard=True)
        net.setupIO(dataset, numLmts=numLmts, blankTime=0)
        
        if probeEnergy is True:
            eProbe = net.board.probe(
                probeType=ProbeParameter.ENERGY,
                probeCondition=PerformanceProbeCondition(
                    tStart=tStart,
                    tEnd=tEnd,
                    bufferSize=bufferSize,
                    binSize=binSize,
                )
            )
            info = 'tStart: {}, tEnd: {}, bufferSize:{}, binSize: {}, run: {}, numChips:{}, numLmts:{}, net.numSteps:{}'.format(
                tStart, tEnd, bufferSize, binSize, tEnd, numChips, numLmts, net.net.numSteps
            )
            self.record_info(info)
            results = net.run(tEnd - slack)
            return net, eProbe, results
        else:
            results = net.run()
            labels = dataset.labels
            accuracy = nxSlayer.s2l.checkAccuracy(labels, results)
            print('Final accuracy is {:.2f}%'.format(accuracy * 100))
            return net, accuracy, results

    def save_performance_profiling(self, net, eProbe, dataset, probe_properties=None):
        config = self.loihi_config
        dynamicPower = net.board.energyTimeMonitor.powerProfileStats.power['dynamic'] / config['numChips'] / net.numCopiesPerChip
        timePerTick = net.board.energyTimeMonitor.powerProfileStats.timePerTimestep
        
        dynamicEnergyPerSample = dynamicPower * timePerTick * dataset.numSteps / 1000

        performance = {
            'Dynamic power': float(dynamicPower),
            'Dynamic power unit': 'mW',
            'Time per tick': float(timePerTick),
            'Ticks / sample': int(dataset.numSteps),
            'Latency per sample':  float(timePerTick) * int(dataset.numSteps),
            'Tim per tick unit': 'us',
            'Dynamic energy Per Sample': float(dynamicEnergyPerSample),
            'Dynamic energy Per Sample unit': 'uJ',
        }
        dump_path = self.loihi_dump_path
        self.record_hyper_parameter(dump_path, self.alias + 'energy_consuming_report', **performance)
        for key, value in performance.items():
            info = '{}: {}'.format(key, value)
            self.record_info(info)
        
        if probe_properties is None:
            probe_properties = self.energy_probe_properties
        
        probe_data = {}
        for key in probe_properties:
            probe_data[key] = getattr(eProbe, key)
        
        self.save_objects(probe_data, dump_path, self.alias + 'probe_data.bin')
        
        fig, ax = prepare_fig_axs()
        ax = plot_execution_time(ax, probe_data)
        save_fig(fig, dump_path, '{}execution_time'.format(self.alias))
        
        fig, ax = prepare_fig_axs()
        ax = plot_power(ax, probe_data)
        save_fig(fig, dump_path, '{}power_per_timestep'.format(self.alias))
        
    
    def default_config(self, loihi_config: Optional[dict] = None):
        mnn_config_path = self.model_dir + '{}_config.yaml'.format(self.args.save_name)
        snn_params_path = self.model_dir + '{}.pth'.format(self.args.save_name)
        dump_path = self.loihi_dump_path
        with open(mnn_config_path, 'r') as f:
            mnn_config:dict = yaml.safe_load(f)
        
        config = {}
        config.update(mnn_config)
        
        config.update({
            'simulation_time': self.sample_length,
            'tau_v': 1 / self.voltage_decay,
            'tau_u': 1. / self.current_decay,
            'refrac': self.refrac,
            'thresh': self.mnn_threshold,
            'maximum_input_rate': 0.1,
            'sample_per_batch': 1,

            'boardName': dump_path + 'mnistBenchmark',
            'numChips': 2,
            'numLmts': 3,
            'numCompartmentPerCore': 100,
            'bufferSize': 1024,
            'binSize': 16,
            'slack': 100,
            'sample_size': self.num_samples,
            'batch_size': 1,
        })
        if loihi_config is not None:
            config.update(loihi_config)
        snn_params = torch.load(snn_params_path, map_location='cpu')
        loihi_params, loihi_thresh = self.params2loihi(snn_params=snn_params)
        
        mnn_type = mnn_config['MODEL']['meta']['mlp_type']
        net_config = mnn_config['MODEL'][mnn_type]
        config['input_dims'] = net_config['structure'][0]
        config['net_structure'] = net_config['structure']
        config['num_class'] = net_config['num_class']
        self.make_dir(dump_path)
        config['dump_path'] = dump_path
        loihi_net_path = dump_path + 'loihi_net.h5'
        config['loihi_net'] = loihi_net_path
        setattr(self, 'loihi_config', config)
        return config
    
    def prepare_dataset(self, *args, **kwargs):
        raise NotImplementedError
    
    def set_enviroment_variable(self):
        os.environ['SLURM'] = '1'
        os.environ['PARTITION'] = "nahuku32"
        os.environ['BOARD'] = "ncl-ext-ghrd-01"
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']= 'python'
    
    def simulation_pipeline(self, dataset, corenum=1,**kwargs):
        input_componets, output_compoents, board = self.model_compiler(corenum=corenum)
        _ = self.run_simulation(dataset, input_componets, output_compoents, board, **kwargs)

class PoissonSpikeDataset:
    def __init__(self, data, labels, sampleLength=100, dt=1, scaleFactor=0.1) -> None:
        """
        The data has the shape [sample_size, batch_size, input_dim], which is more efficient in loihi simulation (run multiple cases in one simulation)
        where labels is a 2D array with shape [sample_size, batch_size].
        """
        self.data = data
        self.labels = labels
        self.sampleLength = sampleLength
        self.dt = dt
        self.numSteps = int(sampleLength / dt)
        self.scaleFactor = scaleFactor
        self.rng = np.random.default_rng()
    
    def __getitem__(self, index: int):
        freqs = self.data[index]
        batch_size, dims = freqs.shape
        spikes = np.greater(freqs.reshape(batch_size, 1, dims) * self.dt, self.rng.uniform(low=0., high=1., size=(batch_size, self.numSteps, dims))).reshape(-1, dims)
        spike_time, neuron_index = np.where(spikes > 0)
        event = Event()
        event.x = np.ones_like(neuron_index)
        event.y = neuron_index - 1
        event.t = spike_time - 2
        event.p = np.zeros_like(neuron_index)
        return event
    
    def __len__(self):
        return len(self.labels)

def prepare_fig_axs(nrows: int = 1, ncols: int = 1, flatten=True, **kwargs):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if flatten and nrows * ncols > 1:
        axs = axs.reshape(-1)
    return fig, axs

def acc_with_time(spike_train, labels):
    spike_train = torch.max(torch.cumsum(spike_train.to_dense(), dim=1), dim=-1)[1]
    samples, time_step = spike_train.size()
    targets = labels.reshape(samples, 1).expand_as(spike_train)
    acc = torch.sum(spike_train == targets, dim=0) / samples
    return acc

class SpikeTrainPorpertyAnalysis:
    def __init__(self, spike_train, labels) -> None:
        self.spike_train = spike_train
        self.labels = labels
    
    def plot_first_nspike(self, ax, nspike=1,**kwargs):
        spike_count = torch.cumsum(torch.sum(self.spike_train.to_dense(), dim=-1), dim=-1) * 1.
        first_spike = torch.argmax((spike_count >= nspike) * 1., dim=1)
        sns.histplot(first_spike.numpy(), ax=ax, **kwargs)
        ax.set_xlabel('time steps')
        ax.set_title('First spike timing CDF')
        return ax
    
    def plot_spike_count_hist(self, ax, **kwargs):
        total_spike = torch.sum(self.spike_train.to_dense(), dim=(1, 2))
        sns.histplot(total_spike.numpy(), bins=int(torch.max(total_spike).item() + 1), ax=ax, discrete=True, **kwargs)
        ax.set_xlabel('spike count')
        ax.set_title('total spike after 100 time steps')
        return ax
    
    def plot_interspike_interval_hist(self, ax, **kwargs):
        spike_indices = self.spike_train.coalesce().indices()
        time_idx = spike_indices[1]
        sample_idx = spike_indices[0]
        isi = []
        for i in range(torch.max(sample_idx).item() + 1):
            spike_time = torch.sort(time_idx[sample_idx == i], stable=True)[0]
            isi.append(spike_time[1:] - spike_time[:-1])
        isi = torch.cat(isi)
        sns.histplot(isi.numpy(), bins=20, ax=ax, **kwargs)
        ax.set_xlabel('time steps')
        ax.set_title('Interspike interval')
        return ax
    
    def plot_fired_neuron_hist(self, ax, **kwargs):
        fired_neurons = torch.sum(torch.sum(self.spike_train.to_dense(), dim=1) > 0, dim=-1)
        sns.histplot(fired_neurons.numpy(), bins=int(torch.max(fired_neurons).item() + 1), ax=ax, **kwargs)
        ax.set_xlabel('neuron count')
        ax.set_title('# neurons that fired')
        return ax
    

def plot_execution_time(ax, probe_data, alpha=0.2, **kwargs):
    execution_keys = ['hostTimePerTimeStep', 'learningTimePerTimeStep', 'managementTimePerTimeStep', 'spikingTimePerTimeStep', 'totalTimePerTimeStep']
    alias = ['Host', 'Learning', 'Management', 'Spiking', 'Total']
    data = defaultdict(list)
    for i, key in enumerate(execution_keys):
        value = probe_data[key]
        name = alias[i]
        num_ele = value.size
        data['time'].extend(value.tolist())
        data['time step'].extend([i for i in range(num_ele)])
        data['type'].extend([name] * num_ele)
    data = pd.DataFrame(data)
    sns.scatterplot(data=data, x='time step', y='time', hue='type', ax=ax, style='type', alpha=alpha, **kwargs)
    ax.set_ylabel('time ({})'.format(probe_data['timeUnits']))
    return ax


def plot_power(ax, probe_data, **kwargs):
    power_keys = ['power', 'powerVdd', 'powerVdda', 'powerVddio', 'powerVddm']
    alias = ['Total', 'VDD', 'VDDA', 'VDDM', 'VDDIO']
    for i, key in enumerate(power_keys):
        ax.plot(probe_data[key], label=alias[i], **kwargs)
    ax.set_xlabel('time step')
    ax.set_ylabel('power ({})'.format(probe_data['powerUnits']))
    ax.legend()
    return ax

def plot_energy(ax, probe_data, alpha=0.2, **kwargs):
    energy_keys = ['hostPhaseEnergyPerTimeStep', 'learningPhaseEnergyPerTimeStep', 'managementPhaseEnergyPerTimeStep', 'spikingPhaseEnergyPerTimeStep', 'totalEnergyPerTimeStep']
    alias = ['Host', 'Learning', 'Management', 'Spiking', 'Total']
    data = defaultdict(list)
    for i, key in enumerate(energy_keys):
        value = probe_data[key]
        name = alias[i]
        num_ele = value.size
        data['energy'].extend(value.tolist())
        data['time step'].extend([i for i in range(num_ele)])
        data['type'].extend([name] * num_ele)
    data = pd.DataFrame(data)
    sns.scatterplot(data=data, x='time step', y='energy', hue='type', ax=ax, style='type', alpha=alpha, **kwargs)
    ax.set_ylabel('energy ({})'.format(probe_data['energyUnits']))
    return ax


def get_plt_pixel_size():
    px = 1/plt.rcParams['figure.dpi']
    return px

def get_plt_cm_size():
    return 1 / 2.54
    
def save_fig(fig, fig_path, fig_name, dpi=300,  bbox_inches='tight', overwrite=True, format='png', **kwargs):
    fig_name = '{}.{}'.format(fig_name, format)
    if not overwrite:
        fig_name = FuncForLoihiCollections.rename_duplicate_file(fig_path, fig_name)
    fig.savefig(fig_path + fig_name, dpi=dpi, bbox_inches=bbox_inches, format=format, **kwargs)
    
                
