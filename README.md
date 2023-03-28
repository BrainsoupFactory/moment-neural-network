
# Table of content
- [moment-neural-network](#moment-neural-network)
  - [The architecture of this repository](#the-architecture-of-this-repository)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
  - [Quick start: three steps to run your first MNN model](#quick-start-three-steps-to-run-your-first-mnn-model)
  - [Configure the MNN model](#configure-the-mnn-model)
  - [Configure additional training options via input arguments.](#configure-additional-training-options-via-input-arguments)
  - [Run simulations of the reconstructed SNN](#run-simulations-of-the-reconstructed-snn)
- [Customize your own MNN model](#customize-your-own-mnn-model)
  - [Custom dataset](#custom-dataset)
  - [Custom loss function](#custom-loss-function)
  - [Custom model](#custom-model)
- [Lead authors](#lead-authors)
- [License](#license)

# moment-neural-network

The moment neural network is a type of second-order artificial neural network model designed to capture the nonlinear coupling of correlated activity of spiking neurons. In brief, the moment neural networks extend conventional rate-based artificial neural network models by incorporating the covariance of fluctuating neural activity. This repository provides a comprehensive framework for simulating and training moment neural networks based on the standard workflow of pytorch. 

## The architecture of this repository

* `mnn_core`: core modules implementing the moment activation and other building blocks of MNN.
* `models`: a module containging various network architectures for fast and convenient model construction
* `snn`: modules for reconstructing SNN from MNN and for simulating the corresponding SNN in a flexible manner.
* `utils`: a collection of useful utilities for training MNN (ANN compatible).

# Dependencies
* python 3
* pytorch: 1.12.1
* torchvision: 0.13.1
* scipy: 1.7.3
* pyyaml: 6.0
* numpy: 1.22.3
* CUDA (optional)

# Getting Started

## Quick start: three steps to run your first MNN model

The following provides a step-by-step instruction to train an MNN to learn MNIST image classification task with a multi-layer perceptron structure.

1. Clone the repository to your local drive.
2. Copy the demo files, **./example/mnist/mnist.py** and **./example/mnist/mnist_config.yaml** to the root directory.
3. Create two directories, **./checkpoint/** (for saving trained model results) and **./data/** (for downloading the MNIST dataset).
4. Run the following command to call the script named `mnist.py` with the config file specified through the option:

   ```
   python mnist.py --config=./mnist_config.yaml
   ```

After training is finished, you should find four files in the **./checkpoint/mnist/** folder：

- Two '.ph' files which contain the trained model parameters.
- One '.yaml' file which is a copy of the config file used for running the training the model.
- One '.txt' log file that prints the standard output during training (such as model performance).
- One directroy called `mnn_net_snn_result` that stores the simulation result of the SNN reconstructed from the trained MNN (if enabled).

## Configure the MNN model

Let's review the content of **mnist.yaml**.

The `MODEL` section is for specifying the architecture of MNN.
`meta`: meta information about model construction.
- `arch`: specifies the model architecture. Currently only mlp-like architecture is available (`arch: mnn_mlp`).
- `mlp_type`: indicates the kind of mlp to be built. For `mnn_mlp`, the model contains one input layer, arbitrary number of hidden layers, and a linear decoder.
`mnn_mlp`: detailed model specification for mlp
- `structure`: you can change the widths of each layer by modifying the values under this field.
- `num_class`: specifies the output dimension.
See `mnn.models.mlp` for under-the-hood details.

The `CRITERION` section indicate the training criterion such as the loss function.
`name`: the name for the loss function. Currently supports ...
`source`: the name of the directory where the loss function is defined. 
`arg`: input arguments to the loss function.
The code will try to find the criterion from `source` that match the `name` and pass required `args` to it.
See `mnn_core.nn.criterion` for under-the-hood details.

Similarly, the optimzer and data augmentation policy are defined under `OPTIMIZER` and `DATAAUG_TRAIN/VAL`, correspoding to the pytorch implementations (`torch.optim` and `torchvision.transforms` ).

There are some advanced options in the config file:

* `save_epoch_state`: at the start of each epoch, the code will store the model parameters.
* `input_prepare`: currently only *flatten_poisson* is valid. It means we first flatten input to a vector and regard it as independent Poisson rate code.
* `scale_factor`: only valid if `input_prepare` is *flatten_poisson*, used to control input range.
* `is_classify`: the task type, if `False`, the best model is determined by the epoch that has minimal loss.
* `background_noise`: this value will add to the diagonal of input covariance (Can be helpful if input covariance is very weak or close to singular)

## Configure additional training options via input arguments.

```
python main_script.py --config=./your_config_file.yaml --OPT=VALUE
```

Some examples of the `OPT` field: 
* `seed`: fix the seed for all RNGs used by the model. By default it is `None` (not fixed)
* `bs`: batch size used in the data loader
* `dir`: directory name for saving training data
* `save_name`: the prefix of file name of training data
* `epochs`: the number of epochs to train.
* `cpu`: manually set device to CPU 

I recommend you to read the func `deploy_config()` in `utils.training_tools.general_prepare`

**Note** all manual argument will be overwritten if the same keys are found in the provided **your_config_file.yaml**

## Run simulations of the reconstructed SNN

We provide utility to automatically reconstruct SNN based on the trained MNN.
A custom simulator of SNN is provided with GPU support but you may use any SNN simulator of your choice.

# Customize your own MNN model

(How to add custom models to the MODEL folder; details of the model class)

## Custom dataset

## Custom loss function

## Custom model

## 

# Lead authors

- **Zhichao Zhu** - *Chief Architect* - [Zhichao Zhu](https://github.com/Acturos)
- **Yang Qi** - *Lead Algorithm Design* - [Yang Qi](https://github.com/qiyangku)

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
