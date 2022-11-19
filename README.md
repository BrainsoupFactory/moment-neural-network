- [moment-neural-network](#moment-neural-network)
  - [The architecture of this repository](#the-architecture-of-this-repository)
- [Getting Started](#getting-started)
  - [Quick start: three steps to run your first MNN model](#quick-start-three-steps-to-run-your-first-mnn-model)
  - [Custom your own MNN model](#custom-your-own-mnn-model)
  - [Configure the training options via argparser.](#configure-the-training-options-via-argparser)
  - [Run corresponding SNN simulation](#run-corresponding-snn-simulation)
- [Authors](#authors)
- [License](#license)

# moment-neural-network

## The architecture of this repository
* `mnn_core`: the implementation of mnn activation and a variety of modules that support the construction of mnn and training.
* `models`: templates for fast and convenient model construction
* `snn`: modules for converting mnn to snn and run simulation in a flexible manner.
* `utils`: a collection of useful utilities for MNN (ANN compatible) training.

# Getting Started

## Quick start: three steps to run your first MNN model

1. Copy the example files, **./example/mnist/mnist.py** and **./example/mnist/mnist_config.yaml** to the root directory
2. Create two directory, **./checkpoint/** (for dump trained model) and **./data/** (for downloading MNIST dataset).
3. run the script.

    ```
    python mnist.py --config=./mnist.yaml
    ```

After train, you should find four files in the **./checkpoint/mnist/**：

- two model files that contains the trained model parameters with suffix .pth
- one config file that record all information to run such training with suffix .yaml
- one log text that record the model performance during the training with suffix .txt
- one directroy that store the snn simulation result base on trained MNN

## Custom your own MNN model

Let's review the content of **mnist.yaml**.
To specify the architecture of MNN, you can modify the `MODEL` zone.
The `meta` provide the information about how to construct model. 
Currently only mlp-like architecture is available (`arch: mnn_mlp`).
The `mlp_type` indicates which kind of mlp to build.
If it is  `mnn_mlp`, MNN model has a linear decoder. 
If `snn_mlp`, the decoder is just another MNN ensemble layer (Summation-BatchNorm-Activation)
You can change the width and depth of MNN by modify the value of `structrue` and `num_class` is the output dimension. 
See `mnn.models.mlp` for further details.

Next, The `Criterion` zone indicate the criterion used in training. 
The code will try to find the criterion from `source` that match the `name` and pass required `args` to it.
I have implemented a family of criterion for MNN, see `mnn_core.nn.criterion` for further details.

Similarly, the optimzer and data augmentation policy is built base on `OPTIMIZER` and `DATAAUG_TRAIN/VAL`, correspoding to the pytorch implementations (`torch.optim` and `torchvision.transforms` )

Alternality, you can rewrite the correspoding functions of `MnistTrainFuncs` in the script **mnist.py** for your own needs

There are some advanced options in configure file:
* `save_epoch_state`: before each epoch, the code will store model parameters before run.
* `input_prepare`: currently only *flatten_poisson* valid, it means we first flatten input and regard it as independent Poisson code.
* 'scale_factor': only valid if `input_prepare` is *flatten_poisson*, used to control input range.
* `is_classify`: the task type, if False, the best model is determined by the epoch that has minimal loss.
* `background_noise`: this value will add to the digonal of input covariance (Can be helpful if input covariance is very weak or semidefinite)

## Configure the training options via argparser.
I recommend you to read the func `deploy_config()` in `utils.training_tools.general_prepare`
Some important options:
* `seed`: the random seed to fix. By default it is None (not fixed)
* `bs`: batch size used in data loader
* `dir`: directory name for dump training data
* `save_name`: the prefix of file name of training data
* `epochs`: the num of epochs to train.
How to use:
```
python your_script.py --OPT=VALUE
```
**Note** all of this argument can be overwrite if the same keys are finded in provided **configure.yaml**

## Run corresponding SNN simulation


# Authors

- **Zhichao Zhu** - *Chief architect and initial work* - [Zachary Zhu](https://github.com/Acturos)

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
