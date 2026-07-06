Moment Neural Network (MNN) Documentation
=========================================

`MNN`_ is a class of deep learning architecture that generalizes
rate-based neural networks to second-order statistical moments. Instead
of propagating only deterministic neural activity, an MNN propagates both
the mean activity and the covariance structure of neural variability.

Once trained, MNN parameters can be used to recover a corresponding
spiking neural network (SNN) without an additional fine-tuning phase. The
trained model is intended to capture realistic firing statistics of
biological neurons, including broadly distributed firing rates, Fano
factors, and weak pairwise correlations.

.. _MNN: https://github.com/BrainsoupFactory/moment-neural-network

Installation
------------

Install the project through pip:


.. code-block:: bash

   pip install moment-neural-network


Alternatively, clone the repository:

.. code-block:: bash

   git clone git@github.com:BrainsoupFactory/moment-neural-network.git
   cd moment-neural-network

Install the package from the repository root:

.. code-block:: bash

   python -m pip install -e .

The package metadata requires Python 3.8 or newer and installs
``torch``, ``torchvision``, ``numpy``, ``scipy``, ``PyYAML``, and
``tqdm``.

Optional extras are available for analysis and Loihi-related workflows:

.. code-block:: bash

   python -m pip install -e ".[analysis]"
   python -m pip install -e ".[loihi]"

The README records the original development dependency versions as
PyTorch 1.12.1, TorchVision 0.13.1, SciPy 1.7.3, PyYAML 6.0, and NumPy
1.22.3. Newer versions may work, but reproducibility-sensitive
experiments should start from those versions.

Purpose
-------

This repository implements Moment Neural Networks as PyTorch-style
models. The project targets a limitation of ordinary rate-based
artificial neural networks: biological and neuromorphic spiking systems
compute with noisy, correlated activity, and that variability can carry
task-relevant information.

The research paper *Learning and inference with correlated neural
variability* by Yang Qi, Zhichao Zhu, Yiming Wei, Lu Cao, Zhigang Wang,
Jie Zhang, Wenlian Lu, and Jianfeng Feng frames the scope of the project
as stochastic neural computing for SNNs. The paper describes
moment-closure learning for SNNs with correlated variability, direct
MNN-to-SNN recovery, realistic firing statistics, uncertainty-aware
inference, and neuromorphic deployment.

In this repository, that theory becomes a software stack for:

* training differentiable MNNs with standard PyTorch optimizers;
* representing each layer by mean and covariance transformations;
* using moment activation functions derived from leaky integrate-and-fire
  neuron statistics;
* recovering a corresponding SNN from trained MNN parameters;
* simulating the reconstructed SNN and comparing its spike statistics
  with the MNN prediction.

The project connects deep learning training workflows, stochastic
spiking-neuron theory, and biologically plausible neural variability.


What MNN Solves
---------------

Conventional neural networks usually pass a single deterministic
activation vector through each layer. That is useful for many machine
learning tasks, but it does not describe the statistics of a spiking
population.

Real spiking neurons show trial-to-trial uncertainty, broad firing-rate
distributions, Fano-factor variability, and weak pairwise correlations.
Those quantities matter when the model is intended to explain neural
computation, quantify uncertainty, or run as an SNN.

MNNs address this by treating each layer state as:

.. code-block:: text

   (u, cov)

where ``u`` is the mean activity or firing-rate-like signal, and ``cov``
is the covariance matrix describing variability and correlations across
neurons.

Layers transform both objects. For example, a linear layer maps the mean
by the weight matrix and maps the covariance bilinearly:

.. code-block:: text

   u   -> W u + b
   cov -> W cov W^T

The moment activation maps input-current moments through an
approximation of LIF spike-output moments. This lets the model learn
task-relevant mean activity and correlation structure together.

The repository is especially useful for:

* studying correlated neural variability in spiking systems;
* training MNN classifiers or regression models with PyTorch tooling;
* reconstructing SNNs from trained MNNs;
* simulating LIF networks under Poisson or Gaussian input currents;
* reproducing or extending publication experiments from the project
  authors.


Repository Structure
--------------------

The main package is ``mnn``.

* ``mnn/mnn_core``: core MNN math and PyTorch autograd bindings.
* ``mnn/mnn_core/nn``: neural-network layers, losses, activations, batch
  normalization, pooling, and functional utilities.
* ``mnn/models``: ready-made MLP/CNN model wrappers, including MNN,
  SNN-like, ANN, mean-only variants, and Torch-first Pythonic parallel
  APIs.
* ``mnn/snn``: tools for converting trained MNNs to SNNs and simulating
  them, including Torch-first current, neuron, monitor, probe, and
  validation APIs.
* ``mnn/snn/base``: base neuron, current-generator, monitor, and probe
  classes.
* ``mnn/utils/training_tools`` and ``mnn/utils/training_tools_api``:
  YAML-driven training pipeline, dataloader setup, checkpointing,
  logging, resume logic, and distributed-training helpers. New code
  should prefer the single-file ``training_tools_api`` entrypoint.
* ``mnn/utils/dataloaders``: dataset loaders for MNIST and related
  examples.
* ``mnn/analysis``: analysis and visualization helpers.
* ``docs``: Sphinx documentation and tutorials.
* ``examples/mnist``: a runnable MNIST configuration and
  training/simulation script.
* ``publications``: scripts and configs used for paper-specific
  experiments.


Suggested Learning Path
-----------------------

1. Read ``docs/tutorials/tutorial_moment_activation.ipynb`` to understand
   what the activation computes.
2. Run the MNIST wrapper example with
   ``examples/mnist/mnist_config.yaml``.
3. Inspect ``mnn/models/mlp.py`` to see how full MNN layers are
   assembled.
4. Read ``docs/tutorials/tutorial_training_mnn_vanilla.ipynb`` if you
   want to build custom models.
5. Use ``MnnSnnValidate`` to reconstruct and simulate an SNN from a
   trained checkpoint.
6. Explore ``tutorial_EI_network.ipynb`` and ``publications/`` for
   neuroscience-oriented experiments.


Current Limitations and Notes
-----------------------------

* The most mature model path is MLP-based MNN training.
* The convolution implementation is present but appears less developed
  than the MLP stack.
* Full covariance scales quadratically with layer width, so memory use
  can become large.
* CUDA is strongly recommended for larger models or many SNN trials.
* The docs are tutorial-oriented; source files remain the best reference
  for advanced customization.
* Some scripts are publication- or environment-specific and may need
  path, dataset, or device edits before running.


Mental Model
------------

Think of an MNN as a trainable surrogate for an SNN's firing statistics.
During training, it behaves like a differentiable PyTorch model. During
interpretation or deployment, its parameters can be used to reconstruct a
stochastic spiking circuit. The project is therefore not just another
neural-network layer library; it is a bridge between machine-learning
optimization and spike-based neural variability.


Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/tutorial_moment_activation
   tutorials/tutorial_training_mnn
   tutorials/tutorial_training_mnn_vanilla
   tutorials/tutorial_EI_network


Migration Guides
----------------

.. toctree::
   :maxdepth: 1
   :caption: Migration Guides

   legacy_nn_migration
   legacy_models_snn_migration
   legacy_training_tools_migration


Testing
-------

.. toctree::
   :maxdepth: 1
   :caption: Testing

   test_design_and_results
