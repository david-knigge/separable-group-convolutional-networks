# Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups

This code accompanies the paper ["Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups"](https://proceedings.mlr.press/v162/knigge22a.html).

### Abstract

Group convolutional neural networks (G-CNNs) have been shown to increase parameter efficiency and model accuracy by incorporating geometric inductive biases. In this work, we investigate the properties of representations learned by regular G-CNNs, and show considerable parameter redundancy in group convolution kernels. This finding motivates further weight-tying by sharing convolution kernels over subgroups. To this end, we introduce convolution kernels that are separable over the subgroup and channel dimensions. In order to obtain equivariance to arbitrary affine Lie groups we provide a continuous parameterisation of separable convolution kernels. We evaluate our approach across several vision datasets, and show that our weight sharing leads to improved performance and computational efficiency. In many settings, separable G-CNNs outperform their non-separable counterpart, while only using a fraction of their training time. In addition, thanks to the increase in computational efficiency, we are able to implement G-CNNs equivariant to the Sim(2) group; the group of dilations, rotations and translations of the plane. Sim(2)-equivariance further improves performance on all tasks considered, and achieves state-of-the-art performance on rotated MNIST.

### Getting started with regular group convolutions
If you are new to working with regular group convolutions, you may be interested in checking out [these lectures](https://www.youtube.com/watch?v=z2OEyUgSH2c&list=PLJ2Aod97Uj8IH7sT4NpM2MOpPeq0H2_lM&index=2&ab_channel=ErikBekkers) and [this tutorial notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html), part of the Deep Learning 2 course at the University of Amsterdam taught by Erik J. Bekkers. For this tutorial we re-used a lot of code from this repo, so it may help you build an intuition.

### Installation

#### conda
We provide an environment file; ``environment.yml`` containing the required dependencies. Clone the repo and run the following command in the root of this directory:
```
conda env create -f environment.yml
```

### Repository Structure
This repository is organized as follows:
- ``ck_g_cnn`` contains the main PyTorch library of our model.
- ``datasets`` contains the datasets used in our experiments.
- ``config.py`` contains the configuration in which to specify default arguments to be passed to the script.
- ``demo`` contains two demo notebooks; ``visualizing_kernels.ipynb`` and ``visualizing_activations.ipynb``, which contains example code for usage of the modules defined in this repo, and may help to build an intuitive understanding of regular group convolutional networks.

### Using the code

All experiments are run with `main.py`. See `config.py` for all available flags. Flags can be passed as: `--kernel_size 7`.

#### Useful flags

- `--group _` Selects the group whose action we want our model to be equivariant to. Currently implemented: `SE2`, `R2xRplus`, `Sim2`.
- `--num_group_elements _` Selects the number of group elements to sample from `H`.
- `--sampling_method _` Selects the grid sampling method over `H`, can either be `uniform` for uniform sampling or `discretise` for a fixed sampling.
- `--implementation _` Selects the group convolution implementation, choices are `nonseparable`, `separable`, `gseparable`, `dseparable`, `dgseparable`. please see appendix C2 of the paper for a thorough explanation of each different implementation. For `group=Sim2`, to obtain convolutions separable along both the dilation and rotation subgroup, we additionally have choices `separable+2d`, `gseparable+2d`.

### Cite
If you found this work useful in your research, please consider citing:

```
@InProceedings{pmlr-v162-knigge22a,
  title = 	 {Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups},
  author =       {Knigge, David M. and Romero, David W and Bekkers, Erik J},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  year = 	 {2022},
  month = 	 {17--23 Jul},
  pdf = 	 {https://proceedings.mlr.press/v162/knigge22a/knigge22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/knigge22a.html},
}
```

### Run examples

To run an experiment with a G-CNN equivariant to SE(2), with 8 randomly sampled rotation elements and nonseparable kernels:
```
python run_experiment.py \
    --model ckgresnet \
    --group SE2 \
    --num_group_elements 8 \
    --sampling_method uniform \
    --hidden_sizes 32,32,64 \
    --dataset MNIST_rot \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --optim adam \
    --kernel_size 7 \
    --stride 1 \
    --dropout 0 \
    --weight_decay 1e-4 \
    --ck_net_num_hidden 1 \
    --ck_net_hidden_size 64 \
    --first_omega_0 10 \
    --omega_0 10 \
    --implementation nonseparable \
    --pooling 1 \
    --normalisation batchnorm \
    --learning_rate_cosine 1 \
    --padding 1  \
    --no_wandb
```

To run an experiment with a G-CNN equivariant to SE(2), with a fixed sampling of 8 rotation elements and separable kernels:
```
python run_experiment.py \
    --model ckgresnet \
    --group SE2 \
    --num_group_elements 8 \
    --sampling_method discretise \
    --hidden_sizes 32,32,64 \
    --dataset MNIST_rot \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --optim adam \
    --kernel_size 7 \
    --stride 1 \
    --dropout 0 \
    --weight_decay 1e-4 \
    --ck_net_num_hidden 1 \
    --ck_net_hidden_size 64 \
    --first_omega_0 10 \
    --omega_0 10 \
    --implementation separable \
    --pooling 1 \
    --normalisation batchnorm \
    --learning_rate_cosine 1 \
    --padding 1  \
    --no_wandb
```

To run an experiment with a G-CNN equivariant to Sim(2), with 16 randomly sampled rotation elements and 3 discretised dilation elements: 
```
python run_experiment.py \
    --model ckgresnet \
    --group Sim2 \
    --num_group_elements 16,3 \
    --max_scale 1.74 \
    --sampling_method uniform \
    --hidden_sizes 32,32,64 \
    --dataset MNIST_rot \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --optim adam \
    --kernel_size 7 \
    --stride 1 \
    --dropout 0 \
    --weight_decay 1e-4 \
    --ck_net_num_hidden 1 \
    --ck_net_hidden_size 64 \
    --first_omega_0 10 \
    --omega_0 10 \
    --implementation separable+2d \
    --pooling 1 \
    --normalisation batchnorm \
    --learning_rate_cosine 1 \
    --padding 1 \
    --no_wandb
```
