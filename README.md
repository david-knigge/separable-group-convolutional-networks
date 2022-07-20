# Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups

This code accompanies the paper ["Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups"](https://proceedings.mlr.press/v162/knigge22a.html).

### Abstract

Group convolutional neural networks (G-CNNs) have been shown to increase parameter efficiency and model accuracy by incorporating geometric inductive biases. In this work, we investigate the properties of representations learned by regular G-CNNs, and show considerable parameter redundancy in group convolution kernels. This finding motivates further weight-tying by sharing convolution kernels over subgroups. To this end, we introduce convolution kernels that are separable over the subgroup and channel dimensions. In order to obtain equivariance to arbitrary affine Lie groups we provide a continuous parameterisation of separable convolution kernels. We evaluate our approach across several vision datasets, and show that our weight sharing leads to improved performance and computational efficiency. In many settings, separable G-CNNs outperform their non-separable counterpart, while only using a fraction of their training time. In addition, thanks to the increase in computational efficiency, we are able to implement G-CNNs equivariant to the Sim(2) group; the group of dilations, rotations and translations of the plane. Sim(2)-equivariance further improves performance on all tasks considered, and achieves state-of-the-art performance on rotated MNIST.

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

All experiments are run with `main.py`. Flags are handled by [Hydra](https://hydra.cc/docs/intro).
See `cfg/config.yaml` for all available flags. Flags can be passed as `xxx.yyy=value`.

#### Useful flags

- `net.*` describes settings for the models (model definition `models/resnet.py`).
- `kernel.*` describes settings for the MAGNet kernel generator networks.
- `mask.*` describes settings for the FlexConv Gaussian mask.
- `conv.*` describes settings for the convolution operation. It can be used to switch between FlexConv, CKConv, regular Conv, and their separable variants.
- `dataset.*` specifies the dataset to be used, as well as variants, e.g., permuted, sequential.
- `train.*` specifies the settings used for the Trainer of the models.
- `train.do=False`: Only test the model. Useful in combination with pre-training.
- `optimizer.*` specifies and configures the optimizer to be used.
- `debug=True`: By default, all experiment scripts connect to Weights & Biases to log the experimental results. Use this flag to run without connecting to Weights & Biases.
- `pretrained.*`: Use these to load checkpoints before training.

### Reproducing experiments
Please see the [experiments README](/experiments/README.md) for details on reproducing the paper's experiments.

### Cite
If you found this work useful in your research, please consider citing:


## Examples

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
