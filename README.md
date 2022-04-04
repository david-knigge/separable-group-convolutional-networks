# Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups

This code accompanies the paper "Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups". Link to [arxiv](https://arxiv.org/abs/2110.13059).

Requirements: see `environment.yml`.


# Examples

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