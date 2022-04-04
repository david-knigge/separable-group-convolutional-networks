import torch

from ck_g_cnn.groups import SE2, Rplus, Sim2

from ck_g_cnn.nn import CKGCNN
from ck_g_cnn.nn import CKGResNet
from ck_g_cnn.nn import CKGAllCNNC

import wandb

from config import config
from datasets import get_dataloader, ImplementedDatasets

from train_model import train
from test_model import test


if __name__ == "__main__":

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config["cuda"]) else "cpu")
    config["device"] = device

    if config["group"] == "SE2":
        group = SE2().to(device)
    elif config["group"] == "R2xRplus":
        group = Rplus(max_scale=config["max_scale"]).to(device)
    elif config["group"] == "Sim2":
        group = Sim2(max_scale=config["max_scale"]).to(device)

    # load previously trained model if specified
    # or define model
    if config["model_load_path"]:
        net = torch.load(config["model_load_path"]).to(device)

    elif config["model"] == "ckgcnn":

        net = CKGCNN(
            group=group,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            implementation=config["implementation"],
            spatial_in_size=config["spatial_in_size"],
            num_group_elem=config["num_group_elements"],
            dropout=config["dropout"],
            pooling=config["pooling"],
            stride=config["stride"],
            hidden_sizes=config["hidden_sizes"],
            kernel_size=config["kernel_size"],
            bias=config["bias"],
            normalisation=config["normalisation"],
            ck_net_num_hidden=config["ck_net_num_hidden"],
            ck_net_hidden_size=config["ck_net_hidden_size"],
            ck_net_implementation=config["ck_net_implementation"],
            ck_net_first_omega_0=config["first_omega_0"],
            ck_net_omega_0=config["omega_0"],
            sampling_method=config["sampling_method"]
        ).to(device)

    elif config["model"] == "ckgresnet":

        net = CKGResNet(
            group=group,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            implementation=config["implementation"],
            spatial_in_size=config["spatial_in_size"],
            num_group_elem=config["num_group_elements"],
            pooling=config["pooling"],
            dropout=config["dropout"],
            stride=config["stride"],
            bias=config["bias"],
            padding=config["padding"],
            normalisation=config["normalisation"],
            hidden_sizes=config["hidden_sizes"],
            kernel_size=config["kernel_size"],
            ck_net_num_hidden=config["ck_net_num_hidden"],
            ck_net_hidden_size=config["ck_net_hidden_size"],
            ck_net_implementation=config["ck_net_implementation"],
            ck_net_first_omega_0=config["first_omega_0"],
            ck_net_omega_0=config["omega_0"],
            sampling_method=config["sampling_method"],
            widen_factor=config["widen_factor"],
        ).to(device)

    elif config["model"] == "ckgallcnnc":

        net = CKGAllCNNC(
            group=group,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            implementation=config["implementation"],
            spatial_in_size=config["spatial_in_size"],
            num_group_elem=config["num_group_elements"],
            stride=config["stride"],
            pooling=config["pooling"],
            dropout=config["dropout"],
            padding=config["padding"],
            normalisation=config["normalisation"],
            hidden_sizes=config["hidden_sizes"],
            kernel_size=config["kernel_size"],
            bias=config["bias"],
            ck_net_num_hidden=config["ck_net_num_hidden"],
            ck_net_hidden_size=config["ck_net_hidden_size"],
            ck_net_implementation=config["ck_net_implementation"],
            ck_net_first_omega_0=config["first_omega_0"],
            ck_net_omega_0=config["omega_0"],
            sampling_method=config["sampling_method"],
            widen_factor=config["widen_factor"]
        ).to(device)

    # get dataset
    train_set = get_dataloader(dataset=config['dataset'], batch_size=config["batch_size"], train=True, augment=config['augment'])
    test_set = get_dataloader(dataset=config['dataset'], batch_size=config["batch_size"], train=False, augment=config['augment'])

    # define optimizer and loss criterion
    if config["optim"] == "adam":
        optim = torch.optim.Adam(
            params=net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optim"] == "sgd":
        optim = torch.optim.SGD(
            params=net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            nesterov=config["nesterov"]
        )
    elif config["optim"] == "adadelta":
        optim = torch.optim.Adadelta(
            params=net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config['weight_decay']
        )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if config["learning_rate_steps"]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optim,
            milestones=config["learning_rate_steps"],
            gamma=config["learning_rate_gamma"]
        )
    elif config["learning_rate_cosine"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=config["epochs"]
        )
    else:
        scheduler = None

    config["trainable_parameters"] = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"starting training for dataset: {config['dataset']}")
    print(f"number of trainable parameters: {config['trainable_parameters']}")
    print(net)

    if config["wandb"]:
        run = wandb.init(
            project=config["wandb_project_name"],
            config=config,
            group=str(config["dataset"].value),
            name=config["wandb_run_name"]
        )

    # hacky :) store config file with the model so it gets saved alongside the model
    net.config_file = config

    train(
        model=net,
        optim=optim,
        scheduler=scheduler,
        criterion=criterion,
        train_set=train_set,
        print_interval=config["print_interval"],
        model_save_path=config["model_save_path"],
        epochs=config["epochs"],
        device=device,
        test_fn=lambda: test(net, test_set, device=device, loss=criterion),
    )

    if config["wandb"]:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(config['model_save_path'])
        run.log_artifact(artifact)
        run.join()
