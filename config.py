import torch.nn.functional as F
import ck_g_cnn.nn.functional as gF

from datasets.dataset import ImplementedDatasets, get_num_in_channels, get_num_out_channels, get_imsize

from ck_g_cnn.utils import fix_seed

import argparse


# default configuration
config = dict(

    # model configuration
    model='',
    group='',
    num_group_elements='',
    max_scale='',
    hidden_sizes='',
    widen_factor='',
    kernel_size='',
    stride='',
    bias='',

    # in case of ckgsqueezenet, define no channels within gconv block
    squeeze_channels='',

    implementation='',
    sampling_method='',

    # CKNet configuration
    ck_net_num_hidden='',
    ck_net_hidden_size='',
    ck_net_implementation='',

    # SIREN parameters
    first_omega_0='',
    omega_0='',

    # experiment / hyperparameter configuration
    dataset='',
    augment='',
    epochs='',
    batch_size='',
    dropout='',

    # optimizer
    optim='',
    nesterov='',
    momentum='',
    learning_rate_gamma='',
    learning_rate_steps='',
    learning_rate_cosine='',
    learning_rate='',
    weight_decay='',

    # utilities
    wandb_project_name="ck-g-cnn",
    wandb_run_name="",
    print_interval=10,
    cuda=True,
    model_save_path='./',
    model_load_path='',
    wandb=True,
    fix_seed=False,
)


def parse_args():
    """ Parse command line arguments for this experiment, will be used to possibly override default configuration
    :return: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train CK-G-CNN model")

    parser.add_argument('--model_save_path', type=str, required=False)
    parser.add_argument('--model_load_path', type=str, required=False)
    parser.add_argument('--print_interval', type=int, required=False, help="Interval at which to print loss.")
    parser.add_argument('--no_cuda', required=False, action='store_true')
    parser.add_argument('--no_wandb', required=False, action='store_true')
    parser.add_argument('--wandb_project_name', type=str, required=False)
    parser.add_argument('--wandb_run_name', type=str, required=False, default='')
    parser.add_argument('--fix_seed', required=False, type=int)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--optim', type=str, choices=["adam", "sgd", "adadelta"], required=True)
    parser.add_argument('--nesterov', action="store_true", required=False)
    parser.add_argument('--learning_rate_steps', type=int, nargs='+', default=[])
    parser.add_argument('--learning_rate_cosine_annealing', action='store_true')
    parser.add_argument('--learning_rate_gamma', type=float, required=False, default=0.1)
    parser.add_argument('--momentum', type=float, required=False, default=0.0)
    parser.add_argument('--learning_rate_cosine', type=int, required=False,
                        default=0)

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)

    parser.add_argument(
        '--dataset',
        choices=[
            "MNIST", "MNIST_rot", "MNIST_scale", "CIFAR10", "CIFAR10_rot", "MNIST_rot_scale", "STL10", "CIFAR100", "Galaxy10"
        ],
        required=True
    )
    parser.add_argument('--augment', required=False, action='store_true')

    parser.add_argument('--group', type=str, required=True, help="Group to incorporate equivariance for.", choices=['SE2', 'Sim2', 'R2xRplus'])
    parser.add_argument('--num_group_elements', type=str, required=True,
                        help="Size of discretised groups.")
    parser.add_argument('--sampling_method', type=str, required=True, choices=["discretise", "uniform"],
                        help="Which method to use in sampling group elements from the Lie group.")
    parser.add_argument('--max_scale', type=float, required=False, help="In case of scale equivariance, where to truncate the scale group.")

    parser.add_argument('--hidden_sizes', type=str, required=True, help="List of hidden layer sizes.")
    parser.add_argument('--kernel_size', type=int, required=True, help="Kernel size to use throughout the network.")
    parser.add_argument('--stride', type=int, required=True, help="If not 1, implements strided convolutions.")
    parser.add_argument('--bias', action='store_true', required=False, help="Use bias in group convolutions.")

    parser.add_argument('--normalisation', type=str, choices=['batchnorm', 'instancenorm',
                        'layernorm'], required=False, help="Use normalisation")
    parser.add_argument('--no_normalisation', required=False, action="store_true", help="Use normalisation")

    parser.add_argument('--pooling', required=False, type=int, help="Use pooling")
    parser.add_argument('--no_pooling', required=False, action="store_true", help="Do not use "
                                                                                "pooling")

    parser.add_argument('--padding', required=False, type=int, help="Use padding.")
    parser.add_argument('--no_padding', required=False, action="store_true", help="Do not use "
                                                                                  "padding.")

    parser.add_argument('--implementation', required=True, type=str, choices=["nonseparable","separable","gseparable", "dseparable", "dgseparable", "separable+2d","gseparable+2d"],help="Lifting and Gconv implementation type.")
    parser.add_argument('--model', type=str, required=True, choices=["ckgcnn", "ckgresnet", "ckgwideresnet", "ckgbottlenecknet", "ckgcnnsosnovik", "ckgallcnnc", "ckgallcnncbottleneck", "benchmarkcnn", "benchmarkwrn", "benchmarkwrnivan"])
    parser.add_argument('--bottleneck_factor', type=float, required=False, default=1,
                        help="In case of model ckgbottlenecknet, factor of number of channels "
                             "squeezed to.")
    parser.add_argument('--widen_factor', type=float, required=False, default=1, help="Optionally widen number of channels.")

    parser.add_argument('--dummy_var_for_wandb', type=int, required=False, help="Wandb var to "
                                                                                "grid search over .")

    parser.add_argument('--ck_net_num_hidden', type=int, required=True, help="Number of hidden layers in CKnet.")
    parser.add_argument('--ck_net_hidden_size', type=int, required=True, help="Size of hidden layers in CKnet.")
    parser.add_argument('--ck_net_implementation', type=str, required=False, default="SIREN", choices=["SIREN", "RFF", "MFN"],
                        help="Type of high-frequency prior to place on cknet.")

    parser.add_argument('--first_omega_0', type=float, required=True, help="First omega_0 of CKnet Sine layer.")
    parser.add_argument('--omega_0', type=float, required=True, help="Omega_0 of CKnet Sine layer.")

    args = parser.parse_args()

    # these arguments override configuration values
    if args.dataset is not None:
        if ImplementedDatasets.is_implemented(args.dataset):
            config["dataset"] = ImplementedDatasets[args.dataset]
        else:
            raise ValueError(f"Dataset {args.dataset} not implemented.")
    if args.augment:
        config['augment'] = True
    else:
        config['augment'] = False

    # optimiser parameters
    config["optim"] = args.optim
    config["nesterov"] = args.nesterov
    config["learning_rate"] = args.learning_rate
    config["learning_rate_steps"] = args.learning_rate_steps
    config["learning_rate_gamma"] = args.learning_rate_gamma
    config["learning_rate_cosine"] = args.learning_rate_cosine

    # training parameters
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["dropout"] = args.dropout
    config["weight_decay"] = args.weight_decay
    config['momentum'] = args.momentum

    # experiment parameters
    config["model"] = args.model
    if args.model == "ckgbottlenecknet":
        if not args.bottleneck_factor:
            raise ValueError(f"For model {args.model}, the argument '--bottleneck_factor' is required.")
    config["group"] = args.group
    if args.group in ["Sim2", "R2xRplus"]:
        if not args.max_scale:
            raise ValueError(f"For group {args.group}, the argument 'max_scale' is required.")
        else:
            config["max_scale"] = args.max_scale

    config["num_group_elements"] = [int(s) for s in args.num_group_elements.split(",")]
    if len(config["num_group_elements"]) == 1:
        config["num_group_elements"] = config["num_group_elements"][0]

    config["hidden_sizes"] = [int(s) for s in args.hidden_sizes.split(",")]
    config["bottleneck_factor"] = args.bottleneck_factor
    config["widen_factor"] = args.widen_factor

    config["kernel_size"] = args.kernel_size
    config["stride"] = args.stride
    if args.bias: config["bias"] = args.bias

    config["implementation"] = args.implementation
    config["sampling_method"] = args.sampling_method

    if args.normalisation: config["normalisation"] = args.normalisation
    if args.no_normalisation: config["normalisation"] = False
    if not args.no_normalisation and not args.normalisation:
        raise ValueError("Either no normalisation or a form of normalisation needs to be "
                         "specified.")

    if args.pooling: config["pooling"] = True
    if args.no_pooling: config["pooling"] = False
    if not args.pooling and not args.no_pooling:
        raise ValueError("Either no pooling or pooling needs to be specified.")

    if args.padding: config["padding"] = True
    if args.no_padding: config["padding"] = False
    if not args.padding and not args.no_padding:
        raise ValueError("Either padding or no padding needs to be specified.")

    config["ck_net_num_hidden"] = args.ck_net_num_hidden
    config["ck_net_hidden_size"] = args.ck_net_hidden_size
    config["ck_net_implementation"] = args.ck_net_implementation

    config["first_omega_0"] = args.first_omega_0
    config["omega_0"] = args.omega_0

    # create save path based on current setup
    config['model_save_path'] = (
            config['model_save_path'] +
            f"{config['model']}+{config['implementation']}+h_s-{str(config['hidden_sizes'])}+k-{str(config['kernel_size'])}+om-{config['omega_0']}+f_om-{config['first_omega_0']}+do-{config['dropout']}+wd-{config['weight_decay']}+n_el-{config['num_group_elements']}+grp-{config['group']}+smplng-{config['sampling_method']}+implm-{config['ck_net_implementation']}+ds-{config['dataset'].value}.pt"
    )

    # wandb run name to overwrite
    run_name_template = f"{config['implementation']}+{config['model']}"
    if args.wandb_run_name:
        config['wandb_run_name'] = run_name_template + ('+' + args.wandb_run_name)

    if args.model_load_path: config["model_load_path"] = args.model_load_path
    if args.print_interval: config["print_interval"] = args.print_interval
    if args.no_cuda: config["cuda"] = not args.no_cuda
    if args.no_wandb: config["wandb"] = not args.no_wandb
    if args.wandb_project_name: config["wandb_project_name"] = args.wandb_project_name
    if args.fix_seed: config["fix_seed"] = bool(args.fix_seed)

    return args


# parse any command line arguments that may have been specified
parse_args()

# fix seed if configuration requires it
if config["fix_seed"]:
    fix_seed(seed=42)

# determine the number of input channels for our model
config["in_channels"] = get_num_in_channels(config["dataset"])
config["out_channels"] = get_num_out_channels(config["dataset"])
config["spatial_in_size"] = get_imsize(config["dataset"])
