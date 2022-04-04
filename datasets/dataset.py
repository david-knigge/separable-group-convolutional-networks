import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from datasets.galaxy10 import Galaxy10Dataset
from datasets.pcam import PCamDataset

from datasets.mnist_rot import RotatedMNIST
from datasets.utils import pil_loader

from datasets.utils import Cutout

from enum import Enum, auto


class ImplementedDatasets(Enum):
    """
    Defines the set of supported datasets to be used in experiments
    """
    MNIST = "MNIST"
    MNIST_rot = "MNIST_rot"
    MNIST_scale = "MNIST_scale"
    MNIST_rot_scale = "MNIST_rot_scale"

    CIFAR10 = "CIFAR10"
    CIFAR10_rot = "CIFAR10_rot"
    CIFAR100 = "CIFAR100"

    STL10 = "STL10"

    Galaxy10 = "Galaxy10"

    PCam = "PCam"

    @classmethod
    def is_implemented(cls, value):
        return value in cls._value2member_map_


def get_num_in_channels(dataset):
    """ Return the number of input channels for given dataset

    @param dataset: ImplementedDatasets enum type
    """
    if dataset in [ImplementedDatasets.MNIST, ImplementedDatasets.MNIST_rot, ImplementedDatasets.MNIST_scale,
                   ImplementedDatasets.MNIST_rot_scale]:
        return 1  # greyscale
    elif dataset in [ImplementedDatasets.CIFAR10, ImplementedDatasets.CIFAR10_rot, ImplementedDatasets.STL10,
                     ImplementedDatasets.Galaxy10, ImplementedDatasets.PCam, ImplementedDatasets.CIFAR100]:
        return 3  # rgb
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


def get_num_out_channels(dataset):
    """ Return the number of output channels for given dataset

        @param dataset: ImplementedDatasets enum type
        """
    if dataset in [
        ImplementedDatasets.MNIST,
        ImplementedDatasets.MNIST_rot,
        ImplementedDatasets.MNIST_scale,
        ImplementedDatasets.MNIST_rot_scale,
        ImplementedDatasets.CIFAR10,
        ImplementedDatasets.CIFAR10_rot,
        ImplementedDatasets.Galaxy10,
        ImplementedDatasets.STL10,
    ]:
        return 10  # 10 classes
    elif dataset == ImplementedDatasets.PCam:
        return 2
    elif dataset == ImplementedDatasets.CIFAR100:
        return 100
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


def get_imsize(dataset):

    if dataset in [
        ImplementedDatasets.MNIST,
        ImplementedDatasets.MNIST_rot,
        ImplementedDatasets.MNIST_scale,
        ImplementedDatasets.MNIST_rot_scale
    ]:
        return 28

    elif dataset in [
        ImplementedDatasets.CIFAR10,
        ImplementedDatasets.CIFAR10_rot,
        ImplementedDatasets.CIFAR100
    ]:
        return 32

    elif dataset == ImplementedDatasets.Galaxy10:
        return 69

    elif dataset == ImplementedDatasets.STL10:
        return 96

    else:
        raise ValueError(f"Dataset {dataset} not supported.")


def get_dataloader(dataset, batch_size, train=True, root="../data", augment=False):
    if dataset == ImplementedDatasets.MNIST:

        tf = [transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))]
        ds = torchvision.datasets.MNIST(root, train=train, download=True, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.MNIST_rot:

        if augment and train:
            ds = RotatedMNIST(root=root, partition="train" if train else "validation", augment="randomrot")
        else:
            ds = RotatedMNIST(root=root, partition="train" if train else "validation", augment="None")

    elif dataset == ImplementedDatasets.CIFAR10:

        if augment and train:
            tf = [
                transforms.RandomCrop(32, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            print('augmentation')
        else:
            tf = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.CIFAR10_rot:

        tf = [transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              transforms.RandomRotation((0, 360))
              ]

        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.Galaxy10:

        tf = [
            transforms.ToTensor(),
            transforms.Normalize((0.1086, 0.0934, 0.0711), (0.1471, 0.1229, 0.1030)),
        ]

        ds = Galaxy10Dataset(root=root, train=train, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.PCam:

        tf = [
            transforms.ToTensor(),
            transforms.Normalize((0.7008, 0.5384, 0.6916), (0.2350, 0.2774, 0.2129))
        ]

        ds = PCamDataset(root=root, train=train, transform=transforms.Compose(tf))

        # h5py files can't handle multiple workers
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    elif dataset == ImplementedDatasets.MNIST_scale:

        tf = [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.0607,), (0.2161,))
        ]

        if train:
            root += "/MNIST_scale/seed_0/scale_0.3_1.0/train"
        else:
            root += "/MNIST_scale/seed_0/scale_0.3_1.0/test"

        ds = datasets.ImageFolder(root, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.CIFAR100:

        tf = [transforms.ToTensor(),
              transforms.Normalize(
                  mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
              )]

        ds = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transforms.Compose(tf))

    elif dataset == ImplementedDatasets.MNIST_rot_scale:

        tf = [
            transforms.ToTensor(),
            transforms.Normalize((0.0634,), (0.2102,))
        ]

        if train:
            root += "/MNIST_rot_scale/train"
        else:
            root += "/MNIST_rot_scale/test"

        ds = datasets.ImageFolder(root, transform=transforms.Compose(tf), loader=pil_loader)

    elif dataset == ImplementedDatasets.STL10:

        if train:

            tf = [
                transforms.RandomCrop(96, padding=12),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                Cutout(1, 32)
            ]

        else:

            tf = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]

        ds = torchvision.datasets.STL10(
            root,
            split="train" if train else "test",
            download=True,
            transform=transforms.Compose(tf)
        )

    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
