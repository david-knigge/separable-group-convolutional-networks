
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch

from argparse import ArgumentParser

import os,sys
ck_g_cnn_source =  os.path.join(os.getcwd(),'..')
if ck_g_cnn_source not in sys.path:
    sys.path.append(ck_g_cnn_source)
from datasets.utils import RandomScaling


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='source folder of the dataset')
    parser.add_argument('--dest', type=str, required=True, help='destination folder for the output')

    parser.add_argument('--download', action='store_true',
                        help='download source dataset if needed.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    source = args.source
    dest = args.dest
    download = args.download
    torch.manual_seed(args.seed)

    random_rot = transforms.RandomRotation(
        degrees=(0, 360),
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    random_scale = RandomScaling(min_factor=0.3, max_factor=1.0)

    tf = transforms.Compose([
        transforms.Grayscale(),
        random_rot,
        random_scale
    ])

    train_set = datasets.MNIST(root=source, train=True, download=download, transform=tf)
    test_set = datasets.MNIST(root=source, train=False, download=download, transform=tf)
    concat_dataset = ConcatDataset([train_set, test_set])

    train, test_valid = torch.utils.data.random_split(concat_dataset, [12000, 58000], generator=torch.Generator().manual_seed(args.seed))
    valid, test = torch.utils.data.random_split(test_valid, [8000, 50000], generator=torch.Generator().manual_seed(args.seed))

    for idx, (data, target) in enumerate(train):
        save_dir = f'{dest}/train/{target}/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        data.convert("L").save(save_dir + f'{idx}.png')

    for idx, (data, target) in enumerate(test):
        save_dir = f'{dest}/test/{target}/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        data.convert("L").save(save_dir + f'{idx}.png')

    for idx, (data, target) in enumerate(valid):
        save_dir = f'{dest}/valid/{target}/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        data.convert("L").save(save_dir + f'{idx}.png')