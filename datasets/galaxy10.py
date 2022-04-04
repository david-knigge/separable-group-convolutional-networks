import h5py
import numpy as np
import torch.utils.data

from torchvision import datasets, transforms
from torch.utils.data import Dataset


class Galaxy10Dataset(Dataset):

    def __init__(self, root="../data", train=True, transform=None):
        # Load dataset
        f = h5py.File(root + '/Galaxy10/Galaxy10.h5', 'r')

        # The dataset contains two keys: ans and images
        # ans represent the labels with shape (21785,) and the images have shape (21785, 69, 69, 3)
        labels, images = f['ans'], f['images']

        # At first, we want to split our dataset into a training set and a validation set
        train_split = .8

        # Creating data indices for training and validation splits:
        dataset_size = len(images)
        split = int(np.floor(train_split * dataset_size))

        if train:
            images = images[:split]
            labels = labels[:split]
        else:
            images = images[split:]
            labels = labels[split:]

        self.x = images
        self.y = torch.from_numpy(labels).long()
        self.len = len(images)

        self.transform = transform

    def __getitem__(self, item):
        img = self.x[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[item]

    def __len__(self):
        return self.len


if __name__ == "__main__":

    # Load dataset
    tr_ds = Galaxy10Dataset(root="../../data", train=True, transform=transforms.ToTensor())

    d = torch.stack([x[0] for x in tr_ds])
    data = torch.tensor(d)

    print(f'data mean: {torch.mean(data, dim=(0, -1, -2))}')

    print(f'data std: {torch.std(data, dim=(0, -1, -2))}')
