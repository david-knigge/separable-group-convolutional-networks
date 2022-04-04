import h5py
import torch.utils.data

from torchvision import transforms
from torch.utils.data import Dataset


class PCamDataset(Dataset):

    def __init__(self, root="../data", train=True, transform=None):

        if train:
            # Load dataset
            f_1 = h5py.File(root + '/pcam/camelyonpatch_level_2_split_train_x.h5', 'r')
            f_2 = h5py.File(root + '/pcam/camelyonpatch_level_2_split_train_y.h5', 'r')
        else:
            f_1 = h5py.File(root + '/pcam/camelyonpatch_level_2_split_test_x.h5', 'r')
            f_2 = h5py.File(root + '/pcam/camelyonpatch_level_2_split_test_y.h5', 'r')

        self.y, self.x = f_2['y'], f_1['x']
        self.len = len(self.y)
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
    tr_ds = PCamDataset(root="../../data", train=True, transform=transforms.ToTensor())

    d = torch.stack([x[0] for x in tr_ds])
    data = torch.tensor(d)

    print(f'data mean: {torch.mean(data, dim=(0, -1, -2))}')

    print(f'data std: {torch.std(data, dim=(0, -1, -2))}')

    print(f'max data: {torch.amax(data, dim=(0, -1, -2))}')

    print(f'min data: {torch.amin(data, dim=(0, -1, -2))}')
