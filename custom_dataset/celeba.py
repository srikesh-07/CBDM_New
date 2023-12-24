import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms, utils, datasets
import random
import numpy as np

from torch import cuda
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

image_size = 64
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
}

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_celeb_loader(data_root, transform=None):
    txt_train = os.path.join(data_root, 'celebA_train_orig.txt')
    txt_val = os.path.join(data_root, 'celebA_val_orig.txt')
    txt_test = os.path.join(data_root, 'celebA_test_orig.txt')
    data_root = os.path.join(data_root, "img_align_celeba")
    print(data_root)

    # data_root = '/home/temp/data/CelebA/'
    set_train = LT_Dataset(data_root, txt_train, transform=transform)
    set_val = LT_Dataset(data_root, txt_val, transform=transform)
    set_test = LT_Dataset(data_root, txt_test, transform=transform)
    # train_loader = DataLoader(set_train, batch_size, shuffle=True, num_workers=num_workers,pin_memory=cuda.is_available())
    # val_loader = DataLoader(set_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())
    # test_loader = DataLoader(set_test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())

    return set_train, set_val, set_test