from __future__ import print_function
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset
from torchvision.transforms import v2

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""
class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_img, label = self.base_dataset[idx]

        # Preserve original image
        raw_img_copy = raw_img.copy() if isinstance(raw_img, Image.Image) else raw_img

        # Apply transform
        transformed_img = self.transform(raw_img) if self.transform else raw_img

        return transformed_img, label, raw_img_copy

def get_data_folder():
    """
    return the path to store the data
    """
    #data_folder = './data/'
    data_folder = '/home/zhl/workspace/dataset/CIFAR100'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CIFAR100BackCompat(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

class CIFAR100Instance(CIFAR100BackCompat):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def custom_collate(batch):
    transformed, labels, raw_imgs = zip(*batch)
    return torch.stack(transformed), torch.tensor(labels), raw_imgs  

checkpoint_path = '/export/livia/home/vision/Bkarimian/CONCH/checkpoints/conch/pytorch_model.bin'

def get_pcam_dataloaders(data_folder, batch_size=128, num_workers=8, shuffle_train = True):
    """
    pcam
    """

    train_transform = v2.Compose([
                            v2.Resize(224),
                            v2.CenterCrop(224),
                            v2.ToTensor(),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
                        ])
    test_transform =  v2.Compose([
                            v2.Resize(224),
                            v2.ToTensor(),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
                        ])

    train_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform= None,
                )
    train_set = CustomDataset(train_set, transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers, collate_fn=custom_collate)

    val_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "val"
                )
    val_loader = DataLoader(val_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    
    test_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "test"
                )
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader
