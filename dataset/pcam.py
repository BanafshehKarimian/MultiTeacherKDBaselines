from __future__ import print_function
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset
from torchvision.transforms import v2
import itertools
import random
from dataset.Embedding_loader import EmbeddingLoader

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

class RepeatDatasetR(Dataset):
    def __init__(self, base_dataset, T: int, q: int):
        self.base = base_dataset
        items = list(range(T))
        self.combos = list(itertools.combinations(items, q))
        self.length = len(base_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        base_idx = idx 
        img  = self.base[base_idx][0]
        label = self.base[base_idx][1]
        combo_idx = random.choice(range(len(self.combos)))
        return img, label, torch.tensor(self.combos[combo_idx])
class RepeatDataset(Dataset):
    def __init__(self, base_dataset, T: int, q: int):
        self.base = base_dataset
        items = list(range(T))
        self.combos = list(itertools.combinations(items, q))
        self.k = len(self.combos)
        self.length = len(base_dataset) * self.k

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        base_idx = idx // self.k
        img  = self.base[base_idx][0]
        label = self.base[base_idx][1]
        combo_idx = idx % self.k
        return img, label, torch.tensor(self.combos[combo_idx])
    
def get_pcam_dataloaders(data_folder, batch_size=128, num_workers=8, shuffle_train = True, subset_frac=1.0, no_norm = False, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], return_dataset = False, teachers_cnt = None):
    """
    pcam
    """
    train_transform = [v2.Resize(224), v2.ToTensor()]#, v2.CenterCrop(224)
    test_transform = [ v2.Resize(224), v2.ToTensor(),]
    if not no_norm:
        train_transform.append(v2.Normalize(mean=mean, std=std))
        test_transform.append(v2.Normalize(mean=mean, std=std))
    train_transform = v2.Compose(train_transform)
    test_transform =  v2.Compose(test_transform)

    train_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform= test_transform,
                )
    #train_set = CustomDataset(train_set, transform=train_transform)
    if subset_frac < 1.0:
        n = len(train_set)
        k = max(1, int(round(subset_frac * n)))
        subset_idx = np.random.choice(n, size=k, replace=False)
        train_set = Subset(train_set, subset_idx)

    val_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "val"
                )
    
    test_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "test"
                )
    
    if return_dataset:
        if teachers_cnt:
            return RepeatDatasetR(train_set, T = teachers_cnt, q = 2), RepeatDataset(val_set, T = teachers_cnt, q = 2), RepeatDataset(test_set, T = teachers_cnt, q = 2)
        return train_set, val_set, test_set

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers)#, collate_fn=custom_collate
    val_loader = DataLoader(val_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader

class AdditionalDataset(Dataset):
    def __init__(self, base_dataset, embeddings, transform=None):
        self.base_dataset = base_dataset
        self.embeddings = embeddings
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        items, label_ = self.embeddings[idx]
        assert label == label_
        return img, items, label
    
def get_pcam_dataloaders_with_embedding(args, batch_size=128, num_workers=8, shuffle_train = True, subset_frac=1.0, no_norm = False, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], return_dataset = False, teachers_cnt = None):
    """
    pcam
    """
    train_transform = [v2.Resize(224), v2.ToTensor()]#, v2.CenterCrop(224)
    test_transform = [ v2.Resize(224), v2.ToTensor(),]
    if not no_norm:
        train_transform.append(v2.Normalize(mean=mean, std=std))
        test_transform.append(v2.Normalize(mean=mean, std=std))
    train_transform = v2.Compose(train_transform)
    test_transform =  v2.Compose(test_transform)

    train_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform= test_transform,
                )
    
    val_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "val"
                )
    
    test_set = datasets.PCAM(
                    root="/export/livia/home/vision/Bkarimian/RLMTKD/",
                    download=False,
                    transform=test_transform,
                    split = "test"
                )
    
    train = EmbeddingLoader(teacher_names = args.teachers, base_path = args.base_dir, split = "train")
    test = EmbeddingLoader(teacher_names = args.teachers, base_path = args.base_dir, split = "test")
    val = EmbeddingLoader(teacher_names = args.teachers, base_path = args.base_dir, split = "val")
    
    train_loader = DataLoader(AdditionalDataset(train_set, train),
                              batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers)
    val_loader = DataLoader(AdditionalDataset(val_set, val),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(AdditionalDataset(test_set, test),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader
