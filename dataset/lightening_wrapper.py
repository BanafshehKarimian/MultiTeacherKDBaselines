from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import open_clip
import torch
class PLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, training_data, valid_data, test_data):
        super().__init__()
        self.training_data = training_data
        self.valid_data = valid_data        
        self.test_data = test_data
        

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)