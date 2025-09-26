import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from models.model_utils import get_models, models_size
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch import optim
import open_clip
import pytorch_lightning as pl
import open_clip
from PIL import Image
from torchmetrics.functional import auroc
import timm
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
class RewardModel(nn.Module):
    def __init__(self, teachers, embed_dim, embed_dims=[1024], scale_factor=5.0, device = 'cuda'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mappers = {dim:nn.Linear(dim, embed_dim) for dim in embed_dims}
        self.device = device
        self.teachers = teachers
        for teacher in self.teachers:
            teacher.eval()
            teacher.to(self.device)
            for param in teacher.parameters():
                param.requires_grad = False

        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.reward_head.to(self.device)
        print("model created")


    def mapp(self, x):
        self.mappers[x.shape[1]].to(self.device)
        embedding = self.mappers[x.shape[1]](x.to(self.device))
        return embedding
    
    def forward(self, embedding):
        reward = self.reward_head(embedding)
        return torch.tanh(reward.squeeze(-1) * self.scale_factor)

    def compute_loss(self, rewards_chosen, rewards_rejected):
        return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

    def get_embeddings(self, imgs, y, confidence = True):
        emb = []
        value = []
        for (img, teacher) in zip(imgs, self.teachers):
            with torch.no_grad():
                k, v = teacher(img.to(self.device))
            emb.append(self.mapp(k))
            v = torch.softmax(v, axis = 1)
            value.append([v[i][int(j.item())].item() for (i,j) in zip(range(len(v)), y)])
        chosen = torch.zeros_like(emb[0])
        rejected = torch.zeros_like(emb[0])
        for i in range(len(value[0])):
            best = np.argmax([value[0][i], value[1][i]])
            chosen[i] = emb[best][i]
            rejected[i] = emb[1-best][i]
        return chosen, rejected


    def process_batch(self, batch):
        img = batch[0]#.to(self.device)
        lab = batch[1].to(self.device)
        chosen, rejected = self.get_embeddings(img, lab)
        chosen = chosen.to(self.device)
        rejected = rejected.to(self.device)
        rewards_chosen = self.forward(chosen)
        rewards_rejected = self.forward(rejected)
        loss = self.compute_loss(rewards_chosen, rewards_rejected)
        return rewards_chosen, rewards_rejected, loss




class PLRewardModel(pl.LightningModule):
    def __init__(self, teachers, embed_dim, embed_dims = [1024], batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, scheduler = False):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.rewards_chosen = []
        self.rewards_rejected = []
        self.model = RewardModel(teachers=teachers, embed_dim = embed_dim, embed_dims=embed_dims)
        self.model.to(self.model.device)

        print("model created")
        print(self.device)
        


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max = 12500, eta_min = 0.0001)
        if self.scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
    

    def forward(self, x):
        return self.model.forward(x)

    def process_batch(self, batch):
        return self.model.process_batch(batch)

    def training_step(self, batch, batch_idx):
        rewards_chosen, rewards_rejected, loss = self.process_batch(batch)
        self.rewards_chosen.append(rewards_chosen)
        self.rewards_rejected.append(rewards_rejected)
        self.log('train_loss', loss, batch_size=self.batch_size)        
        return loss

    def validation_step(self, batch, batch_idx):
        rewards_chosen, rewards_rejected, loss = self.process_batch(batch)
        self.rewards_chosen.append(rewards_chosen)
        self.rewards_rejected.append(rewards_rejected)
        self.log('val_loss', loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        rewards_chosen, rewards_rejected, loss = self.process_batch(batch)
        self.rewards_chosen.append(rewards_chosen)
        self.rewards_rejected.append(rewards_rejected)
        self.log('test_loss', loss, batch_size=self.batch_size)

if __name__=="__main__":
    model = RewardModel([get_models("vitl14", "pcam"), get_models("vitb16", "pcam")], embed_dim = 1024, embed_dims = [models_size["vitl14"], models_size["vitb16"]])
    print(model.process_batch((torch.zeros((64, 3, 224, 224)),torch.zeros((64, 1)))))