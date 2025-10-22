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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple

class RewardModel(nn.Module):
    """
    Reward model with per-teacher linear mappers → shared embedding → scalar reward.
    Expects each batch to provide a list of (embedding, logits) pairs (one per teacher)
    and ground-truth labels for score extraction.
    """
    def __init__(
        self,
        embed_dim: int,
        embed_dims: Sequence[int] = (1024,),
        scale_factor: float = 5.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dims: List[int] = list(embed_dims)
        self.scale_factor = scale_factor
        self.num_classes = num_classes

        # Register teacher mappers properly so optimizer sees them
        self.mappers = nn.ModuleList([nn.Linear(d, embed_dim) for d in self.embed_dims])

        # Keep your simple head (you can swap to a stronger head later)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )

    # ---- Core ops ----
    def mapp(self, x: torch.Tensor, teacher_idx: int) -> torch.Tensor:
        """Map a teacher-specific embedding x → shared embed_dim."""
        return self.mappers[teacher_idx](x)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return scaled tanh reward in [-tanh(scale), +tanh(scale)]. Shape: [B]."""
        reward = self.reward_head(embedding)              # [B,1]
        return reward.squeeze(-1)#torch.tanh( * self.scale_factor)

    # ---- Loss & scoring ----
    '''@torch.no_grad()
    def get_score(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Extract the probability of the ground-truth class from logits.
        logits: [B, C], labels: [B]
        Returns: [B]
        """
        probs = F.softmax(logits, dim=-1)
        return probs.gather(1, labels.view(-1, 1)).squeeze(1)'''

    def compute_loss(self, scores: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        perm = torch.argsort(scores, dim=1, descending=True)
        r_sorted = rewards.gather(1, perm)
        denom = torch.logcumsumexp(r_sorted.flip(1), dim=1).flip(1)
        loss_per = (denom - r_sorted).sum(dim=1)
        return loss_per.mean()

    # ---- Batch API ----
    def get_scores_and_rewards(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],  # list of (embd:[B,D_i], logits:[B,C])
        labels: torch.Tensor,                            # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(pairs)
        scores_list, rewards_list = [], []
        for i, (embd_i, logits_i, s_i) in enumerate(pairs):
            r_i = self.forward(self.mapp(embd_i, i))     # [B]
            scores_list.append(s_i)
            rewards_list.append(r_i)

        scores = torch.stack(scores_list, dim=1)         # [B, T]
        rewards = torch.stack(rewards_list, dim=1)       # [B, T]
        return scores, rewards

    def process_batch(
        self,
        batch: Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairs, labels = batch
        scores, rewards = self.get_scores_and_rewards(pairs, labels)
        loss = self.compute_loss(scores, rewards)
        return scores, rewards, loss



class PLRewardModel(pl.LightningModule):
    def __init__(self, embed_dim, embed_dims = [1024], batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, scheduler = False):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.model = RewardModel(embed_dim = embed_dim, embed_dims=embed_dims)
        #self.model.to(self.model.device)

        print("model created")
        


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
        _, _, loss = self.process_batch(batch)
        self.log('train_loss', loss, batch_size=self.batch_size)        
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self.process_batch(batch)
        self.log('val_loss', loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        _, _, loss = self.process_batch(batch)
        self.log('test_loss', loss, batch_size=self.batch_size)

if __name__=="__main__":
    model = RewardModel([get_models("vitl14", "pcam"), get_models("vitb16", "pcam")], embed_dim = 1024, embed_dims = [models_size["vitl14"], models_size["vitb16"]])
    print(model.process_batch((torch.zeros((64, 3, 224, 224)),torch.zeros((64, 1)))))