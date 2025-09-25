import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

__all__ = ['VIT']


class VIT(nn.Module):
    def __init__(self, num_classes, model, image_embed_size):
        super(VIT, self).__init__()
        self.model = model
        self.image_embed_size = image_embed_size
        self.fc = nn.Linear(self.image_embed_size, num_classes)
        #for param in self.model.parameters():
        #    param.requires_grad = False


    def forward(self, x, is_feat=False, preact=False):
        f0 = self.model.forward_features(x)
        f1 = self.model.forward_head(f0)
        out = self.fc(f1)        
        if is_feat:
            return [f0[:, 1:, :].mean(dim=1), f1], out
        else:
            return out
