import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .VIT import VIT

__all__ = ['VITB8']


class VITB8(VIT):
    def __init__(self, num_classes):
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb16", trust_repo=True)
        image_embed_size = 768
        super().__init__(num_classes, model, image_embed_size)
            


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 224, 224)
    net = VITB8(num_classes=100)
    logit = net(x)
    from util import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 224, 224)) / 1e6))