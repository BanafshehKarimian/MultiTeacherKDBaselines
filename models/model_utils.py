import torch
import timm
import torch.nn as nn
from copy import deepcopy
from models.regnet import *
models = {
                "DINOL14": lambda: torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True),
                "vits16": lambda: torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vits16", trust_repo=True),
                "vits8": lambda: torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vits8", trust_repo=True),
                "vitb16": lambda: torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb16", trust_repo=True),
                "vitb8": lambda: torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8", trust_repo=True),
                "UNI": lambda: timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True),

            }
classifiers = {
                "RegNetY_400MF": lambda: RegNetY_400MF,
                "RegNetX_400MF": lambda: RegNetX_400MF
            }

models_size = {
                "DINOL14": 1024,
                "vits16": 384,
                "vits8": 384,
                "vitb16": 768,
                "vitb8": 768,
                "UNI": 1024,
                "RegNetY_400MF": 384,
                "RegNetX_400MF": 384,
            }

class ModelWrapper(nn.Module):
    def __init__(self, model, layer = "fc"):
        super().__init__()
        self.model = deepcopy(model)

        self.fc = deepcopy(getattr(self.model, layer))
        setattr(self.model, layer, nn.Identity())
        for name, param in self.model.named_parameters():
            if name not in ["fc0.weight", "fc0.bias"]:
                param.requires_grad = False

    def forward(self, x):
        features = self.model(x)
        out = self.fc(features)
        return features, out

def get_models(name, dataset = None, with_fc = True, add_layer = True, num_classes = 2):
    if name in models.keys():
        model = models[name]()
        if dataset:
            checkpoint = torch.load(f"/export/livia/home/vision/Bkarimian/baseline/MTKD-RL/chkpnt/teachers/models/{name}_{dataset}_lr_0.005_decay_0.0005_trial_0/{name}_best.pth")
            checkpoint['model'] = {k.replace("model.",""): v for k,v in checkpoint['model'].items()}
            if with_fc:
                in_features = checkpoint['model']['fc.weight'].shape[1]
                '''if add_layer:
                    model.fc0 = torch.nn.Linear(in_features, in_features)
                    model.fc_act = torch.nn.GELU()'''
                model.fc = torch.nn.Linear(in_features, num_classes)
                '''for name, param in model.named_parameters():
                    print(name)
                print(checkpoint['model'].keys())'''
                model.load_state_dict(checkpoint['model'], strict = True)
                model = ModelWrapper(model)
            else:
                model.load_state_dict(checkpoint['model'], strict = True)
        else:
            in_features = model(torch.zeros((64, 3, 224, 224))).shape[1]
            if with_fc:
                model.fc = torch.nn.Linear(in_features, num_classes)
                model = ModelWrapper(model)
        return model
    model = classifiers[name]()(num_classes = num_classes)
    checkpoint = torch.load(f"./models/checkpoints/{name}_best.pth", weights_only = False)
    model.load_state_dict(checkpoint['model'], strict = True)
    return ModelWrapper(model, layer = "linear")