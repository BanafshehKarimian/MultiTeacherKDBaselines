from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import open_clip
import torch
import numpy as np
import os
from pathlib import Path
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
class EmbeddingLoader(Dataset):
    def __init__(self, teacher_names, base_path, split = "train", dtype=torch.float32):
        super().__init__()
        self.teacher_paths = sorted([str(p) for p in Path(base_path).glob(f"*_{split}_embedding.npy") if any(tname in p.name for tname in teacher_names)])
        self.split = split
        self.embd_arrays = None      
        self.logit_arrays = None      
        self.scores_arrays = None     
        self.dtype = dtype
        z0 = np.load(self.teacher_paths[0], mmap_mode="r", allow_pickle=False)
        self._N = z0.shape[0]

        zl =  np.load(f"{base_path}/log_labels_{split}.npy")
        self.labels = np.asarray(zl).astype(np.int64)
        self.embedding_size = self._get_size()

    def __len__(self):
        return self._N
    
    def _get_size(self):
        embd_arrays = [np.load(pth, mmap_mode="r") for pth in self.teacher_paths]
        return [embd.shape[1] for embd in embd_arrays]
    
    def _worker_lazy_loader(self):
        if self.embd_arrays is not None:
            return
        self.embd_arrays = [np.load(pth, mmap_mode="r") for pth in self.teacher_paths]
        self.logit_arrays = [np.load(pth.replace("embedding", "logit"), mmap_mode="r") for pth in self.teacher_paths]
        self.scores_arrays = [np.load(pth.replace("embedding", "scores_test"), mmap_mode="r") for pth in self.teacher_paths]

    def __getitem__(self, idx):
        try:
            self._worker_lazy_loader()
            items = []
            for i in range(len(self.embd_arrays)):
                emb = torch.from_numpy(self.embd_arrays[i][idx].copy())
                logit = torch.from_numpy(self.logit_arrays[i][idx].copy())
                score = self.scores_arrays[i][idx]
                items.append((emb, logit, score))  
            '''for embs in self.embedding_size:
                emb = torch.rand(embs)
                logit = torch.rand(2)
                items.append((emb, logit))'''
            return items, self.labels[idx]
        except Exception as e:
            raise RuntimeError(f"Dataset error at idx={idx}: {e}") from e
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None  # do not pickle open NpzFile handles
        return state
    
if __name__=="__main__":
    loader = EmbeddingLoader(["vits8", "vitb8"], "./chkpnt/teachers/logs")
    print(len(loader), loader[0], loader[-1], loader.embedding_size)