import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.pcam import get_pcam_dataloaders
from models.model_utils import (
    ModelWrapper,
    models,
    models_mean,
    models_std,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def get_logits_embeddings(
    loader: DataLoader,
    model: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a model on a DataLoader and return logits, labels, and embeddings as numpy arrays.
    """
    labels, logits, embds = [], [], []

    model.to(DEVICE)
    model.eval()
    torch.backends.cudnn.benchmark = True

    for batch in tqdm(loader, desc="Extracting", leave=False):
        img, label = batch[0], batch[1]
        embd, logit = model(img.to(DEVICE))
        labels.append(label.cpu())
        logits.append(logit.cpu())
        embds.append(embd.cpu())

    logits_np = torch.cat(logits, dim=0).numpy()
    labels_np = torch.cat(labels, dim=0).numpy()
    embds_np = torch.cat(embds, dim=0).numpy()
    return logits_np, labels_np, embds_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, default=["UNI"])
    parser.add_argument("--base", type=str, default="./chkpnt/teachers/checkpoints/")
    parser.add_argument("--base-dir", type=str, default="./chkpnt/teachers/log")
    parser.add_argument("--output", type=str, default="chkpnt")  # (kept, unused)
    parser.add_argument("--checkpoints", nargs="+", type=str, default=[""])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--sfx", type=str, default="")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for model_name, ckpt in zip(args.models, args.checkpoints):
        # Load data (use the correct mean/std for each model)
        train_loader, val_loader, test_loader = get_pcam_dataloaders(
            data_folder="",
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle_train=False,
            mean=models_mean[model_name],
            std=models_std[model_name],
        )

        # Init model and load checkpoint
        net = models[model_name]()
        ckpt_path = Path(args.base) / ckpt
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        key = "state_dict" if "state_dict" in checkpoint else "model"
        state = {k.replace("model.", ""): v for k, v in checkpoint[key].items()}

        # Infer in_features for the final FC and replace it to match num_classes
        if "fc.weight" not in state:
            raise KeyError("Checkpoint is missing 'fc.weight'. Cannot infer in_features.")
        in_features = state["fc.weight"].shape[1]
        net.fc = torch.nn.Linear(in_features, args.num_classes)

        net.load_state_dict(state, strict=True)
        model = ModelWrapper(net)

        # --- TEST ---
        yp, y, embd = get_logits_embeddings(test_loader, model)
        np.save(f"{base_dir}/log_labels_test.npy", y)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_test_logit.npy", yp)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_test_embedding.npy", embd)

        # --- VAL ---
        yp, y, embd = get_logits_embeddings(val_loader, model)
        np.save(f"{base_dir}/log_labels_val.npy", y)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_val_logit.npy", yp)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_val_embedding.npy", embd)


        # --- TRAIN ---
        yp, y, embd = get_logits_embeddings(train_loader, model)
        np.save(f"{base_dir}/log_labels_train.npy", y)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_train_logit.npy", yp)
        np.save(f"{base_dir}/log_{model_name}{args.sfx}_train_embedding.npy", embd)

if __name__ == "__main__":
    main()
