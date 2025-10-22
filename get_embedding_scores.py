import argparse
import numpy as np

import torch

import torch.nn.functional as F
import seaborn as sns
import pandas as pd

def plot_histograms_displot(data, bins=30, col_wrap=3, density=False, save_path = None):
    df = pd.DataFrame(
        [(str(k), float(x)) for k, v in data.items() for x in np.ravel(v)],
        columns=["key", "value"]
    )
    g = sns.displot(
        df, x="value", col="key", col_wrap=col_wrap, kind="hist",
        bins=bins, stat=("density" if density else "count"),
        facet_kws={"sharey": True}
    )
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches="tight")
    return g

def get_logits_as_score(args):
    embeddings = {}
    scores = {}
    for split in ["train", "test", "val"]:
        embeddings[split] = np.load(f"{args.base_dir}/log_{args.model}{args.sfx}_{split}_embedding.npy")
        logit =  torch.tensor(np.load(f"{args.base_dir}/log_{args.model}{args.sfx}_{split}_logit.npy"))
        labels = torch.tensor(np.load(f"{args.base_dir}/log_labels_{split}.npy"))
        if args.norm:
            logit = F.softmax(logit, dim=-1)
        scores[split] = logit.gather(1, labels.view(-1, 1)).squeeze(1)
    return scores

def get_score1(args):
    device = args.device
    noise_std = float(getattr(args, "noise_std", 1))
    T = args.noise_trials#int(getattr(args, "noise_trials", 10))
    embeddings = {}
    scores = {}

    model_chkpt = torch.load(f"./chkpnt/teachers/checkpoints/pcam_{args.model}{args.sfx}.ckpt")
    key = "state_dict" if "state_dict" in model_chkpt else "model"
    sd = {k.replace("model.", ""): v for k, v in model_chkpt[key].items()}

    fc_weight = sd["fc.weight"].to(device=device, dtype=torch.float32)
    fc_bias   = sd["fc.bias"].to(device=device, dtype=torch.float32)    
    C, D = fc_weight.shape
    fc = torch.nn.Linear(D, C, bias=True).to(device)
    with torch.no_grad():
        fc.weight.copy_(fc_weight)
        fc.bias.copy_(fc_bias)
    fc.eval()
    for split in ["train", "test", "val"]:
        emb_np = np.load(f"{args.base_dir}/log_{args.model}{args.sfx}_{split}_embedding.npy")
        emb = torch.as_tensor(emb_np, dtype=torch.float32, device=device)
        orig_logits =  torch.tensor(np.load(f"{args.base_dir}/log_{args.model}{args.sfx}_{split}_logit.npy"), dtype=torch.float32, device=device)
        labels = torch.tensor(np.load(f"{args.base_dir}/log_labels_{split}.npy"), dtype=torch.long, device=device)
        orig_pred = F.softmax(orig_logits, dim=-1).argmax(dim=-1)
        noisy_logits_list = []
        noisy_pred_list = []
        noisy_acc_list = []

        with torch.no_grad():
            for _ in range(T):
                noise = torch.randn_like(emb) * noise_std
                emb_noisy = emb + noise
                logits_noisy = fc(emb_noisy)                 
                preds_noisy = logits_noisy.argmax(dim=-1)    

                noisy_logits_list.append(logits_noisy)
                noisy_pred_list.append(preds_noisy)
        
        noisy_pred = torch.stack(noisy_pred_list, dim=0)
        mask_flip = (noisy_pred == orig_pred.unsqueeze(0).expand(T, -1)).to(torch.float32)
        mask_correct = (orig_pred == labels).to(torch.float32)
        print(mask_correct.mean())
        scores[split] = (mask_correct*mask_flip).mean(dim=0).cpu()
    return scores

def save(data, args):
    for split in data.keys():
        np.save(f"{args.base_dir}/log_{args.model}{args.sfx}_{split}_{args.output}", data[split])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="./chkpnt/teachers/log")
    parser.add_argument("--model", type=str, default="DINOL14")
    parser.add_argument("--sfx", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scoring-type", type=str, default="logits")
    parser.add_argument("--noise-trials", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=1.0)
    args = parser.parse_args()

    score_functions = {"logits":get_logits_as_score, "probs":get_logits_as_score, "test": get_score1}
    args.norm = False
    if args.scoring_type in ["probs"]:
        args.norm = True
    args.output = f"scores_{args.scoring_type}.npy"
    args.img = f"./vis/{args.scoring_type}_hist_{args.model}{args.sfx}.png"
    score = score_functions[args.scoring_type](args)
    plot_histograms_displot(score, save_path = args.img)
    save(score, args)