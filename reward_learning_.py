import argparse
from models.reward_model import PLRewardModel
from dataset.pcam import get_pcam_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from models.model_utils import get_models, models_size, models_mean, models_std
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import pytorch_lightning as pl
import torch
import os
from dataset.lightening_wrapper import PLDataModule
from torchvision.transforms import v2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dataset", type = str, default = "pcam")
    parser.add_argument("--teachers", nargs = "+", type = str, default = ["UNI", "DINOL14"])
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epoch", type = int, default = 3)
    parser.add_argument("--val_int", type = float, default = 0.1)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--dir", type=str, default='output')
    parser.add_argument("--checkpoint-dir", type=str, default='./chkpnt/teachers/checkpoints')
    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.dir
    output_name = args.output
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    teachers = []
    teachers_norm = []
    embed_dims = []
    for teacher in args.teachers:
        all_chkpnt = os.listdir(f"{args.checkpoint_dir}/{teacher}")
        for name in all_chkpnt:
            print(name, teacher)
            teachers.append(get_models(teacher, args.dataset, with_fc = True, num_classes = args.num_classes, file_name = f"{args.checkpoint_dir}/{teacher}/{name}").to("cuda" if torch.cuda.is_available() else "cpu"))
            teachers[-1].eval()
            teachers_norm.append(v2.Normalize(mean=models_mean[teacher], std=models_std[teacher]))
            embed_dims.append(models_size[teacher])
    
    logger=TensorBoardLogger(output_base_dir, name=output_name)
    if args.wandb:
        wandb.init(project=f"RLMTKD_{args.dataset}", reinit=True)
        wandb_logger = WandbLogger(log_model=False)
        logger = [wandb_logger, logger]
    
    
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_dataloaders("", batch_size=args.batch, num_workers=args.num_workers)
    elif args.dataset == 'pcam':
        train_loader, val_loader, test_loader = get_pcam_dataloaders("", batch_size=args.batch, num_workers=args.num_workers, no_norm=True, return_dataset = True, teachers_cnt = len(teachers))
    else:
        raise NotImplementedError(args.dataset)
    data = PLDataModule(args.batch, args.num_workers, train_loader, val_loader, test_loader)

    model = PLRewardModel(teachers, args.embed_dim, embed_dims, batch_size = args.batch, lr = args.lr, teachers_norm = teachers_norm)

    trainer = pl.Trainer(
            max_epochs=args.epoch,
            accelerator=device,
            devices=1,
            val_check_interval = args.val_int,        
            logger=logger,
            gradient_clip_val=args.clip, 
        )
    trainer.fit(model, data)
    print(trainer.checkpoint_callback.best_model_path)
    model = PLRewardModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, teachers = teachers, embed_dim = args.embed_dim, embed_dims = embed_dims, batch_size = args.batch, lr = args.lr)
    print(trainer.test(model=model, datamodule=data))

