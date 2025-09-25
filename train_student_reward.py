import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


from train_loops import train_avg, test
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
from setting import  teacher_model_path_dict
from dataset.pcam import get_pcam_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger
from models.util import Regress, TransFeat


class RewardModel(nn.Module):
    def __init__(self, embed_dim = 384, scale_factor=5.0, device = 'cuda'):
        super().__init__()
        self.scale_factor = scale_factor
        self.device = device

        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.reward_head.to(self.device)
        print("model created")
    
    def forward(self, embedding):
        reward = self.reward_head(embedding)
        return torch.tanh(reward.squeeze(-1) * self.scale_factor)




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')  # 32*2
                         
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')#0.001
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')


import models
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.005, type=float, help='learning rate')#0.001
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67', 'pcam'], help='dataset')
parser.add_argument('--trial', type=str, default='1', help='trial id')


def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path

def main():
    args = parser.parse_args()
    args.teacher_name_str = "_".join(args.teacher_name_list)
    print('args.teacher_name_str', args.teacher_name_str)
    args.teacher_num = len(args.teacher_name_list)

    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    print(f'===> info is : {info}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {args.checkpoint_dir}')

    if args.rank == 0 :
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))

    if args.seed is not None :
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu is not None :
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed # True
    print(f'======> args.distributed is {args.distributed}')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f'======> ngpus_per_node is {ngpus_per_node}') 
    

    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)  


def get_feat_trans(model, args):
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).cuda()
        
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        
    def load_teacher(model_path, n_cls, model_t, opt=None):
        model = model_dict[model_t](num_classes=n_cls).cuda()
        map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
        model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
        model.eval()
        for t_n, t_p in model.named_parameters():
            t_p.requires_grad = False
        return model


    def load_teacher_list(opt):
        print('==> loading teacher model list')
        teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, opt)
                            for model_name in opt.teacher_name_list]
        print('==> done')
        return teacher_model_list

    def get_teacher_feat_dims(teacher_models, args):
        teacher_num = len(teacher_models)
        x = torch.rand(args.res).cuda()
        feature_dims = []
        for t in teacher_models :
            feature, logits = t(x, is_feat=True)
            feature_dims.append(feature[-2].size())
            
        return feature_dims


    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    elif args.dataset.startswith('pcam'):
        args.n_cls = 2
        args.res = (1, 3, 224, 224)
            
    teacher_models = load_teacher_list(args)
    
    ##### load student model #####
    model = model_dict[args.arch](num_classes=args.n_cls).cuda()
    
    args.start_epoch = 0
    if len(args.resume) != 0:
        map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
        model_info_dict = torch.load(args.resume, map_location=map_location)
        model.load_state_dict(model_info_dict['model'])
        args.start_epoch = model_info_dict['epoch']
        
    print('======> load student model finish')
    args.t_feat_dims = get_teacher_feat_dims(teacher_models, args)

    feat_trans = get_feat_trans(model, args)
    print("===> get feat_trans finish......")
        
    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                for teacher in teacher_models :
                    teacher.cuda(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu]) 
                for teacher in teacher_models : 
                    teacher = torch.nn.parallel.DistributedDataParallel(teacher,device_ids=[args.gpu])
                
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                for teacher in teacher_models :
                    teacher = teacher.cuda()
                
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        for teacher in teacher_models :
            teacher = teacher.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ################### loss function and optimizer ###################
   
          
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_div = DistillKL(args.kd_T).to(device)
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(feat_trans)

    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)#0.001
    
    ################### load data ###################
    if args.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)
    elif args.dataset == 'pcam':
        train_loader, val_loader = get_pcam_dataloaders(args.data, batch_size=args.batch_size, num_workers=args.workers)
    ################### train model ###################
    best_acc = 0.  # best test accuracy
    
    '''t_results = []
    for t_model in teacher_models:
        acc = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
        t_results.append(round(acc, 2))
    args.logger.info('Teacher accruacy: '+ str(t_results))'''

    reward_model = RewardModel()
    checkpoint = torch.load("./chkpnt/reward_cifar100.ckpt")['state_dict']
    checkpoint = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in checkpoint.items()}
    reward_model.load_state_dict(checkpoint)
    reward_model.to(device)
    reward_model.eval()
    for epoch in range(args.start_epoch, args.epochs) :
        train_avg(train_loader, model, criterion_list, optimizer, epoch, device, args, feat_trans, teacher_models, reward_model)
        acc = test(epoch, model, device, val_loader, criterion_ce, args)

        if args.rank == 0 :
            state = {
                    'epoch' : epoch + 1,
                    'arch' : args.arch, 
                    'model': model.module.state_dict() if args.distributed else model.state_dict() ,
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
            }

            torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True
            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

    
    if args.rank == 0 :
        args.logger.info('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        top1_acc = test(epoch, model, device, val_loader, criterion_ce, args)
        args.logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir,  args.arch + '_best.pth.tar')))

if __name__ == '__main__' :
     main()





    


