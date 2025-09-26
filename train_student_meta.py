#Source code: https://github.com/Rorozhl/MMKD
from __future__ import print_function
import os
import re
import argparse
import time
import sys

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datetime
from utils import set_logger


from models import model_dict
from models.meta_util import LogitsWeight, MatchLogits, FeatureWeight, MatchFeature
from dataset.buffer import HardBuffer
from helper.meta_optimizer import MetaSGD
from helper.meta_loops import train_distill_multi_teacher as train, validate, validate_multi
from distiller_zoo import DistillKL
#from helper.util import adjust_learning_rate_cifar, save_dict_to_json, reduce_tensor, LAYER, adjust_meta_learning_rate
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.pcam import get_pcam_dataloaders
from helper.util import AverageMeter, accuracy, reduce_tensor, adjust_learning_rate, accuracy_list

from setting import teacher_model_path_dict
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)


split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print-freq', type=int, default=300, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--distill_decay', action='store_true', default=False,
                        help='distillation decay')

    parser.add_argument('--meta_warmup', type=int, default=0, help='meta_warmup')
    parser.add_argument('--meta_freq', type=int, default=5, help='meta_freq')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Initial learning rate for meta networks')
    parser.add_argument('--meta_wd', type=float, default=1e-4)
    parser.add_argument('--rollback', default=True, action="store_true", help='if roll_back')
    parser.add_argument('--hard_buffer', default=True, action="store_true", help='if hard buffer')
    parser.add_argument('--load_model', default=False, action="store_true", help='if hard buffer')
    parser.add_argument('--buffer_size', type=int, default=256, help='if hard buffer')
 
    # dataset
    parser.add_argument('--dataset', type=str, default='pcam', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67', 'pcam'], help='dataset')

    # model
    parser.add_argument('--arch', type=str, default='VITS8',
                        choices=['VITB16','VITB8','VITS16','VITS8','resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34', 'resnet8x4_double', 'MobileNetV2_Imagenet', 'ResNet18Double', 'ShuffleV2_Imagenet',
                                 'resnet8x4', 'resnet32x4', 'resnet20x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg8_imagenet', 'vgg16', 'vgg19', 'ResNet50', 'ShuffleV2_0_5', 'ResNet10',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'shufflenet_v2_x0_5'])
    parser.add_argument('--path-t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--path-s', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--teacher-name-list', default=['DINOL14', 'UNI'], type=str, nargs='+', help='teacher models')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'inter'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('--factor', default=2, type=int)
    parser.add_argument('--convs', action='store_true', default=False)

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--nesterov', action='store_true', help='if use nesterov')
    parser.add_argument('--preact', action='store_true', help='preact features')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[-2, 0, 1, 2, 3, 4])
    parser.add_argument('--c_embed', type=int)

    # multi teacher
    parser.add_argument("--teacher_num", type=int, default=1, help='use multiple teacher')
    parser.add_argument("--ensemble_method", default="CAMKD", type=str, choices=['AEKD', 'AVERAGE_LOSS', 'CAMKD', 'EBKD', 'META'])
    parser.add_argument('-C', type=float, default=0.6, help='torelance for disagreement among teachers')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--svm_norm', default=False, action="store_true", help='if use norm when compute with svm')

    # switch for edge transformation
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')

    opt = parser.parse_args()
    # set different learning rate from these 4 models
    if opt.arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = '../save/meta/1113_other/students/models'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))  

    opt.teacher_model_name = list(teacher_model_path_dict.keys())

    
    opt.teacher_name_str = "_".join(opt.teacher_name_list)
    print('args.teacher_name_str', opt.teacher_name_str)
    opt.teacher_num = len(opt.teacher_name_list)

    opt.model_name = opt.arch + '_'+ opt.dataset+ '_'+ 'MMKD'+'_'+ opt.trial+'_'+str(opt.teacher_num)+'_'+opt.teacher_name_str


    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = opt.model_name + info_time
    print(f'===> info is : {info}')
    opt.info = info
    opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, info)
    if not os.path.isdir(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {opt.checkpoint_dir}')

    opt.save_folder = opt.checkpoint_dir
    
    return opt

def load_teacher(model_path, n_cls, model_t, opt=None):
    print('==> loading teacher model')
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    model.eval()
    for t_n, t_p in model.named_parameters():
        t_p.requires_grad = False
    print('==> done')
    return model

def load_teacher_list(opt):
        print('==> loading teacher model list')
        teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], opt.n_cls, model_name, opt)
                            for model_name in opt.teacher_name_list]
        print('==> done')
        return teacher_model_list



def load_student_and_weight(model_s, WeightLogits, WeightFeature, FeatureMatch, opt):
    print('==> loading student model')
    map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    model_s.load_state_dict(torch.load(opt.path_s, map_location=map_location)['model'])
    WeightLogits.load_state_dict(torch.load(opt.path_s, map_location=map_location)['weight_logits'])
    WeightFeature.load_state_dict(torch.load(opt.path_s, map_location=map_location)['weight_inter'])
    FeatureMatch.load_state_dict(torch.load(opt.path_s, map_location=map_location)['feature_match'])
    cur_epoch = torch.load(opt.path_s, map_location=map_location)['epoch']
    return model_s, WeightLogits, WeightFeature, FeatureMatch, cur_epoch


total_time = time.time()
best_acc = 0
def main():
    
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.rank == 0 :
        opt.log_txt = os.path.join(opt.checkpoint_dir, opt.info + '.txt')
        opt.logger = set_logger(opt.log_txt)
        opt.logger.info("==========\nArgs:{}\n==========".format(opt))

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(opt.seed)

    
    if opt.dataset.startswith('cifar100'):
        opt.n_cls = 100
        data = (2, 3, 32, 32)
    elif opt.dataset.startswith('imagenet'):
        opt.n_cls = 1000
        data = (2, 3, 224, 224)
    elif opt.dataset.startswith('pcam'):
        opt.n_cls = 2
        data = (2, 3, 224, 224)
            
    model_t_list = load_teacher_list(opt)
    
    ##### load student model #####
    model_s = model_dict[opt.arch](num_classes=opt.n_cls).cuda()
        
    feat_t_list = []
    model_s.eval()
    model_s.cuda()
    for model_t in model_t_list:
        model_t.eval()
        model_t.cuda()
    for model_t in model_t_list:
        feat_t, _ = model_t(torch.rand(data).cuda(), is_feat=True)
        feat_t_list.append(feat_t)
    feat_s, _ = model_s(torch.rand(data).cuda(), is_feat=True)
    module_list = nn.ModuleList([])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T) 
    criterion_kd = DistillKL(opt.kd_T)

    WeightLogits = LogitsWeight(n_feature=opt.n_cls*(opt.teacher_num+1), teacher_num=opt.teacher_num).cuda()
    t_n = [t_feat[-2].shape[1] for t_feat in feat_t_list]
    WeightFeature = FeatureWeight(opt.batch_size, opt.teacher_num).cuda()
    weight_params = list(WeightLogits.parameters()) + list(WeightFeature.parameters())
    weight_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)
    FeatureMatch = MatchFeature(opt.teacher_num, feat_s[-2].shape[1], t_n, convs=opt.convs).cuda()
    
    for p in model_s.parameters(): p.requires_grad_(True)
    for p in FeatureMatch.parameters(): p.requires_grad_(True)
    model_s_params = list(model_s.parameters()) + list(FeatureMatch.parameters())
    model_s_optimizer = MetaSGD(model_s_params,
                        [model_s, FeatureMatch],
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay, nesterov=opt.nesterov, rollback=opt.rollback, cpu=False)
    
    cur_epoch = 1
    if opt.load_model:
        model_s, WeightLogits, WeightFeature, FeatureMatch, cur_epoch = load_student_and_weight(model_s, WeightLogits, WeightFeature, FeatureMatch, opt)
        

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    criterion_list.cuda()

    module_list.extend(model_t_list)
    module_list.cuda()
    model_s.cuda()

    if opt.dataset == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_dataloaders(data_folder=opt.dataset,
                                                        batch_size=opt.batch_size,
                                                        num_workers=opt.num_workers)
    elif opt.dataset == 'pcam':
        train_loader, val_loader, test_loader = get_pcam_dataloaders(opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

    train_state = {}
    if opt.hard_buffer:
        hardBuffer = HardBuffer(batch_size=opt.batch_size, buffer_size=opt.buffer_size)

    def inner_objective(data, is_avg=False, matching_only=False):
        input, target, _ = data
        input = input.float()

        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        feat_s, logit_s = model_s(input, is_feat=True, preact=opt.preact)
        feat_t_list = []
        logit_t_list = []
        with torch.no_grad():
            for model_t in model_t_list:
                feat_t, logit_t = model_t(input, is_feat=True, preact=opt.preact)
                feat_t = [f.detach() for f in feat_t]
                feat_t_list.append(feat_t)
                logit_t_list.append(logit_t.detach())

        loss_div_list = [criterion_div(logit_s, logit_t, is_ca=True)
                            for logit_t in logit_t_list]
        loss_div = torch.stack(loss_div_list, dim=1)
        logits_weight = WeightLogits(logit_t_list, logit_s.detach())
        loss_div = torch.mul(logits_weight, loss_div).sum(-1).mean()

        last_feat_t = [feat_t1[-2] for feat_t1 in feat_t_list]
        feature_weight = WeightFeature(last_feat_t, feat_s[-2].detach())
        # feature_weight = torch.ones(logits_weight.shape) / opt.teacher_num
        loss_kd = FeatureMatch(feat_s[-2], last_feat_t, feature_weight)

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 1))
        train_state['acc1'] = acc1
        train_state['acc5'] = acc5

        if matching_only:
            return opt.alpha*loss_div+opt.beta*loss_kd

        loss_cls = criterion_cls(logit_s, target)
        total_loss = loss_cls + loss_div + opt.beta*loss_kd
        train_state['total_loss'] = total_loss.item()

        if opt.hard_buffer:
            if not hardBuffer.is_full():
                hardBuffer.put(input, target)
            else:
                bo = (logit_s.argmax(1) != target)
                hardBuffer.update(input[bo], target[bo])

        return total_loss
    def outer_objective(data):
        input, target, _ = data
        input = input.float()

        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        feat_s, logit_s = model_s(input, is_feat=True, preact=opt.preact)
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 1))
        train_state['hard_acc1'] = acc1
        train_state['hard_acc5'] = acc5
        loss_cls = criterion_cls(logit_s, target)
        return loss_cls
    for epoch in range(cur_epoch, opt.epochs + 1):
        torch.cuda.empty_cache()

        #adjust_learning_rate_cifar(model_s_optimizer, epoch, opt)

        time1 = time.time()
        print("==> training...")
        model_s.train()
        # set teacher as eval()
        [model_t.eval() for model_t in module_list[-opt.teacher_num:]]

        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]
        criterion_kd = criterion_list[2]

        # model_t = module_list[-1]
        model_t_list = module_list[-opt.teacher_num:]

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        hard_top1 = AverageMeter()
        hard_top5 = AverageMeter()
        n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
        end = time.time()
        for idx, data in enumerate(train_loader):
            input, target, _ = data
            data_time.update(time.time() - end)
            if opt.batch_size > input.size(0): continue
            # ===================train student=====================

            model_s_optimizer.zero_grad()
            if epoch < opt.meta_warmup:
                inner_objective(data, is_avg=True).backward()
            else:
                inner_objective(data).backward()
            model_s_optimizer.step(None)

            # model_s_optimizer.zero_grad()
            # weight_optimizer.zero_grad()
            # inner_objective(data).backward()
            # model_s_optimizer.step(None)
            # weight_optimizer.step()

            losses.update(train_state['total_loss'], input.size(0))
            top1.update(train_state['acc1'][0], input.size(0))
            top5.update(train_state['acc5'][0], input.size(0))

            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} \t'
                  'Acc@1 {top1.val:.3f} \t'
                  'Acc@5 {top5.val:.3f} '.format(
                      epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

            # ===================train weight=====================
            if idx % opt.meta_freq == 0 or idx == n_batch - 1:
                if opt.hard_buffer:
                    hard_data = hardBuffer.sample()
                    # model_s_optimizer.zero_grad()
                    # model_s_optimizer.step(outer_objective, (hard_data, hard_label))
                    # hard_top1.update(train_state['hard_acc1'][0], input.size(0))
                    # hard_top5.update(train_state['hard_acc5'][0], input.size(0))
                    for j in range(2):               
                        model_s_optimizer.zero_grad()
                        model_s_optimizer.step(inner_objective, hard_data, matching_only=True)
                    
                    model_s_optimizer.zero_grad()
                    model_s_optimizer.step(outer_objective, hard_data)

                    model_s_optimizer.zero_grad()
                    weight_optimizer.zero_grad()
                    outer_objective(hard_data).backward()
                    model_s_optimizer.meta_backward()
                    weight_optimizer.step()
                else:
                    # model_s_optimizer.zero_grad()
                    # model_s_optimizer.step(outer_objective, data)
                    # hard_top1.update(train_state['hard_acc1'][0], input.size(0))
                    # hard_top5.update(train_state['hard_acc5'][0], input.size(0))                
                    for j in range(2):               
                        model_s_optimizer.zero_grad()
                        model_s_optimizer.step(inner_objective, data, matching_only=True)

                    model_s_optimizer.zero_grad()
                    model_s_optimizer.step(outer_objective, data)

                    model_s_optimizer.zero_grad()
                    weight_optimizer.zero_grad()
                    outer_objective(data).backward()
                    model_s_optimizer.meta_backward()
                    weight_optimizer.step()
            

        time2 = time.time()
        train_acc, train_acc_top5, train_loss, avg_time = top1.avg, top5.avg, losses.avg, data_time.avg
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}, Data {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1, avg_time))
            opt.logger.info('epoch:{}, train_acc: {}, train_loss: {}'.format(epoch, train_acc, train_loss))

        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)        

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            opt.logger.info('epoch:{}, test_acc: {}, test_loss: {}'.format(epoch, test_acc, test_loss))


            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                    'weight_logits': WeightLogits.state_dict(),
                    'weight_inter': WeightFeature.state_dict(),
                    'feature_match': FeatureMatch.state_dict()
                }

                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.arch))
                opt.logger.info('test_best_metrics: epoch:{}, test_acc: {}, test_loss: {}'.format(epoch, test_acc, test_loss))
                print('saving the best model!')
                torch.save(state, save_file)
    
    checkpoint = torch.load(save_file, map_location=torch.device('cpu'))
    model_s.load_state_dict(checkpoint['model'])    
    test_acc, test_acc_top5, test_loss = validate(test_loader, model_s, criterion_cls, opt)
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('test accuracy:', test_acc)
        opt.logger.info('test accuracy: {}'.format(test_acc))


    
if __name__ == '__main__':
    main()