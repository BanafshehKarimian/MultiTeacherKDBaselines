from __future__ import print_function, division

import sys
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .util import AverageMeter, accuracy, reduce_tensor, adjust_learning_rate, accuracy_list


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    # Create a GradScaler for mixed precision training
    #scaler = amp.GradScaler()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        if opt.dataset == 'imagenet':
            adjust_learning_rate(optimizer, epoch, idx, len(train_loader), opt.learning_rate)
        if opt.dali is None:
            input, target, raw_inputs = batch_data
        else:
            input, target = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        data_time.update(time.time() - end)
        
        input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, target, topk=(1, 1))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, opt):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            if opt.dali is None:
                input, target = batch_data
            else:
                input, target = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 1))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg
