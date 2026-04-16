# import math
import os
import torch
import torch.nn as nn
import time
# import numpy as np
import torch.optim as optim
# import torchvision
# from torch.autograd import Variable
# import subprocess
from Models import get_model
from Data_Handler import get_data_loader

from plotting import testAndMakeCombinedPlots, generate_convergence_plots

from torch.utils.tensorboard import SummaryWriter
# from MS_SSIM_L1 import SSIM
# from PerceptualLoss import VGGLoss

# from options import parser
# import traceback
# import socket
# from datetime import datetime
import shutil
import sys
import glob


def train(opt, data_loader, valid_loader, net):
    start_epoch = 0
    # define loss function
    loss_function = nn.MSELoss()
    # loss_function = SSIM(data_range=255)
    # loss_function = VGGLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()
    if len(opt.weights) > 0:  # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint', opt.weights)

        net.load_state_dict(checkpoint['state_dict'])
        if opt.lr == 1:  # continue as it was
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
        # verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        step_size, gamma = int(opt.scheduler.split(
            ',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
        if len(opt.weights) > 0:
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.n_epoch):
        count = 0
        mean_loss = 0

        # for param_group in optimizer.param_groups:
        #     print('\nLearning rate', param_group['lr'])

        for i, bat in enumerate(data_loader):
            lr, hr = bat[0], bat[1]
            optimizer.zero_grad()

            sr = net(lr.to(opt.device))
            loss = loss_function(sr, hr.to(opt.device))
            # loss = loss_function(sr.repeat(1, 3, 1, 1), (hr.to(opt.device)).repeat(1, 3, 1, 1))

            loss.backward()
            optimizer.step()

            # Status and display
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch + 1, opt.n_epoch,
                                                    i + 1, len(data_loader), loss.data.item()), end='')

            count += 1
            if opt.log and count * opt.batch_size // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                opt.writer.add_scalar(
                    'data/mean_loss_per_1000', mean_loss / count, epoch)
                opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                print(epoch, count * opt.batchSize, t1, mem,
                      mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0

        # Scheduler
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate', param_group['lr'])
                break

        # Printing
        mean_loss = mean_loss / len(data_loader)
        t1 = time.perf_counter() - opt.t0
        eta = (opt.n_epoch - (epoch + 1)) * t1 / (epoch + 1)
        o_str = '\nEpoch [%d/%d] done, mean loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
            epoch + 1, opt.n_epoch, mean_loss, t1, eta)
        print(o_str)
        print(o_str, file=opt.fid)
        opt.fid.flush()
        if opt.log:
            opt.writer.add_scalar(
                'data/mean_loss', mean_loss / len(data_loader), epoch)

        # TEST
        if (epoch + 1) % opt.test_interval == 0:
            testAndMakeCombinedPlots(net, valid_loader, opt, epoch)

        if (epoch + 1) % opt.save_interval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if len(opt.scheduler) > 0:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch + 1))

    checkpoint = {'epoch': opt.n_epoch,
                  'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict()}
    if len(opt.scheduler) > 0:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, opt.out + '/final.pth')


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return torch.mean(torch.pow((x - y), 2))


def main(opt):
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    os.makedirs(opt.out, exist_ok=True)
    shutil.copy2('Train.py', opt.out)
    # os.rename(opt.out+'/Train.py',opt.out+'/Train_option.txt')

    opt.fid = open(opt.out + '/log.txt', 'w')

    o_str = 'ARGS: ' + ' '.join(sys.argv[:])
    print(opt, '\n')
    print(opt, '\n', file=opt.fid)
    print('\n%s\n' % o_str)
    print('\n%s\n' % o_str, file=opt.fid)

    print('getting dataloader', opt.root)
    data_loader, valid_loader = get_data_loader(opt)

    if opt.log:
        opt.writer = SummaryWriter(log_dir=opt.out, comment='_%s_%s' % (
            opt.out.replace('\\', '/').split('/')[-1], opt.model))
        opt.train_stats = open(opt.out.replace(
            '\\', '/') + '/train_stats.csv', 'w')
        opt.test_stats = open(opt.out.replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print('iter,n_sample,time,memory,mean_loss', file=opt.train_stats)
        print('iter,time,memory,psnr,ssim', file=opt.test_stats)

    t0 = time.perf_counter()
    net = get_model(opt)

    if not opt.test:
        train(opt, data_loader, valid_loader, net)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(opt.weights) > 0:  # load previous weights?
            checkpoint = torch.load(opt.weights)
            print('loading checkpoint', opt.weights)
            net.load_state_dict(checkpoint['state_dict'])
            print('time: %0.1f' % (time.perf_counter() - t0))
        testAndMakeCombinedPlots(net, valid_loader, opt)

    opt.fid.close()
    if not opt.test:
        generate_convergence_plots(opt, opt.out + '/log.txt')

    print('time: %0.1f' % (time.perf_counter() - t0))

    # optional clean up
    if opt.disposableTrainingData and not opt.test:
        print('deleting training data')
        # preserve a few samples
        os.makedirs('%s/training_data_subset' % opt.out, exist_ok=True)

        sample_count = 0
        for file in glob.glob('%s/*' % opt.root):
            if os.path.isfile(file):
                basename = os.path.basename(file)
                shutil.copy2(file, '%s/training_data_subset/%s' % (opt.out, basename))
                sample_count += 1
                if sample_count == 10:
                    break
        shutil.rmtree(opt.root)
