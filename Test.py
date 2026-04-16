# -*- coding: utf-8 -*-
"""
test the model
"""

# Initialise
import torch
# import matplotlib.pyplot as plt
# import torchvision
import skimage
# import torchvision.transforms as transforms
import numpy as np
# from numpy import pi, exp
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import cv2
# import time
# from PIL import Image
# import scipy.ndimage as ndimage
# import torch.nn as nn
import os
from skimage import io, exposure
import glob
# import torchvision.transforms as transforms
# import argparse
from skimage.metrics import structural_similarity
from Models import get_model
# from Contrast import Get_contrast
from model_output.out4.Train import options


def main():
    opt = options()
    # three modes: if nch_in==9, input should have at least 9 frames
    # opt.nch_in = 9
    opt.out = 'model_output/out4'  # revise import
    opt.input_name = 'Experiment_data/argosim'
    # opt.input_name = 'Training_data/SIMdata1_mydata/BSC1cell'
    opt.n_test = 20
    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    opt.weights = opt.out + '/final.pth'
    opt.results = opt.out
    # n_raw = 16
    os.makedirs('%s' % opt.results, exist_ok=True)
    files = glob.glob('%s/*.tif' % opt.input_name)
    if opt.n_test <= len(files):
        files = files[:opt.n_test]
    # reconstruction
    net = load_model(opt)
    for n1, filename in enumerate(files):
        print('[%d/%d] Reconstructing %s' % (n1 + 1, len(files), filename))
        wide_image, ml_image = ml_lattice_sim_reconstruct(net, filename, opt)
        # target_image = (io.imread(filename)[-2])/255
        ml_image_uint = (255 * ml_image / np.max(ml_image)).astype('uint8')
        wide_image_uint = (255 * wide_image / np.max(wide_image)).astype('uint8')
        # print('SSIM_wide=%0.4f' % structural_similarity(wide_image, target_image, data_range=1))
        # print('SSIM_ml=%0.4f' % structural_similarity(ml_image, target_image, data_range=1))
        skimage.io.imsave('%s/test_wf_%d.jpg' % (opt.results, n1), wide_image_uint)
        skimage.io.imsave('%s/test_srml_%d.jpg' % (opt.results, n1), ml_image_uint)


def load_model(opt):
    print('Loading model')
    print(opt)
    net = get_model(opt)
    print('loading checkpoint', opt.weights)
    checkpoint = torch.load(opt.weights, map_location=opt.device)
    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # net.module.load_state_dict(state_dict)
    net.load_state_dict(state_dict)

    return net


def preprocess(stack, opt):  # preprocess for images, output input_image for model and wide_image

    n_images = stack.shape[0]
    if stack.shape[1] > opt.image_size or stack.shape[2] > opt.image_size:
        print('Over 512x512! Cropping')
        stack = stack[:, :512, :512]

    # 1 raw_image+1 wide_image
    # 1 raw_image+1 wide_image+pattern
    if n_images in (2, 3):
        wide_image = stack[1]
        input_images = stack[:n_images]

    # 1 raw_image+1 wide_image+otf+gt+pattern
    elif n_images == 5:
        wide_image = stack[1]
        if opt.nch_in == 2:
            input_images = stack[:2]
        elif opt.nch_in == 3:
            input_images = stack[[0, 1, -1]]
        else:
            print('no appropriate nch_in')
            input_images = stack[:opt.nch_in]

    # n raw_images
    elif n_images in (9, 16, 25):
        wide_image = np.mean(stack, 0)
        input_images = stack[:n_images]

    # n raw_images+1 wide_image+otf+gt+pattern
    elif n_images in (13, 20, 29):
        wide_image = stack[-4]
        if opt.task == 'wide_raw':
            input_images = stack[[0, -4]]
        elif opt.task == 'wide_raw_pattern':
            input_images = stack[[0, -4, -1]]
        elif opt.task == 'raw_pattern':
            input_images = stack[[0, -1]]
        elif opt.task == 'raws':
            if opt.nch_in == 3:
                input_images = stack[[0, 3, 6]]
            elif opt.nch_in == 4:
                input_images = stack[[0, 4, 8, 12]]
            elif opt.nch_in == 5:
                input_images = stack[[0, 5, 10, 15, 20]]
            else:
                input_images = stack[:opt.nch_in]
        elif opt.task == 'wide':
            input_images = stack[-4]
            input_images = np.reshape(input_images, (1, stack.shape[1], stack.shape[2]))
        else:
            print('no appropriate task')
            input_images = stack[:opt.nch_in]

    else:
        print('no appropriate input')
        wide_image = stack[0]
        input_images = stack[:n_images]

    input_images = input_images.astype('float') / np.max(input_images)  # used to be /255
    wide_image = wide_image.astype('float') / np.max(wide_image)

    if opt.norm == 'adapt_hist':
        for i in range(len(input_images)):
            input_images[i] = exposure.equalize_adapthist(input_images[i], clip_limit=0.001)
        wide_image = exposure.equalize_adapthist(wide_image, clip_limit=0.001)
        input_images = torch.from_numpy(input_images).float()
        wide_image = torch.from_numpy(wide_image).float()
    else:
        # normalise
        input_images = torch.from_numpy(input_images).float()
        wide_image = torch.from_numpy(wide_image).float()
        wide_image = (wide_image - torch.min(wide_image)) / (torch.max(wide_image) - torch.min(wide_image))

        if opt.norm == 'minmax':
            for i in range(len(input_images)):
                input_images[i] = (input_images[i] - torch.min(input_images[i])) / (
                        torch.max(input_images[i]) - torch.min(input_images[i]))

    return input_images, wide_image


def ml_lattice_sim_reconstruct(model, filename, opt):
    stack = io.imread(filename)
    input_image, wide_image = preprocess(stack, opt)
    wide_image = wide_image.numpy()

    with torch.no_grad():
        ml_image = model(input_image.unsqueeze(0).to(opt.device))
        ml_image = ml_image.cpu()
        ml_image = torch.clamp(ml_image, min=0, max=1)
    ml_image = ml_image.squeeze().numpy()  # squeeze the dimension (size 1)

    if opt.norm == 'adapt_hist':
        ml_image = exposure.equalize_adapthist(ml_image, clip_limit=0.01)

    return wide_image, ml_image


if __name__ == '__main__':
    main()
