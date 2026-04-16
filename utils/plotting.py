import torch
import matplotlib.pyplot as plt
# import torchvision
# import skimage
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
# import scipy.ndimage as ndimage
# import torch.nn as nn
# import os
from PerceptualLoss import VGGLoss, Perceptual_loss134
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.switch_backend('agg')

toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()
perceptual_loss = VGGLoss()
def testAndMakeCombinedPlots(net,loader,opt,idx=0):

    def PSNR_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        MSE = np.mean( (I0-I1)**2 )
        PSNR = 20*np.log10(1/np.sqrt(MSE))
        return PSNR

    def SSIM_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        return structural_similarity(I0, I1,data_range = 1)
        # return compare_ssim(I0, I1, multichannel=True)

    def vgg_loss(p0, p1):

        I0, I1 = toTensor(p0), toTensor(p1)
        I0 = (I0.repeat(1, 3, 1, 1)).to(device)
        I1 = (I1.repeat(1, 3, 1, 1)).to(device)
        loss = (perceptual_loss(I0, I1)).to("cpu")
        return np.array(loss)

    def calcScores(img, hr=None, makeplotBool=False, plotidx=0, title=None):
        if makeplotBool:
            plt.subplot(1,3,plotidx)
            plt.gca().axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img,cmap='gray')
        if not hr == None:
            psnr,ssim = PSNR_numpy(img,hr),SSIM_numpy(img,hr)
            vggloss = vgg_loss(img, hr)
            if makeplotBool: plt.title('%s (%0.2fdB/%0.3f/%0.2f)' % (title,psnr,ssim,vggloss))
            return psnr,ssim, vggloss
        if makeplotBool: plt.title(r'GT ($\infty$/1.000)')


    count, mean_bc_psnr, mean_sr_psnr, mean_bc_ssim, mean_sr_ssim, mean_bc_vggloss, mean_sr_vggloss = 0,0,0,0,0,0,0

    for i, bat in enumerate(loader):
        # lr(input lower) hr(gt) sr(output from net)
        lr_bat, hr_bat, wf_bat = bat[0], bat[1], bat[2]
        with torch.no_grad():
            sr_bat = net(lr_bat.to(opt.device))
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)): # loop over batch
            makeplotBool = (idx < 1 or (idx+1) % opt.plot_interval == 0 or idx == opt.n_epoch - 1) and count < opt.n_plot
            if opt.test: makeplotBool = True

            wf, sr, hr = wf_bat.data[j], sr_bat.data[j], hr_bat.data[j]

            sr = torch.clamp(sr,min=0,max=1) 

            # fix to deal with 3D deconvolution
            if opt.nch_out > 1:
                wf = wf[wf.shape[0] // 2] # channels are not for colours but separate grayscale frames, take middle
                sr = sr[sr.shape[0] // 2]
                hr = hr[hr.shape[0] // 2]

            ### Common commands
            # lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)
            wf, sr, hr = toPIL(wf), toPIL(sr), toPIL(hr)

            if opt.scale == 2:
                wf = wf.resize((1024,1024), resample=Image.BICUBIC)
                # bc = bc.resize((1024,1024), resample=Image.BICUBIC)
                hr = hr.resize((1024,1024), resample=Image.BICUBIC)

            if makeplotBool: plt.figure(figsize=(10,5),facecolor='white')
            bc_psnr, bc_ssim, bc_vggloss = calcScores(wf, hr, makeplotBool, plotidx=1, title='WF')
            sr_psnr, sr_ssim, sr_vggloss = calcScores(sr, hr, makeplotBool, plotidx=2, title='SR')
            calcScores(hr, None, makeplotBool, plotidx=3)
            
            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim
            mean_bc_vggloss += bc_vggloss
            mean_sr_vggloss += sr_vggloss

            if makeplotBool:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                plt.savefig('%s/combined_epoch%d_%d.png' % (opt.out,idx+1,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
                if opt.test:
                    wf.save('%s/lr_epoch%d_%d.png' % (opt.out,idx+1,count))
                    sr.save('%s/sr_epoch%d_%d.png' % (opt.out,idx+1,count))
                    hr.save('%s/hr_epoch%d_%d.png' % (opt.out,idx+1,count))

            count += 1
            if count == opt.n_test: break
        if count == opt.n_test: break
    
    summarystr = ""
    if count == 0: 
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. bc: %0.2f dB / %0.4f / %0.2f, sr: %0.2f dB / %0.4f / %0.2f \n' % (count, mean_bc_psnr / count, mean_bc_ssim / count, mean_bc_vggloss, mean_sr_psnr / count, mean_sr_ssim / count, mean_sr_vggloss)
    print(summarystr)
    print(summarystr,file=opt.fid)
    opt.fid.flush()
    if opt.log and not opt.test:
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx,t1,mem,mean_sr_psnr / count, mean_sr_ssim / count, mean_sr_vggloss / count, file=opt.test_stats)
        opt.test_stats.flush()


def generate_convergence_plots(opt,filename):
    fid = open(filename,'r')
    psnrlist = []
    ssimlist = []
    vggloss_list = []

    for line in fid:
        if 'sr: ' in line:
            psnrlist.append(float(line.split('sr: ')[1].split(' dB')[0]))
            ssimlist.append(float(line.split('sr: ')[1].split(' / ')[1]))
            vggloss_list.append(float(line.split('sr: ')[1].split(' /')[2]))
    
    plt.figure(figsize=(16,5),facecolor='white')
    plt.subplot(131)
    plt.plot(psnrlist,'.-')
    plt.title('PSNR')

    plt.subplot(132)
    plt.plot(ssimlist,'.-')
    plt.title('SSIM')

    plt.subplot(133)
    plt.plot(vggloss_list, '.-')
    plt.title('VGGLOSS')

    plt.savefig('%s/convergencePlot.png' % opt.out, dpi=300)