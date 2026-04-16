# coding: utf-8

import argparse
import Run


# Options
def options():

    opt = argparse.Namespace()
    
    opt.root = 'Training_data/SIMdata_mydata_square_ln,Training_data/SIMdata_mydata_line_ln,Training_data/SIMdata_mydata_hexagonal_ln'   # root of training data
    opt.out = 'model_output/tt'              # root of results
    opt.task = 'wide_raw_pattern'  # task (wide_raw, wide_raw_pattern, raws, wide)
    opt.norm = 'minmax'                     # minmax0.9 fra = 0.9

    # model
    opt.model = 'rcan'
    opt.lr = 1e-4    # decrease with the epoch increasing
    opt.log = False
    
    # data
    opt.image_size = 512
    opt.weights = ''                      # initial weight
    
    # computation 
    opt.workers = 0
    opt.batch_size = 1
    opt.cpu = False
    
    # input/output layer options
    opt.scale = 1                       # output size = imageSize * scale
    opt.nch_in = 3
    opt.nch_out = 1
    
    # architecture options 
    opt.n_arch = 0
    opt.n_res_blocks = 1
    opt.n_res_groups = 1
    opt.reduction = 2
    opt.n_feats = 4
    
    # training
    opt.n_train = 21
    opt.n_epoch = 5
    opt.disposableTrainingData = False
    opt.save_interval = 5
    opt.scheduler = '20,0.5'            # step_size, gamma
    
    # test options
    opt.n_test = 5
    opt.test_interval = 1
    opt.test = False
    opt.batch_size_test = 1
    opt.plot_interval = 1
    opt.n_plot = 3

    return opt


if __name__ == '__main__':
    opt = options()                   
    Run.main(opt)

