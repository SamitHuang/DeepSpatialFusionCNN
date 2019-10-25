from __future__ import print_function
import os
import torch
import argparse

net_type="resnet18" 
dataset="2018"
fold_idx_def=1
PATH_PREFIX="./" 

class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Classification of breast cancer histology')
        parser.add_argument('--fold-index', type=int, default=fold_idx_def)
        parser.add_argument('--net-type', type=str, default=net_type,  help='type of network, resnet18/50, densenet')
        parser.add_argument('--network', type=str, default='0', help='train patch-wise network: 1, image-wise network: 2 or both: 0 (default: 0)')
        parser.add_argument('--patches-overlap', type=int, default=0, help='when training or testing image-wise model, dividing patches in a overlap or non-overlap way' )

        args_part = parser.parse_args()
        fold_idx = args_part.fold_index
        
        DEFAULT_DATASET_PATH = PATH_PREFIX + "dataset" + dataset + "/fold" + str(fold_idx) 
        DEFAULT_TEST_PATH = PATH_PREFIX + "dataset" + dataset + "/fold" + str(fold_idx) + "/validation/all" 
        
        DEFAULT_CHK_PNT_PATH = PATH_PREFIX + "checkpoints/"+""+net_type+"_DNNvoter"+dataset+"_fold"+str(fold_idx)

        parser.add_argument('--dataset-path', type=str, default=DEFAULT_DATASET_PATH,  help='dataset path (default: ./dataset)')
        parser.add_argument('--testset-path', type=str, default=DEFAULT_TEST_PATH, help='fill_e or directory address to the test set')
        parser.add_argument('--checkpoints-path',type=str, default=DEFAULT_CHK_PNT_PATH,help='models are saved here')
        parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
        parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
        parser.add_argument('--patch-stride', type=int, default=256, metavar='N', help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
        parser.add_argument('--beta1', type=float, default=0.9, metavar='M', help='Adam beta1 (default: 0.9)')
        parser.add_argument('--beta2', type=float, default=0.999, metavar='M', help='Adam beta2 (default: 0.999)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--ensemble', type=bool, default=False, help='whether to use model ensemble on test-set prediction (default: 1)')
        parser.add_argument('--debug', type=int, default=0, help='debugging (default: 0)')
        parser.add_argument('--channels', type=int, default=1, help='number of channels created by the patch-wise network that feeds into the image-wise network (default: 1)')
        self._parser = parser

    def parse(self, verbose=1):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()
        opt.debug = opt.debug != 0

        args = vars(opt)
        if verbose:
            print('\n------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------\n')

        return opt
