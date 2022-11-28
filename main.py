#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from train import finetune, evaluate, pretrain, supervised, eval_disent, pretrain_metric, pretrain_cluster, pretrain_cluster_metric
from datasets import get_dataloaders
from utils import experiment_config, print_network, init_weights
import model.network as models


warnings.filterwarnings("ignore")

default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch SPCL', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--n_classes', type=int, default=10,
                    help='Number of classes in Contrastive Training.')
parser.add_argument('--pre_classes', type=int, default=10,
                    help='Number of classes samples used in the pretrain stage.')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--finetune_learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='lars',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--finetune_optimiser', default='sgd',
                    help='Finetune Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--temperature', type=float, default=0.5, help='NT_Xent Temperature Factor')
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--no_distributed', dest='distributed', action='store_false',
                    help='Whether or Not to Use Distributed Training. (Default: True)')
parser.set_defaults(distributed=True)
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
parser.set_defaults(finetune=False)
parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')
parser.add_argument('--gpu_id', type=str, default='0,1,2,3',
                    help='Perform Supervised Pre-Training. (Default: False)')
parser.add_argument('--proj_dim', type=int, default=128,
                    help='Number of Samples Per Batch.')
parser.set_defaults(supervised=False)
parser.add_argument('--dip_proj', dest='dip_proj', action='store_true',
                    help='Perform DIP loss after projection or not. (Default: False)')
parser.set_defaults(dip_proj=False)
parser.add_argument('--dip_weight', type=float, default=1e-2,
                    help='DIP weight Factor.')

parser.add_argument('--contras_weight', type=float, default=1.0,
                    help='Contrastive learning weight Factor.')

parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--num_prototypes', type=int, default=512,
                    help='Number of prototypes used in Kmeans.')

parser.add_argument('--clustering_use', dest='clustering_use', action='store_true',
                    help='Perform clustering to generate pseudo label or not. (Default: False)')
parser.set_defaults(clustering_use=False)
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Verbose or not. (Default: False)')
parser.set_defaults(verbose=False)
parser.add_argument('--cluster_weight', type=float, default=1e-2,
                    help='cluster weight Factor.')
parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")

parser.add_argument('--cluster_loss', type=str, choices=['CE', 'SCE'],
                        default='CE', help='clustering loss type (default: CE)')
parser.add_argument('--metric_loss', type=str, choices=['BCE', 'SCE', 'L1', 'L2'],
                        default='BCE', help='metric loss type (default: BCE)')
parser.add_argument('--classifier_init_freq', type=int, default=1,
                    help='Frequency epoch of initializing the classifier during pretrain stage.')


# visualization 
parser.add_argument('--visual_cls', dest='visual_cls', action='store_true',
                    help='Visual_cls or not. (Default: False)')
parser.add_argument('--visual_freq', type=int, default=1,
                    help='Frequency epoch of visualization during pretrain stage.')
                    
parser.add_argument('--train_sample', type=int, default=100, help='number of train samples per class')
parser.add_argument('--val_sample', type=int, default=20, help='number of valid samples per class')


parser.add_argument('--eval_disent', dest='eval_disent', action='store_true',
                    help='Evaluate disentanglement or not. (Default: False)')
parser.set_defaults(eval_disent=False)


parser.add_argument('--metric_learn', dest='metric_learn', action='store_true',
                    help='metric_learn or not. (Default: False)')
parser.set_defaults(metric_learn=False)
parser.add_argument('--metric_weight', type=float, default=1e-2,
                    help='Metric learning weight Factor.')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Debug or not. (Default: False)')
parser.set_defaults(debug=False)


def setup(distributed):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed

    Taken from https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate

    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 420
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank


def main():
    """ Main """

    # Arguments
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Setup Distributed Training
    device, local_rank = setup(distributed=args.distributed)

    # Get Dataloaders for Dataset of choice
    dataloaders, args = get_dataloaders(args)

    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    # Get available models from /model/network.py
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # If model exists
    if any(args.model in model_name for model_name in model_names):

        # Load model
        base_encoder = getattr(models, args.model)(
            args, num_classes=args.n_classes)  # Encoder

        proj_head = models.projection_MLP(args)
        sup_head = models.Sup_Head(args)

    else:
        raise NotImplementedError("Model Not Implemented: {}".format(args.model))

    # Remove last FC layer from resnet
    base_encoder.fc = nn.Sequential()

    # Place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(6)  # n cpu threads / n processes per node

        base_encoder = DistributedDataParallel(base_encoder.cuda(),
                                               device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        proj_head = DistributedDataParallel(proj_head.cuda(),
                                            device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        sup_head = DistributedDataParallel(sup_head.cuda(),
                                           device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        # Only print from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        # If non Distributed use DataParallel
        if torch.cuda.device_count() > 1:
            base_encoder = nn.DataParallel(base_encoder)
            proj_head = nn.DataParallel(proj_head)
            sup_head = nn.DataParallel(sup_head)

        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

        base_encoder.to(device)
        proj_head.to(device)
        sup_head.to(device)

        args.print_progress = True

    # Print Network Structure and Params
    if args.print_progress:
        print_network(base_encoder, args)  # prints out the network architecture etc
        logging.info('\npretrain: {} - train: {} - valid: {} - test: {}'.format(
            len(dataloaders['pretrain'].dataset), len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    # launch model training or inference
    if not args.finetune and not args.eval_disent:

        ''' Pretraining / Finetuning / Evaluate '''

        if not args.supervised:
            # Pretrain the encoder and projection head
            proj_head.apply(init_weights)

            if args.metric_learn and not args.clustering_use:
                pretrain_metric(base_encoder, proj_head, dataloaders, args)

            elif args.clustering_use and not args.metric_learn:
                pretrain_cluster(base_encoder, proj_head, dataloaders, args)

            elif args.metric_learn and args.clustering_use:
                pretrain_cluster_metric(base_encoder, proj_head, dataloaders, args)

            else:
                pretrain(base_encoder, proj_head, dataloaders, args)
        else:
            supervised(base_encoder, sup_head, dataloaders, args)

        print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

        # Load the pretrained model
        checkpoint = torch.load(args.load_checkpoint_dir)

        # Load the encoder parameters
        base_encoder.load_state_dict(checkpoint['encoder'])

        # Initalize weights of the supervised / classification head
        sup_head.apply(init_weights)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, sup_head, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_topk = evaluate(
            base_encoder, sup_head, dataloaders, 'test', args.finetune_epochs, args)

       # top ceil(n_class/2) accuracy 
        top_k = math.ceil(args.n_classes / 2)
        if top_k > 5:
            top_k = 5
        print('[Test] num_class {} - loss {:.4f} - acc {:.4f} - acc_top_{} {:.4f}'.format(
            args.n_classes, test_loss, test_acc, top_k, test_acc_topk))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()

    elif not args.finetune and args.eval_disent:

        ''' Evaluating the disentanglement of the learned representation '''

        # Do not Pretrain, just finetune and inference

        print("load_checkpoint_dir:", args.load_checkpoint_dir)

        print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

        # Load the pretrained model
        checkpoint = torch.load(args.load_checkpoint_dir)

        # Load the encoder parameters
        base_encoder.load_state_dict(checkpoint['encoder'])  # .cuda()

        # Evaluating the disentanglement of the learned representation
        eval_disent(base_encoder, dataloaders, args)

        # # Evaluate the pretrained model and trained supervised head
        # test_loss, test_acc, test_acc_topk = evaluate(
        #     base_encoder, sup_head, dataloaders, 'test', args.finetune_epochs, args)

        # # top ceil(n_class/2) accuracy 
        # top_k = math.ceil(args.n_classes / 2)
        # print('[Test] num_class {} - loss {:.4f} - acc {:.4f} - acc_top_{} {:.4f}'.format(
        #     args.n_classes, test_loss, test_acc, top_k, test_acc_topk))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()

    else:

        ''' Finetuning / Evaluate '''

        # Do not Pretrain, just finetune and inference

        print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

        # Load the pretrained model
        checkpoint = torch.load(args.load_checkpoint_dir)

        # Load the encoder parameters
        base_encoder.load_state_dict(checkpoint['encoder'])  # .cuda()

        # Initalize weights of the supervised / classification head
        sup_head.apply(init_weights)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, sup_head, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_topk = evaluate(
            base_encoder, sup_head, dataloaders, 'test', args.finetune_epochs, args)

        # top ceil(n_class/2) accuracy 
        top_k = math.ceil(args.n_classes / 2)
        if top_k > 5:
            top_k = 5
        print('[Test] num_class {} - loss {:.4f} - acc {:.4f} - acc_top_{} {:.4f}'.format(
            args.n_classes, test_loss, test_acc, top_k, test_acc_topk))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
