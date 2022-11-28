# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, STL10, ImageNet, CIFAR100, ImageFolder

from utils import *

def get_dataloaders(args):
    ''' Retrives the dataloaders for the dataset of choice.
    '''
    if args.dataset == 'cifar10':
        dataset = 'CIFAR10'

        args.class_names = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )  # 0,1,2,3,4,5,6,7,8,9 labels

        args.crop_dim = 32
        # args.n_channels, args.n_classes = 3, 10
        args.n_channels = 3

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        dataloaders = cifar_dataloader(args, dataset_paths)

    elif args.dataset == 'cifar100':
        dataset = 'CIFAR100'

        args.class_names = None

        args.crop_dim = 32
        args.n_channels, args.n_classes = 3, 100

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        dataloaders = cifar_dataloader(args, dataset_paths)

    elif args.dataset == 'stl10':
        dataset = 'STL10'

        args.class_names = (
            'airplane', 'bird', 'car', 'cat',
            'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'
        )  # 0,1,2,3,4,5,6,7,8,9 labels

        args.crop_dim = 96
        # args.n_channels, args.n_classes = 3, 10
        args.n_channels = 3

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test'),
                         'pretrain':  os.path.join(working_dir, 'unlabeled')}

        dataloaders = stl10_dataloader(args, dataset_paths)

    elif args.dataset == 'imagenet':
        dataset = 'ImageNet'

        args.class_names = None

        args.crop_dim = 224
        args.n_channels, args.n_classes = 3, 1000

        # Get and make dir to download dataset to.
        target_dir = args.dataset_path

        if not target_dir is None:
            dataset_paths = {'train': os.path.join(target_dir, 'train'),
                             'test':  os.path.join(target_dir, 'val')}

            dataloaders = imagenet_dataloader_ori(args, dataset_paths)

        else:
            NotImplementedError('Please Select a path for the {} Dataset.'.format(args.dataset))

    elif args.dataset == 'imagenet100':
        dataset = 'ImageNet'

        args.class_names = None

        args.crop_dim = 224
        args.n_channels, args.n_classes = 3, 100

        # Get and make dir to download dataset to.
        target_dir = args.dataset_path

        if not target_dir is None:
            dataset_paths = {'train': os.path.join(target_dir, 'train'),
                             'test':  os.path.join(target_dir, 'val')}

            dataloaders = imagenet100_dataloader(args, dataset_paths)

        else:
            NotImplementedError('Please Select a path for the {} Dataset.'.format(args.dataset))

    elif args.dataset == 'tinyimagenet':
        dataset = 'TinyImageNet'

        args.class_names = None

        args.crop_dim = 64
        args.n_channels, args.n_classes = 3, 200

        # Get and make dir to download dataset to.
        target_dir = args.dataset_path

        if not target_dir is None:
            dataset_paths = {'train': os.path.join(target_dir, 'train'),
                             'test':  os.path.join(target_dir, 'val')}

            dataloaders = imagenet_dataloader(args, dataset_paths)

        else:
            NotImplementedError('Please Select a path for the {} Dataset.'.format(args.dataset))
    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    return dataloaders, args

def imagenet_dataloader_ori(args, dataset_paths):
    ''' Loads the ImageNet dataset.
        Returns: train/valid/test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Base train and test augmentaions
    transf = {
        'pretrain': transforms.Compose([
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'train': transforms.Compose([
            # rnd_color_jitter,
            # rnd_grey,
            # guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
        'test':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
    }

    config = {'train': 'train', 'test': 'val'}

    # datasets = {i: ImageFolder(root=dataset_paths[i]) for i in config.keys()}
    datasets = {i: ImageNet(root=dataset_paths[i], split=config[i]) for i in config.keys()}


    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    # 50 is the num of samples to be split into the val set for each class (1000)
    data, labels = random_split_image_folder(data=np.asarray(datasets['train'].samples),
                                             labels=datasets['train'].targets,
                                             n_classes=args.n_classes,
                                             n_samples_per_class=np.repeat(50, args.n_classes).reshape(-1))

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    # original full training set
    datasets['train_valid'] = CustomDataset(data=np.asarray(datasets['train'].samples),
                                            labels=torch.from_numpy(np.stack(datasets['train'].targets)), transform=transf['pretrain'], two_crop=args.twocrop)

    # original test set
    datasets['test'] = CustomDataset(data=np.asarray(datasets['test'].samples),
                                     labels=torch.from_numpy(np.stack(datasets['test'].targets)), transform=transf['test'], two_crop=False)

    # make new pretraining set without validation samples
    # datasets['pretrain'] = CustomDataset(data=np.asarray(data['train']),
    #                                      labels=labels['train'], transform=transf['pretrain'], two_crop=args.twocrop)
    if args.metric_learn:
        datasets['pretrain'] = CustomDataset_metric(data=np.asarray(data['train']), 
                                                    labels=labels['train'], args=args, transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)
    else:
        datasets['pretrain'] = CustomDataset(data=np.asarray(data['train']),
                                            labels=labels['train'], transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)
    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=np.asarray(data['train']),
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=np.asarray(data['valid']),
                                      labels=labels['valid'], transform=transf['valid'], two_crop=False)

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['pretrain'].labels)

    config = {
        'pretrain': WeightedRandomSampler(s_weights,
                                          num_samples=len(s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=False,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

def imagenet100_dataloader(args, dataset_paths):
    ''' Loads the ImageNet dataset.
        Returns: train/valid/test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Base train and test augmentaions
    transf = {
        'pretrain': transforms.Compose([
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
            ]),
        'train': transforms.Compose([
            # rnd_color_jitter,
            # rnd_grey,
            # guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
        'test':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
    }

    config = {'train': 'train', 'test': 'val'}

    # datasets = {i: ImageFolder(root=dataset_paths[i]) for i in config.keys()}

    datasets = {i: ImageNet(root=dataset_paths[i], split=config[i]) for i in config.keys()}

    val_samples = 50

    print("Original Dataset")
    print('train:', len(datasets['train']))
    print('test:', len(datasets['test']))
    # print('pretrain:', len(datasets['pretrain']))

    # acquire 100 classes
    for i in config.keys():
        if i == 'train' or 'test':
            idx = torch.tensor(datasets[i].targets) == 0
            for c in range(1, args.n_classes): 
                idx += torch.tensor(datasets[i].targets) == c
            datasets[i] = MySubset(datasets[i], np.where(idx==1)[0], dataset_type='imagenet')
    
    print("ImageNet-100 Dataset")
    print('train:', len(datasets['train']))
    print('test:', len(datasets['test']))

    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    # 500 is the num of samples to be split into the val set for each class (10)
    data, labels = random_split_imagenet100(data=datasets['train'].data,
                                labels=datasets['train'].targets,
                                pre_classes=args.pre_classes,
                                n_classes=args.n_classes,
                                n_samples_per_class=np.repeat(val_samples, max(args.pre_classes,args.n_classes)).reshape(-1))

    # save original full training set
    datasets['train_valid'] = CustomDataset(data=datasets['train'].data,
                                            labels=torch.from_numpy(np.stack(datasets['train'].targets)), transform=transf['pretrain'], two_crop=args.twocrop)

    # make new pretraining set without validation samples
    # datasets['pretrain'] = CustomDataset(data=data['pretrain'],
    #                                      labels=labels['pretrain'], transform=transf['pretrain'], two_crop=args.twocrop)
    if args.metric_learn:
        datasets['pretrain'] = CustomDataset_metric(data=data['pretrain'],
                                            labels=labels['pretrain'], args=args, transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)
    else:
        datasets['pretrain'] = CustomDataset(data=data['pretrain'],
                                            labels=labels['pretrain'], transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['test'] = CustomDataset(data=datasets['test'].data,
                                      labels=datasets['test'].targets, transform=transf['test'], two_crop=False)

    print("Dataset of {} classes used in the pretrain stage".format(args.pre_classes))
    print('pretrain:', len(datasets['pretrain']))
    print("Dataset of {} classes used in the finetune stage".format(args.n_classes))
    print('train:', len(datasets['train']))
    print('valid:', len(datasets['valid']))
    print('test:', len(datasets['test']))
    # print('pretrain:', len(datasets['pretrain']))

    # weighted sampler weights for new training set
    s_weights = sample_weights(labels['train'])
    pre_s_weights = sample_weights(labels['pretrain'])

    config = {
        'pretrain': WeightedRandomSampler(pre_s_weights,
                                          num_samples=len(pre_s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=False,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

    # # weighted sampler weights for full(f) training set
    # f_s_weights = sample_weights(datasets['train'].targets)

    # # return data, labels dicts for new train set and class-balanced valid set
    # # 50 is the num of samples to be split into the val set for each class (1000)
    # data, labels = random_split_image_folder(data=np.asarray(datasets['train'].samples),
    #                                          labels=datasets['train'].targets,
    #                                          n_classes=args.n_classes,
    #                                          n_samples_per_class=np.repeat(50, args.n_classes).reshape(-1))

    # # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    # # original full training set
    # datasets['train_valid'] = CustomDataset(data=np.asarray(datasets['train'].samples),
    #                                         labels=torch.from_numpy(np.stack(datasets['train'].targets)), transform=transf['pretrain'], two_crop=args.twocrop)

    # # original test set
    # datasets['test'] = CustomDataset(data=np.asarray(datasets['test'].samples),
    #                                  labels=torch.from_numpy(np.stack(datasets['test'].targets)), transform=transf['test'], two_crop=False)

    # # make new pretraining set without validation samples
    # datasets['pretrain'] = CustomDataset(data=np.asarray(data['train']),
    #                                      labels=labels['train'], transform=transf['pretrain'], two_crop=args.twocrop)

    # # make new finetuning set without validation samples
    # datasets['train'] = CustomDataset(data=np.asarray(data['train']),
    #                                   labels=labels['train'], transform=transf['train'], two_crop=False)

    # # make class balanced validation set for finetuning
    # datasets['valid'] = CustomDataset(data=np.asarray(data['valid']),
    #                                   labels=labels['valid'], transform=transf['valid'], two_crop=False)

    # # weighted sampler weights for new training set
    # s_weights = sample_weights(datasets['pretrain'].labels)

    # config = {
    #     'pretrain': WeightedRandomSampler(s_weights,
    #                                       num_samples=len(s_weights), replacement=True),
    #     'train': WeightedRandomSampler(s_weights,
    #                                    num_samples=len(s_weights), replacement=True),
    #     'train_valid': WeightedRandomSampler(f_s_weights,
    #                                          num_samples=len(f_s_weights), replacement=True),
    #     'valid': None, 'test': None
    # }

    # if args.distributed:
    #     config = {'pretrain': DistributedSampler(datasets['pretrain']),
    #               'train': DistributedSampler(datasets['train']),
    #               'train_valid': DistributedSampler(datasets['train_valid']),
    #               'valid': None, 'test': None}

    # dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
    #                              num_workers=8, pin_memory=True, drop_last=True,
    #                              batch_size=args.batch_size) for i in config.keys()}

    # return dataloaders

class MySubset(Dataset):

    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, dataset_type='stl10'):
        self.dataset = dataset
        self.indices = indices
        print('self.indices:', self.indices.shape)
        if dataset_type == 'cifar':
            self.data = dataset.data
            print('self.dataset.targets:', np.array(self.dataset.targets).shape)
            self.targets = np.array(self.dataset.targets)[self.indices]
        elif dataset_type == 'imagenet':
            self.data = np.array(self.dataset.samples)[self.indices]
            print('self.dataset.targets:', np.array(self.dataset.targets).shape)
            self.targets = np.array(self.dataset.targets)[self.indices]
        else:
            print('self.dataset.labels:', self.dataset.labels.shape)
            self.labels = self.dataset.labels[self.indices]

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        return image, target

    def __len__(self):
        return len(self.indices)

def stl10_dataloader(args, dataset_paths):
    ''' Loads the STL10 dataset.
        Returns: train/valid/test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Base train and test augmentaions
    transf = {
        'pretrain': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            # guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'train': transforms.Compose([
            transforms.ToPILImage(),
            # rnd_color_jitter,
            # rnd_grey,
            # guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid':  transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
        'test':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
    }

    transf['pretrain'] = transf['train']

    config = {'train': 'train', 'test': 'test', 'pretrain': 'unlabeled'}

    datasets = {i: STL10(root=dataset_paths[i], transform=transf[i],
                         split=config[i], download=True) for i in config.keys()}

    print("Original Dataset")
    print('train:', len(datasets['train']))
    print('test:', len(datasets['test']))
    print('pretrain:', len(datasets['pretrain']))

    for i in config.keys():
        if i == 'test':
            idx = torch.tensor(datasets[i].labels) == 0
            for c in range(1, args.n_classes): 
                idx += torch.tensor(datasets[i].labels) == c
            datasets[i] = MySubset(datasets[i], np.where(idx==1)[0])
    
    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].labels)

    # return data, labels dicts for new train set and class-balanced valid set
    # 50 is the num of samples to be split into the val set for each class (10)
    # data, labels = random_split(data=datasets['train'].data,
    #                             labels=datasets['train'].labels,
    #                             pre_classes=args.n_classes,
    #                             n_classes=args.n_classes,
    #                             n_samples_per_class=np.repeat(50, args.n_classes).reshape(-1))

    pretrain_labels = np.load('./pretrain_pseudo_labels.npy')

    data, labels = random_split_stl10(data=datasets['train'].data,
                                labels=datasets['train'].labels,
                                pre_data=datasets['pretrain'].data,
                                pre_labels=pretrain_labels,
                                pre_classes=args.pre_classes,
                                n_classes=args.n_classes,
                                n_samples_per_class=np.repeat(50, max(args.pre_classes,args.n_classes)).reshape(-1))
    
    # pretrain_data, _ = split_pretrain_n_classes(data=datasets['pretrain'].data,
    #                                 labels=pretrain_labels,
    #                                 pre_classes=args.pre_classes,
    #                                 n_classes=args.n_classes)

    # save original full training set
    datasets['train_valid'] = datasets['train'].data

    # make new pretraining set without validation samples
    datasets['pretrain'] = CustomDataset(data=data['pretrain'],
                                         labels=None, transform=transf['pretrain'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'], two_crop=False)

    print("Dataset of {} classes used in the pretrain stage".format(args.pre_classes))
    print('pretrain:', len(datasets['pretrain']))
    print("Dataset of {} classes used in the finetune stage".format(args.n_classes))
    print('train:', len(datasets['train']))
    print('valid:', len(datasets['valid']))
    print('test:', len(datasets['test']))

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['train'].labels)

    config = {
        'pretrain': None,
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = dict()
    for i in config.keys():
        if i == 'pretrain':
            dataloaders[i] = DataLoader(datasets[i], sampler=config[i],
                            num_workers=8, pin_memory=True, drop_last=True,
                            batch_size=args.batch_size)
        else:
            dataloaders[i] = DataLoader(datasets[i], sampler=config[i],
                            num_workers=8, pin_memory=True, drop_last=False,
                            batch_size=args.batch_size)
            # drop_last=True in the original paper
    return dataloaders


def cifar_dataloader(args, dataset_paths):
    ''' Loads the CIFAR-10 and CIFAR-100 dataset.
        Returns: train/valid/test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])

    # Base train and test augmentaions
    transf = {
        'pretrain': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'train': transforms.Compose([
            transforms.ToPILImage(),
            # rnd_color_jitter,
            # rnd_grey,
            # guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid':  transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize]),
        'test':  transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim * 0.875, args.crop_dim * 0.875)),
            transforms.Resize((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            normalize])
    }

    config = {'train': True, 'test': False}

    if args.dataset == 'cifar10':

        datasets = {i: CIFAR10(root=dataset_paths[i], transform=transf[i],
                               train=config[i], download=True) for i in config.keys()}
        
        train_samples = args.train_sample
        
        val_samples = args.val_sample


    elif args.dataset == 'cifar100':

        datasets = {i: CIFAR100(root=dataset_paths[i], transform=transf[i],
                                train=config[i], download=True) for i in config.keys()}

        train_samples = args.train_sample
        
        val_samples = args.val_sample

    print("Original Dataset")
    print('train:', len(datasets['train']))
    print('test:', len(datasets['test']))
    # print('pretrain:', len(datasets['pretrain']))

    for i in config.keys():
        if i == 'test':
            idx = torch.tensor(datasets[i].targets) == 0
            for c in range(1, args.n_classes): 
                idx += torch.tensor(datasets[i].targets) == c
            datasets[i] = MySubset(datasets[i], np.where(idx==1)[0], dataset_type='cifar')
    
    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    # 500 is the num of samples to be split into the val set for each class (10)
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].targets,
                                pre_classes=args.pre_classes,
                                n_classes=args.n_classes,
                                train_samples_per_class=np.repeat(train_samples, args.n_classes).reshape(-1),
                                n_samples_per_class=np.repeat(val_samples, max(args.pre_classes,args.n_classes)).reshape(-1))

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new pretraining set without validation samples
    if args.metric_learn:
        datasets['pretrain'] = CustomDataset_metric(data=data['pretrain'],
                                            labels=labels['pretrain'], args=args, transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)
    else:
        datasets['pretrain'] = CustomDataset(data=data['pretrain'],
                                            labels=labels['pretrain'], transform=transf['pretrain'], transform_valid=transf['valid'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['test'] = CustomDataset(data=datasets['test'].data,
                                      labels=datasets['test'].targets, transform=transf['test'], two_crop=False)

    print("Dataset of {} classes used in the pretrain stage".format(args.pre_classes))
    print('pretrain:', len(datasets['pretrain']))
    print("Dataset of {} classes used in the finetune stage".format(args.n_classes))
    print('train:', len(datasets['train']))
    print('valid:', len(datasets['valid']))
    print('test:', len(datasets['test']))
    # print('pretrain:', len(datasets['pretrain']))

    # weighted sampler weights for new training set
    s_weights = sample_weights(labels['train'])
    pre_s_weights = sample_weights(labels['pretrain'])

    config = {
        'pretrain': WeightedRandomSampler(pre_s_weights,
                                          num_samples=len(pre_s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=False,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders
