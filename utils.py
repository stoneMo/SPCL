# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import time
import random


import torch
from torch.utils.data import Dataset
import torch.nn as nn

from PIL import Image, ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def init_weights(m):
    '''Initialize weights with zeros
    '''
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SPCL, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, transform=None, transform_valid=None, target_transform=None, two_crop=False):

        if isinstance(data, list):
            data = np.array(data)
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        if isinstance(labels, np.ndarray):
            labels = torch.Tensor(labels)  

        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels

        self.compute_feature = True

        self.transform = transform
        self.transform_valid = transform_valid
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.data[index]

        if self.transform is not None:

            if self.transform_valid is not None and self.compute_feature:
                img = self.transform_valid(image)
            else:
                # Data augmentation and normalisation
                img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)
        # print("img:", img.shape)
        # when STL10 'unlabelled'
        if self.labels is None:
            return torch.Tensor([index]), img, torch.Tensor([0])
        else:
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.Tensor(self.labels) 

            return torch.Tensor([index]), img, self.labels[index].long()

class CustomDataset_metric(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SPCL, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, args, transform=None, transform_valid=None, target_transform=None, two_crop=False):

        if isinstance(data, list):
            data = np.array(data)
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        # if isinstance(labels, np.ndarray):
        #     labels = torch.Tensor(labels)  
        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels
        
        self.num_classes = len(np.unique(self.labels))
        # self.num_classes = args.num_prototypes

        self.data_dict = self.loadToMem()

        self.compute_feature = True

        self.transform = transform
        self.transform_valid = transform_valid
        self.target_transform = target_transform
        self.two_crop = two_crop
        
        self.pretrain = True

    def loadToMem(self):
        print("begin loading training dataset to memory")
        print("num_classes:", self.num_classes)
        data_dict = {}
        for n in range(self.num_classes):
            data_dict[n] = []
            for i in range(len(self.labels)):
                if self.labels[i] == n:
                    data_dict[n].append(i)
        print("finish loading training dataset to memory")
        return data_dict

    def get_image(self, index):
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.data[index]
        
        return image

    def get_transform_image(self, image):

        if self.transform is not None:
            
            if self.transform_valid is not None and self.compute_feature:
                img = self.transform_valid(image)
            else:
                # Data augmentation and normalisation
                img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)
        
        return img

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if self.two_crop and self.pretrain:
            label = None
            img1 = None
            img2 = None
            # get image from same class
            if index % 2 == 1:
                label = 1.0
                idx1 = random.randint(0, self.num_classes - 1)
                # while idx1 not in self.data_dict:
                #     idx1 = random.randint(0, self.num_classes - 1)
                while len(self.data_dict[idx1]) < 2:
                    idx1 = random.randint(0, self.num_classes - 1) 
                image1_index = random.choice(self.data_dict[idx1])
                image2_index = random.choice(self.data_dict[idx1])
            # get image from different class
            else:
                label = 0.0
                idx1 = random.randint(0, self.num_classes - 1)
                # while idx1 not in self.data_dict:
                #     idx1 = random.randint(0, self.num_classes - 1)
                while len(self.data_dict[idx1]) < 1:
                    idx1 = random.randint(0, self.num_classes - 1)

                idx2 = random.randint(0, self.num_classes - 1)
                # while idx2 not in self.data_dict:
                #     idx2 = random.randint(0, self.num_classes - 1)
                while len(self.data_dict[idx2]) < 1:
                    idx2 = random.randint(0, self.num_classes - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, self.num_classes - 1)
                image1_index = random.choice(self.data_dict[idx1])
                image2_index = random.choice(self.data_dict[idx2])

            image1_label = self.labels[image1_index]
            image2_label = self.labels[image2_index]
            images_label = torch.from_numpy(np.array([image1_label, image2_label])).long()

            image1 = self.get_image(image1_index)
            image2 = self.get_image(image2_index)

            img1 = self.get_transform_image(image1)
            img2 = self.get_transform_image(image2)

            similarity_label = torch.from_numpy(np.array([label], dtype=np.float32))

            return similarity_label, img1, img2, images_label
        
        else:

            image = self.get_image(index)
            img = self.get_transform_image(image)

            # print("img:", img.shape)
            # when STL10 'unlabelled'
            if self.labels is None:
                return torch.Tensor([index]), img, torch.Tensor([0])
            else:
                if isinstance(self.labels, np.ndarray):
                    self.labels = torch.Tensor(self.labels)  

                return torch.Tensor([index]), img, self.labels[index].long()

def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.

        Specifically for the image folder class
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    # label_list = np.unique(labels)
    # print("label_list:", label_list)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        # print("n_classes:", n_classes)
        # print("i:", i)
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        # print("c_idx:", c_idx)
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}

def split_pretrain_n_classes(data, labels, pre_classes, n_classes):
    """ Split given n classes from a pretrain set.
    """
    pretrain_x, pretrain_y = [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        if i < pre_classes:
            pretrain_x.extend(data[c_idx])
            pretrain_y.extend(labels[c_idx])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'pretrain': torch.stack(pretrain_x)}, \
            {'pretrain': torch.stack(pretrain_y)}
    # transforms list of np arrays to tensor
    return {'pretrain': torch.from_numpy(np.stack(pretrain_x))}, \
        {'pretrain': torch.from_numpy(np.stack(pretrain_y))}

def random_split_imagenet100(data, labels, pre_classes, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    pretrain_x, pretrain_y, train_x, train_y, valid_x, valid_y = [], [], [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    max_classes = max(pre_classes, n_classes)

    for i in range(max_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        if i < n_classes:
            train_x.extend(data[train_samples])
            train_y.extend(labels[train_samples])
            valid_x.extend(data[valid_samples])
            valid_y.extend(labels[valid_samples])
        if i < pre_classes:
            pretrain_x.extend(data[train_samples])
            pretrain_y.extend(labels[train_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'pretrain': pretrain_x, 'train': train_x, 'valid': valid_x}, \
            {'pretrain': torch.stack(pretrain_y), 'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'pretrain': pretrain_x,
            'train': train_x,
            'valid': valid_x}, \
        {'pretrain': torch.from_numpy(np.stack(pretrain_y)),
         'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}

def random_split_stl10(data, labels, pre_data, pre_labels, pre_classes, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    pretrain_x, pretrain_y, train_x, train_y, valid_x, valid_y = [], [], [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(pre_labels, list):
        pre_labels = np.array(pre_labels)

    max_classes = max(pre_classes, n_classes)

    for i in range(max_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        if i < n_classes:
            train_x.extend(data[train_samples])
            train_y.extend(labels[train_samples])
            valid_x.extend(data[valid_samples])
            valid_y.extend(labels[valid_samples])
            
        if i < pre_classes:
            pretrain_x.extend(data[train_samples])
            pretrain_y.extend(labels[train_samples])

            pre_idx = (np.array(pre_labels) == i).nonzero()[0]
            pretrain_x.extend(pre_data[pre_idx])
            pretrain_y.extend(pre_labels[pre_idx])

    if isinstance(data, torch.Tensor) or isinstance(pre_data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'pretrain': torch.stack(pretrain_x), 'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'pretrain': torch.stack(pretrain_y), 'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'pretrain': torch.from_numpy(np.stack(pretrain_x)),
            'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'pretrain': torch.from_numpy(np.stack(pretrain_y)),
         'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}

def random_split(data, labels, pre_classes, n_classes, train_samples_per_class, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    pretrain_x, pretrain_y, train_x, train_y, valid_x, valid_y = [], [], [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    max_classes = max(pre_classes, n_classes)

    for i in range(max_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # print("c_idx:", len(c_idx))
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        # train_samples = np.setdiff1d(c_idx, valid_samples)
        train_samples = np.setdiff1d(c_idx, valid_samples)[:train_samples_per_class[i]]
        # assign class c samples to validation, and remaining to training
        if i < n_classes:
            train_x.extend(data[train_samples])
            train_y.extend(labels[train_samples])
            valid_x.extend(data[valid_samples])
            valid_y.extend(labels[valid_samples])
        if i < pre_classes:
            pretrain_x.extend(data[train_samples])
            pretrain_y.extend(labels[train_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'pretrain': torch.stack(pretrain_x), 'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'pretrain': torch.stack(pretrain_y), 'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'pretrain': torch.from_numpy(np.stack(pretrain_x)),
            'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'pretrain': torch.from_numpy(np.stack(pretrain_y)),
         'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}

def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def experiment_config(parser, args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')

    os.makedirs(run_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    model_dir = os.path.join(run_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)


    args.miniExp_dir = os.path.join(model_dir, 'miniExp')
    args.figs_dir = os.path.join(model_dir, 'figs')
    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')

    if not args.finetune and not args.eval_disent:
        args.load_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.miniExp_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)
    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args


def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count