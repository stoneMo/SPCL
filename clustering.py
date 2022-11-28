import faiss
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix, find
from torch.utils import data
from torchvision import transforms
from torch.utils.data.sampler import Sampler

from utils import *

def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

def run_kmeans(x, nmb_clusters, args, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    # losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]

def cluster_assign(images_lists, dataset, args):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    image_indexes = sorted(image_indexes)
    indexes = np.argsort(image_indexes)
    pseudolabels = np.asarray(pseudolabels)[indexes]

    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])
    t = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.008, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    if args.metric_learn:
        return ReassignedDataset_metric(image_indexes, pseudolabels, dataset, t)
    else:
        return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

        print("set(pseudolabels):", set(pseudolabels))

        self.two_crop = True

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            image = dataset.data[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((image, pseudolabel))
        return images

    def get_image(self, index):

        if isinstance(self.imgs[index][0], np.str_):
            # Load image from path
            image = Image.open(self.imgs[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.imgs[index][0]
        
        return image

    def get_transform_image(self, image):

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        # if self.target_transform is not None:

        #     # Transforms the target, i.e. object detection, segmentation
        #     target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)
        
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        image = self.get_image(index)
        img = self.get_transform_image(image)

        # print("img:", img.shape)

        pseudo_label = self.imgs[index][1]

        # print("pseudo_label:", pseudo_label)

        if pseudo_label is None:
            return torch.Tensor([index]), img, torch.Tensor([0])
        else:
            if not isinstance(pseudo_label, torch.Tensor):
                pseudo_label = torch.Tensor([pseudo_label])  

            return torch.Tensor([index]), img, pseudo_label.long()

    def __len__(self):
        return len(self.imgs)

class ReassignedDataset_metric(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        
        self.transform = transform

        self.num_classes = len(set(pseudolabels))

        self.data_dict = self.loadToMem()

        self.two_crop = True

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            image = dataset.data[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((image, pseudolabel))
        return images

    def loadToMem(self):
        print("begin loading training dataset to memory")
        data_dict = {}
        for n in range(self.num_classes):
            data_dict[n] = []
            for i in range(len(self.imgs)):
                if self.imgs[i][1] == n:
                    data_dict[n].append(i)
        print("finish loading training dataset to memory")
        return data_dict

    def get_image(self, index):

        if isinstance(self.imgs[index][0], np.str_):
            # Load image from path
            image = Image.open(self.imgs[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.imgs[index][0]
        
        return image

    def get_transform_image(self, image):

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        # if self.target_transform is not None:

        #     # Transforms the target, i.e. object detection, segmentation
        #     target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)
        
        return img

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if self.two_crop:

            label = None
            img1 = None
            img2 = None
            # get image from same class
            if index % 2 == 1:
                label = 1.0
                idx1 = random.randint(0, self.num_classes - 1)
                image1_index = random.choice(self.data_dict[idx1])
                image2_index = random.choice(self.data_dict[idx1])
            # get image from different class
            else:
                label = 0.0
                idx1 = random.randint(0, self.num_classes - 1)
                idx2 = random.randint(0, self.num_classes - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, self.num_classes - 1)
                image1_index = random.choice(self.data_dict[idx1])
                image2_index = random.choice(self.data_dict[idx2])

            image1_label = self.imgs[image1_index][1]
            image2_label = self.imgs[image2_index][1]
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

            pseudo_label = self.imgs[index][1]

            # print("pseudo_label:", pseudo_label)

            if pseudo_label is None:
                return torch.Tensor([index]), img, torch.Tensor([0])
            else:
                if not isinstance(pseudo_label, torch.Tensor):
                    pseudo_label = torch.Tensor([pseudo_label])  

                return torch.Tensor([index]), img, pseudo_label.long()

    def __len__(self):
        return len(self.imgs)

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)