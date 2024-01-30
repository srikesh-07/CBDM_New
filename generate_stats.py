import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset
from tqdm import tqdm
from score.fid import get_statistics
import os
from itertools import combinations
import json
from torchvision import transforms
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
import torch
from score.inception import InceptionV3
from score.improved_prd import IPR
from itertools import combinations
from torchvision.datasets import CIFAR10, CIFAR100
from custom_dataset.celeba import get_celeb_loader, image_size as celeb_img_size
from custom_dataset.cub import get_cub_loader, image_size as cub_img_size
from custom_dataset.imagenet import load_data, image_size as imagenet_img_size
from custom_dataset.utils import *


def gen_transform(img_size):
    tran_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    return tran_transform


# def generate_npz(dataset, indices):
#     imgs = list()
#     for idx in indices:
#         img, _ = dataset[idx]
#         imgs.append(np.array(img))
#     # print(f"Generating Statistics for {dataset.__class__.__name__} "
#     #         f"for {len(dataset)} Images.")
#     # imgs = np.stack(imgs, axis=0)
#     # print(imgs.shape)
#     mu, sigma = get_statistics(imgs)
#     return mu, sigma
    # np.savez(os.path.join("stats", f"{name.lower()}.train"),
    #             mu=mu,
    #             sigma=sigma)


def get_classwise_indices(dataset):
    indices =  dict()
    for idx, (_, y) in enumerate(dataset):
        if indices.get(y, None) is None:
            indices[y] = [idx]
        else:
            indices[y].append(idx)
    return indices

def get_classwise_statistics(dataset, indices, save_dir=None):
    statistics = dict()

    for class_id, cls_indices in indices.items():
        imgs = list()
        for idx in cls_indices:
            img, _ = dataset[idx]
            imgs.append(np.array(img))
        mu, sigma = get_statistics(imgs)
        statistics[class_id] = {'mu': mu,
                                'sigma': sigma}
        
    if save_dir:
        np.save(os.path.join(save_dir, 'classwise_stats.npy'))

    return statistics



def custom_kldiv_mean_covarinace(dist_1, dist_2, return_transformed=True):
    k = len(dist_2['sigma'])
    sigma_2_inv = np.linalg.inv(dist_2['sigma'])
    trace = np.trace(np.dot(sigma_2_inv, dist_1['sigma']))
    mu_diff = dist_2['mu'] - dist_1['mu']
    diff_term = np.dot(mu_diff.T, np.dot(sigma_2_inv, mu_diff))
    det_term = np.log(np.linalg.det(dist_2['sigma']) / np.linalg.det(dist_1['sigma']))
    kl_div = 0.5 * (trace + diff_term - k + det_term)
    return kl_div


def class_pairwise_kldiv(classwise_stats):
    kldict = list()
    kldivs = list()
    for cls_id1, cls_id2 in tqdm(combinations(range(len(classwise_stats)), 2)):
        kl_div = round(custom_kldiv_mean_covarinace(classwise_stats[cls_id1], classwise_stats[cls_id2]), 3)
        new_kl_div = 1 / 1 + kl_div
        kldict.append({
            'dist_1': cls_id1,
            'dist_2': cls_id2,
            'kl_div': kl_div,
            'new_kl_div': new_kl_div
        })
        kldivs.append(new_kl_div)

        kl_div = round(custom_kldiv_mean_covarinace(classwise_stats[cls_id2], classwise_stats[cls_id1]), 3)
        new_kl_div = 1 / 1 + kl_div
        kldict.append({
            'dist_1': cls_id2,
            'dist_2': cls_id1,
            'kl_div': kl_div,
            'new_kl_div': 1 / (1 + kl_div)
        })
        kldivs.append(new_kl_div)

    kl_div_mean = np.mean(kldivs)

    return kldict, kl_div_mean


    

def gen_custom_stats(root, imb_factor=0.01, savedir=None):
    # if dataset_name == 'cifar10':
    #     dataset = CIFAR10(
    #             root=root,
    #             train=True,
    #             download=True,
    #             transform=gen_transform(32))
    # elif dataset_name == 'cifar100':
    #     dataset = CIFAR100(
    #             root=root,
    #             # root='...',
    #             train=True,
    #             download=True,
    #             transform=gen_transform(32))
    # elif dataset_name == 'cifar10lt':
    dataset = ImbalanceCIFAR10(
            root=root,
            # root='...',
            imb_type='exp',
            imb_factor=imb_factor,
            rand_number=0,
            train=True,
            transform=gen_transform(32),
            target_transform=None,
            download=True)
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
    print('[INFO] Obtaining Classwise Indices')
    classwise_indices = get_classwise_indices(dataset)
    print('[INFO] Generating Classwise Statistics')
    classwise_stats = get_classwise_statistics(dataset, classwise_indices, save_dir=savedir)
    print('[INFO] Calculating Class-pairwise KL-Divergence')
    classwise_kldiv, kl_div_mean = class_pairwise_kldiv(classwise_stats)

    if savedir:
        with open(os.path.join(savedir, 'classwise_kldiv.json'), 'w') as json_file:
            json.dump(classwise_kldiv, json_file)

    return kl_div_mean

    
    # elif dataset_name == 'cifar100lt':
    #     dataset = ImbalanceCIFAR100(
    #             root='/GPFS/data/yimingqin/dd_code/backdoor/benchmarks/pytorch-ddpm/data',
    #             # root='...',
    #             imb_type='exp',
    #             imb_factor=imb_factor,
    #             rand_number=0,
    #             train=True,
    #             transform=gen_transform(32),
    #             target_transform=None,
    #             download=True)
    # elif dataset_name == "celeba-5":
    #     dataset_root = os.path.join(root, "CelebA")
    #     if not check_celeba5(dataset_root):
    #         download_extract_celeba5(dataset_root)
    #     dataset, _, _ = get_celeb_loader(data_root=dataset_root,
    #                                         transform_mode=gen_transform(celeb_img_size))
    # elif dataset_name == "cub":
    #     dataset_root = os.path.join(root, "CUB")
    #     os.makedirs(dataset_root, exist_ok=True)
    #     dataset, _, _ = get_cub_loader(data_root=dataset_root,
    #                                    transform_mode=gen_transform(cub_img_size))
    # elif dataset_name == "imagenet-lt":
    #     assert os.path.isdir(os.path.join(root, "images")), "ImageNet dataset cannot be automatically downloaded. Downlaod the dataset and create a folder called `images` in the root folder and copy all the images."
    #     if not os.path.isdir(os.path.join(root, "ImageNet_LT")):
    #         dataset_root = os.path.join(root, "ImageNet_LT")
    #         gdown.download_folder(id="19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-",
    #                               output=dataset_root)
    #         assert os.path.isdir(dataset_root), "Invalid Download. Please Check."
    #     dataset, _, _ = load_data(data_root=os.path.join(root, "images"),
    #                               dist_path=os.path.join(root, "ImageNet_LT"),
    #                               phase="train",
    #                               transform=gen_transform(imagenet_img_size),
    #                               random_sampling=num_images
    #                               )
    # else:
    #     print('Please enter a data type included in [cifar10, cifar100, cifar10lt, cifar100lt]')
    #     exit(0)

    # generate_npz(dataset, dataset_name)



if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    gen_custom_stats('data', savedir="test_0.01")
    
