import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset
from tqdm import tqdm
from score.fid import get_statistics
import os
import json
from torchvision import transforms
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
import torch
from score.inception import InceptionV3
from score.improved_prd import IPR
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


def generate_npz(dataset, name):
    imgs = list()
    for idx in tqdm(range(len(dataset))):
        img, _ = dataset[idx]
        imgs.append(np.array(img))
    print(f"Generating Statistics for {dataset.__class__.__name__} "
            f"for {len(dataset)} Images.")
    # imgs = np.stack(imgs, axis=0)
    # print(imgs.shape)
    mu, sigma = get_statistics(imgs)
    np.savez(os.path.join("stats", f"{name.lower()}.train"),
                mu=mu,
                sigma=sigma)

def perform_inference(dataset):
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx1]).to("cuda")
    model.eval()
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=16)
    embeddings = list()

    with torch.no_grad():
        for images, _ in tqdm(loader):
            out = model(images.to("cuda"))
            out = out[0].squeeze(axis=-1).squeeze(axis=-1)
            assert out.shape[-1] == 2048
            embeddings.append(out.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(f"cifar_original.npy", embeddings)
    print("Embeddings are saved...")

def gen_custom_stats(dataset_name, root, imb_factor=0.01, num_images=None):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(
                root=root,
                train=True,
                download=True,
                transform=gen_transform(32))
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(
                root=root,
                # root='...',
                train=True,
                download=True,
                transform=gen_transform(32))
    elif dataset_name == 'cifar10lt':
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
    elif dataset_name == 'cifar100lt':
        dataset = ImbalanceCIFAR100(
                root='/GPFS/data/yimingqin/dd_code/backdoor/benchmarks/pytorch-ddpm/data',
                # root='...',
                imb_type='exp',
                imb_factor=imb_factor,
                rand_number=0,
                train=True,
                transform=gen_transform(32),
                target_transform=None,
                download=True)
    elif dataset_name == "celeba-5":
        dataset_root = os.path.join(root, "CelebA")
        if not check_celeba5(dataset_root):
            download_extract_celeba5(dataset_root)
        dataset, _, _ = get_celeb_loader(data_root=dataset_root,
                                            transform_mode=gen_transform(celeb_img_size))
    elif dataset_name == "cub":
        dataset_root = os.path.join(root, "CUB")
        os.makedirs(dataset_root, exist_ok=True)
        dataset, _, _ = get_cub_loader(data_root=dataset_root,
                                       transform_mode=gen_transform(cub_img_size))
    elif dataset_name == "imagenet-lt":
        assert os.path.isdir(os.path.join(root, "images")), "ImageNet dataset cannot be automatically downloaded. Downlaod the dataset and create a folder called `images` in the root folder and copy all the images."
        if not os.path.isdir(os.path.join(root, "ImageNet_LT")):
            dataset_root = os.path.join(root, "ImageNet_LT")
            gdown.download_folder(id="19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-",
                                  output=dataset_root)
            assert os.path.isdir(dataset_root), "Invalid Download. Please Check."
        dataset, _, _ = load_data(data_root=os.path.join(root, "images"),
                                  dist_path=os.path.join(root, "ImageNet_LT"),
                                  phase="train",
                                  transform=gen_transform(imagenet_img_size),
                                  random_sampling=num_images
                                  )
    else:
        print('Please enter a data type included in [cifar10, cifar100, cifar10lt, cifar100lt]')
        exit(0)

    generate_npz(dataset, dataset_name)
    embeddings_creator = IPR(32, k=5, num_samples=len(dataset))
    manifold = embeddings_creator.compute_manifold(torch.stack([dataset[idx][0] for idx in range(len(dataset))], dim=0))
    # print('saving manifold to', fname, '...')
    os.makedirs("./embeddings", exist_ok=True)
    np.savez_compressed(os.path.join("./embeddings", f"{dataset_name.lower()}_feats"),
                        feature=manifold.features,
                        radii=manifold.radii)

    print("Finished..")


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    gen_custom_stats("cub", "./data")
    gen_custom_stats("celeba-5", "./data")
    gen_custom_stats("imagenet-lt", "/mnt/data1/imagenet_2012/images", num_images=50000)
    
