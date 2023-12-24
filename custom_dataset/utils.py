import os
import shutil

import gdown


def download_extract_celeba5(dataset_root):
    os.makedirs(dataset_root, exist_ok=True)
    gdown.download(id="1SJd_HkmGXoUNLcZ_VLZO2FauJ1Md8wGi", output=os.path.join(dataset_root, "data.zip"))
    shutil.unpack_archive(filename=os.path.join(dataset_root, "data.zip"),
                          extract_dir=dataset_root,
                          format="zip")
    os.remove(os.path.join(dataset_root, "data.zip"))
    shutil.unpack_archive(filename=os.path.join(dataset_root, "img_align_celeba.zip"),
                          extract_dir=dataset_root,
                          format="zip")
    os.remove(os.path.join(dataset_root, "img_align_celeba.zip"))


def check_celeba5(root):
    if not os.path.isdir(os.path.join(root, "img_align_celeba")):
        return False
    elif not os.path.isfile(os.path.join(root, "celebA_test_orig.txt")):
        return False
    elif not os.path.isfile(os.path.join(root, "celebA_train_orig.txt")):
        return False
    elif not os.path.isfile(os.path.join(root, "celebA_val_orig.txt")):
        return False
    else:
        return True
    

