# Instructions

## 1. Run `python generate_stats.py` to generate FID statistics as well as Embeddings for Precision and Recall.

## Note:
### (Only for ImageNet-LT)
1. Download the training dataset - [ILSVRC-2012](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar)
2. Create a folder `images/train` in root directory and extract the tarballs under the folder.
3. Specify the root directory by `--root` as an argument during the training.

## 2. Run below the any of the commands, (DDPM)

## Celeb-A5 (Tested in Colab and Training starts successfully. Post-debugging needs to be done if any error occurs between epochs.)
```
python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/celeba5_cbdm --total_steps 300001 \
        --conditional \
        --data_type celeba-5 --img_size 32 \
        --batch_size 48 --save_step 100000 --sample_step 50000 \
        --cb --tau 0.05 --omega 1.0
```

## CUB (Tested in Colab and Training starts successfully. Post-debugging needs to be done if any error occurs between epochs.)
```
python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/CUB_cbdm --total_steps 300001 \
        --conditional \
        --data_type cub --img_size 32 \
        --batch_size 48 --save_step 100000 --sample_step 50000 \
        --cb --tau 0.1 --omega 0.4
```

## ImageNet-LT (Not Tested due to Datset Size)
```
python main.py --train  \
        --flagfile ./config/cifar100.txt --parallel \
        --logdir ./logs/imagenet-lt_cbdm --total_steps 300001 \
        --conditional \
        --data_type imagenet-lt --img_size 32 \
        --batch_size 48 --save_step 100000 --sample_step 50000 \
        --cb --tau 0.01 --omega 1.6
```
